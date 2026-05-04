"""Main hyperparameter optimization framework."""

import logging
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import yaml
import torch
from omegaconf import DictConfig, OmegaConf

from .utils import set_seed, get_device, setup_logging, save_config
from .data import get_data_loaders, get_dataset_info
from .models import create_model, get_model_info
from .algorithms import create_hpo_algorithm
from .train import train_model
from .eval import evaluate_model, evaluate_hpo_results, create_leaderboard


class HPOFramework:
    """Main hyperparameter optimization framework."""
    
    def __init__(self, config: DictConfig):
        """Initialize HPO framework.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.device = get_device()
        
        # Setup logging
        setup_logging(
            level=config.get('logging_level', 'INFO'),
            log_file=config.get('log_file', None)
        )
        
        # Set random seed
        set_seed(config.get('seed', 42))
        
        # Initialize results storage
        self.results = {}
        self.best_models = {}
        
        logging.info("HPO Framework initialized")
        logging.info(f"Device: {self.device}")
        logging.info(f"Configuration: {OmegaConf.to_yaml(config)}")
    
    def run_experiment(self, experiment_name: str) -> Dict[str, Any]:
        """Run a complete HPO experiment.
        
        Args:
            experiment_name: Name of experiment configuration
            
        Returns:
            Dict containing experiment results
        """
        if experiment_name not in self.config.experiments:
            raise ValueError(f"Unknown experiment: {experiment_name}")
        
        exp_config = self.config.experiments[experiment_name]
        logging.info(f"Starting experiment: {experiment_name}")
        
        # Load data
        train_loader, val_loader, test_loader = get_data_loaders(**exp_config.data)
        
        # Get dataset info
        dataset_info = get_dataset_info(exp_config.data.dataset_name)
        
        # Run HPO algorithms
        algorithm_results = {}
        
        for algorithm_name in exp_config.algorithms:
            logging.info(f"Running algorithm: {algorithm_name}")
            
            # Create HPO algorithm
            hpo_algorithm = create_hpo_algorithm(
                algorithm_name,
                **exp_config.algorithms[algorithm_name]
            )
            
            # Define objective function
            def objective(params):
                return self._objective_function(
                    params, exp_config, train_loader, val_loader, dataset_info
                )
            
            # Run optimization
            start_time = time.time()
            best_params = hpo_algorithm.optimize(objective, exp_config.search_space)
            optimization_time = time.time() - start_time
            
            # Store results
            algorithm_results[algorithm_name] = {
                'best_params': best_params,
                'trial_results': hpo_algorithm.get_trial_results(),
                'optimization_time': optimization_time,
                'best_score': hpo_algorithm.best_score
            }
            
            logging.info(f"Algorithm {algorithm_name} completed in {optimization_time:.2f}s")
            logging.info(f"Best score: {hpo_algorithm.best_score:.4f}")
        
        # Evaluate best models on test set
        test_results = {}
        for algorithm_name, results in algorithm_results.items():
            logging.info(f"Evaluating best model from {algorithm_name} on test set")
            
            # Create and train best model
            best_model = self._create_and_train_model(
                results['best_params'], exp_config, train_loader, val_loader, dataset_info
            )
            
            # Evaluate on test set
            test_metrics = evaluate_model(best_model, test_loader, self.device, dataset_info['task_type'])
            test_results[algorithm_name] = test_metrics
            
            # Store best model
            self.best_models[algorithm_name] = best_model
        
        # Create leaderboard
        leaderboard = create_leaderboard(
            {name: results['trial_results'] for name, results in algorithm_results.items()}
        )
        
        # Compile final results
        experiment_results = {
            'experiment_name': experiment_name,
            'config': exp_config,
            'algorithm_results': algorithm_results,
            'test_results': test_results,
            'leaderboard': leaderboard,
            'dataset_info': dataset_info
        }
        
        self.results[experiment_name] = experiment_results
        
        # Save results
        self._save_results(experiment_name)
        
        logging.info(f"Experiment {experiment_name} completed")
        return experiment_results
    
    def _objective_function(
        self,
        params: Dict[str, Any],
        exp_config: DictConfig,
        train_loader,
        val_loader,
        dataset_info: Dict[str, Any]
    ) -> float:
        """Objective function for HPO optimization.
        
        Args:
            params: Hyperparameters to evaluate
            exp_config: Experiment configuration
            train_loader: Training data loader
            val_loader: Validation data loader
            dataset_info: Dataset information
            
        Returns:
            float: Validation score
        """
        try:
            # Create model
            model_params = {k: v for k, v in params.items() if k in ['n_filters', 'dropout', 'hidden_sizes', 'activation', 'n_blocks', 'base_channels']}
            model_params.update({
                'n_classes': dataset_info['n_classes'],
                'input_size': dataset_info['input_shape'][0] if len(dataset_info['input_shape']) == 1 else None
            })
            
            model = create_model(exp_config.model_name, **model_params)
            
            # Add training parameters
            train_params = params.copy()
            train_params['task_type'] = dataset_info['task_type']
            
            # Train model
            val_score, _ = train_model(
                model, train_loader, val_loader, train_params, self.device,
                max_epochs=exp_config.get('max_epochs', 50),
                early_stopping_patience=exp_config.get('early_stopping_patience', 10),
                verbose=False
            )
            
            return val_score
            
        except Exception as e:
            logging.warning(f"Objective function failed with params {params}: {e}")
            return float('-inf') if exp_config.get('direction', 'maximize') == 'maximize' else float('inf')
    
    def _create_and_train_model(
        self,
        params: Dict[str, Any],
        exp_config: DictConfig,
        train_loader,
        val_loader,
        dataset_info: Dict[str, Any]
    ) -> torch.nn.Module:
        """Create and train a model with given parameters.
        
        Args:
            params: Model and training parameters
            exp_config: Experiment configuration
            train_loader: Training data loader
            val_loader: Validation data loader
            dataset_info: Dataset information
            
        Returns:
            Trained model
        """
        # Create model
        model_params = {k: v for k, v in params.items() if k in ['n_filters', 'dropout', 'hidden_sizes', 'activation', 'n_blocks', 'base_channels']}
        model_params.update({
            'n_classes': dataset_info['n_classes'],
            'input_size': dataset_info['input_shape'][0] if len(dataset_info['input_shape']) == 1 else None
        })
        
        model = create_model(exp_config.model_name, **model_params)
        
        # Add training parameters
        train_params = params.copy()
        train_params['task_type'] = dataset_info['task_type']
        
        # Train model
        train_model(
            model, train_loader, val_loader, train_params, self.device,
            max_epochs=exp_config.get('max_epochs', 100),
            early_stopping_patience=exp_config.get('early_stopping_patience', 10),
            verbose=False
        )
        
        return model
    
    def _save_results(self, experiment_name: str) -> None:
        """Save experiment results.
        
        Args:
            experiment_name: Name of experiment
        """
        results_dir = Path(self.config.get('results_dir', './results'))
        results_dir.mkdir(exist_ok=True)
        
        # Save configuration
        config_path = results_dir / f"{experiment_name}_config.yaml"
        save_config(self.config.experiments[experiment_name], str(config_path))
        
        # Save results summary
        results_summary = {
            'experiment_name': experiment_name,
            'leaderboard': self.results[experiment_name]['leaderboard'],
            'test_results': self.results[experiment_name]['test_results']
        }
        
        results_path = results_dir / f"{experiment_name}_results.yaml"
        with open(results_path, 'w') as f:
            yaml.dump(results_summary, f, default_flow_style=False)
        
        logging.info(f"Results saved to {results_dir}")
    
    def run_ablation_study(self, experiment_name: str, ablation_configs: Dict[str, DictConfig]) -> Dict[str, Any]:
        """Run ablation study.
        
        Args:
            experiment_name: Base experiment name
            ablation_configs: Ablation configurations
            
        Returns:
            Dict containing ablation results
        """
        logging.info(f"Starting ablation study for experiment: {experiment_name}")
        
        # Run base experiment
        base_results = self.run_experiment(experiment_name)
        
        # Run ablation experiments
        ablation_results = {}
        for ablation_name, ablation_config in ablation_configs.items():
            logging.info(f"Running ablation: {ablation_name}")
            
            # Create modified experiment config
            modified_config = self.config.experiments[experiment_name].copy()
            OmegaConf.set_struct(modified_config, False)
            
            # Apply ablation modifications
            for key, value in ablation_config.items():
                OmegaConf.set(modified_config, key, value)
            
            # Temporarily store modified config
            temp_exp_name = f"{experiment_name}_{ablation_name}"
            self.config.experiments[temp_exp_name] = modified_config
            
            # Run experiment
            ablation_results[ablation_name] = self.run_experiment(temp_exp_name)
            
            # Clean up
            del self.config.experiments[temp_exp_name]
        
        # Analyze ablation results
        from .eval import compute_ablation_results
        
        ablation_analysis = compute_ablation_results(
            base_results['algorithm_results']['random_search']['trial_results'],
            {name: results['algorithm_results']['random_search']['trial_results'] 
             for name, results in ablation_results.items()}
        )
        
        return {
            'base_results': base_results,
            'ablation_results': ablation_results,
            'ablation_analysis': ablation_analysis
        }
