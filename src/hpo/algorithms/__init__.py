"""Hyperparameter optimization algorithms and strategies."""

import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import optuna
from optuna.samplers import TPESampler, RandomSampler, CmaEsSampler
from optuna.pruners import MedianPruner, SuccessiveHalvingPruner, HyperbandPruner
import torch
from torch.utils.data import DataLoader

from ..utils import EarlyStopping, get_device, count_parameters
from ..models import create_model
from ..train import train_model
from ..eval import evaluate_model


class BaseHPO(ABC):
    """Base class for hyperparameter optimization algorithms."""
    
    def __init__(self, n_trials: int = 100, timeout: Optional[int] = None, 
                 direction: str = "maximize", pruner: Optional[Any] = None):
        """Initialize HPO algorithm.
        
        Args:
            n_trials: Number of trials to run
            timeout: Maximum time in seconds
            direction: Optimization direction ('maximize' or 'minimize')
            pruner: Optional pruner for early stopping trials
        """
        self.n_trials = n_trials
        self.timeout = timeout
        self.direction = direction
        self.pruner = pruner
        self.best_params = None
        self.best_score = None
        self.trial_results = []
        
    @abstractmethod
    def optimize(self, objective_func: callable, search_space: Dict[str, Any]) -> Dict[str, Any]:
        """Run hyperparameter optimization.
        
        Args:
            objective_func: Function to optimize
            search_space: Search space definition
            
        Returns:
            Dict containing best parameters
        """
        pass
    
    def get_trial_results(self) -> List[Dict[str, Any]]:
        """Get results from all trials."""
        return self.trial_results


class RandomSearch(BaseHPO):
    """Random search hyperparameter optimization."""
    
    def optimize(self, objective_func: callable, search_space: Dict[str, Any]) -> Dict[str, Any]:
        """Run random search optimization."""
        logging.info(f"Starting random search with {self.n_trials} trials")
        
        best_score = float('-inf') if self.direction == "maximize" else float('inf')
        best_params = None
        
        for trial_idx in range(self.n_trials):
            # Sample random parameters
            params = self._sample_params(search_space)
            
            # Evaluate objective
            start_time = time.time()
            score = objective_func(params)
            eval_time = time.time() - start_time
            
            # Track results
            trial_result = {
                'trial': trial_idx,
                'params': params.copy(),
                'score': score,
                'eval_time': eval_time
            }
            self.trial_results.append(trial_result)
            
            # Update best
            if self.direction == "maximize":
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
            else:
                if score < best_score:
                    best_score = score
                    best_params = params.copy()
            
            logging.info(f"Trial {trial_idx}: score={score:.4f}, time={eval_time:.2f}s")
        
        self.best_score = best_score
        self.best_params = best_params
        
        logging.info(f"Random search completed. Best score: {best_score:.4f}")
        return best_params
    
    def _sample_params(self, search_space: Dict[str, Any]) -> Dict[str, Any]:
        """Sample parameters from search space."""
        params = {}
        for param_name, param_config in search_space.items():
            if isinstance(param_config, list):
                params[param_name] = np.random.choice(param_config)
            elif isinstance(param_config, dict):
                if param_config['type'] == 'uniform':
                    params[param_name] = np.random.uniform(
                        param_config['low'], param_config['high']
                    )
                elif param_config['type'] == 'loguniform':
                    params[param_name] = np.exp(np.random.uniform(
                        np.log(param_config['low']), np.log(param_config['high'])
                    ))
                elif param_config['type'] == 'int':
                    params[param_name] = np.random.randint(
                        param_config['low'], param_config['high'] + 1
                    )
                elif param_config['type'] == 'categorical':
                    params[param_name] = np.random.choice(param_config['choices'])
        
        return params


class GridSearch(BaseHPO):
    """Grid search hyperparameter optimization."""
    
    def optimize(self, objective_func: callable, search_space: Dict[str, Any]) -> Dict[str, Any]:
        """Run grid search optimization."""
        logging.info("Starting grid search")
        
        # Generate all parameter combinations
        param_combinations = self._generate_combinations(search_space)
        n_combinations = len(param_combinations)
        
        logging.info(f"Total combinations to evaluate: {n_combinations}")
        
        best_score = float('-inf') if self.direction == "maximize" else float('inf')
        best_params = None
        
        for trial_idx, params in enumerate(param_combinations):
            # Evaluate objective
            start_time = time.time()
            score = objective_func(params)
            eval_time = time.time() - start_time
            
            # Track results
            trial_result = {
                'trial': trial_idx,
                'params': params.copy(),
                'score': score,
                'eval_time': eval_time
            }
            self.trial_results.append(trial_result)
            
            # Update best
            if self.direction == "maximize":
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
            else:
                if score < best_score:
                    best_score = score
                    best_params = params.copy()
            
            logging.info(f"Trial {trial_idx}/{n_combinations}: score={score:.4f}, time={eval_time:.2f}s")
        
        self.best_score = best_score
        self.best_params = best_params
        
        logging.info(f"Grid search completed. Best score: {best_score:.4f}")
        return best_params
    
    def _generate_combinations(self, search_space: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate all parameter combinations."""
        import itertools
        
        param_names = list(search_space.keys())
        param_values = []
        
        for param_name in param_names:
            param_config = search_space[param_name]
            if isinstance(param_config, list):
                param_values.append(param_config)
            elif isinstance(param_config, dict) and param_config['type'] == 'categorical':
                param_values.append(param_config['choices'])
            else:
                raise ValueError(f"Grid search only supports discrete parameters. "
                               f"Parameter '{param_name}' is continuous.")
        
        combinations = []
        for combination in itertools.product(*param_values):
            params = dict(zip(param_names, combination))
            combinations.append(params)
        
        return combinations


class OptunaHPO(BaseHPO):
    """Optuna-based hyperparameter optimization with multiple samplers."""
    
    def __init__(self, n_trials: int = 100, timeout: Optional[int] = None,
                 direction: str = "maximize", sampler: str = "tpe", 
                 pruner: str = "median", **kwargs):
        """Initialize Optuna HPO.
        
        Args:
            n_trials: Number of trials
            timeout: Maximum time in seconds
            direction: Optimization direction
            sampler: Sampler type ('tpe', 'random', 'cmaes')
            pruner: Pruner type ('median', 'hyperband', 'successive_halving')
            **kwargs: Additional arguments
        """
        super().__init__(n_trials, timeout, direction)
        
        # Setup sampler
        if sampler == "tpe":
            self.sampler = TPESampler(**kwargs.get('sampler_kwargs', {}))
        elif sampler == "random":
            self.sampler = RandomSampler(**kwargs.get('sampler_kwargs', {}))
        elif sampler == "cmaes":
            self.sampler = CmaEsSampler(**kwargs.get('sampler_kwargs', {}))
        else:
            raise ValueError(f"Unknown sampler: {sampler}")
        
        # Setup pruner
        if pruner == "median":
            self.pruner = MedianPruner(**kwargs.get('pruner_kwargs', {}))
        elif pruner == "hyperband":
            self.pruner = HyperbandPruner(**kwargs.get('pruner_kwargs', {}))
        elif pruner == "successive_halving":
            self.pruner = SuccessiveHalvingPruner(**kwargs.get('pruner_kwargs', {}))
        else:
            self.pruner = None
    
    def optimize(self, objective_func: callable, search_space: Dict[str, Any]) -> Dict[str, Any]:
        """Run Optuna optimization."""
        logging.info(f"Starting Optuna optimization with {self.sampler.__class__.__name__} sampler")
        
        # Create study
        study = optuna.create_study(
            direction=self.direction,
            sampler=self.sampler,
            pruner=self.pruner
        )
        
        # Define objective function for Optuna
        def optuna_objective(trial):
            params = self._suggest_params(trial, search_space)
            return objective_func(params)
        
        # Run optimization
        study.optimize(
            optuna_objective,
            n_trials=self.n_trials,
            timeout=self.timeout
        )
        
        # Extract results
        self.best_score = study.best_value
        self.best_params = study.best_params
        
        # Convert trial results
        for trial in study.trials:
            trial_result = {
                'trial': trial.number,
                'params': trial.params.copy(),
                'score': trial.value,
                'eval_time': trial.duration.total_seconds() if trial.duration else 0,
                'state': trial.state.name
            }
            self.trial_results.append(trial_result)
        
        logging.info(f"Optuna optimization completed. Best score: {self.best_score:.4f}")
        return self.best_params
    
    def _suggest_params(self, trial: optuna.Trial, search_space: Dict[str, Any]) -> Dict[str, Any]:
        """Suggest parameters using Optuna trial."""
        params = {}
        
        for param_name, param_config in search_space.items():
            if isinstance(param_config, list):
                params[param_name] = trial.suggest_categorical(param_name, param_config)
            elif isinstance(param_config, dict):
                if param_config['type'] == 'uniform':
                    params[param_name] = trial.suggest_float(
                        param_name, param_config['low'], param_config['high']
                    )
                elif param_config['type'] == 'loguniform':
                    params[param_name] = trial.suggest_float(
                        param_name, param_config['low'], param_config['high'], log=True
                    )
                elif param_config['type'] == 'int':
                    params[param_name] = trial.suggest_int(
                        param_name, param_config['low'], param_config['high']
                    )
                elif param_config['type'] == 'categorical':
                    params[param_name] = trial.suggest_categorical(
                        param_name, param_config['choices']
                    )
        
        return params


class MultiFidelityHPO(BaseHPO):
    """Multi-fidelity hyperparameter optimization using Hyperband."""
    
    def __init__(self, max_budget: int = 100, eta: int = 3, min_budget: int = 1,
                 direction: str = "maximize"):
        """Initialize multi-fidelity HPO.
        
        Args:
            max_budget: Maximum budget (e.g., epochs)
            eta: Reduction factor for successive halving
            min_budget: Minimum budget
            direction: Optimization direction
        """
        super().__init__(direction=direction)
        self.max_budget = max_budget
        self.eta = eta
        self.min_budget = min_budget
        self.pruner = HyperbandPruner(min_resource=min_budget, max_resource=max_budget, reduction_factor=eta)
    
    def optimize(self, objective_func: callable, search_space: Dict[str, Any]) -> Dict[str, Any]:
        """Run multi-fidelity optimization."""
        logging.info(f"Starting multi-fidelity optimization (Hyperband)")
        
        # Use Optuna with Hyperband pruner
        optuna_hpo = OptunaHPO(
            n_trials=self.n_trials,
            timeout=self.timeout,
            direction=self.direction,
            sampler="tpe",
            pruner="hyperband",
            pruner_kwargs={
                'min_resource': self.min_budget,
                'max_resource': self.max_budget,
                'reduction_factor': self.eta
            }
        )
        
        # Modify objective to support pruning
        def multi_fidelity_objective(params):
            return objective_func(params, budget=self.max_budget, pruner=self.pruner)
        
        return optuna_hpo.optimize(multi_fidelity_objective, search_space)


def create_hpo_algorithm(algorithm: str, **kwargs) -> BaseHPO:
    """Create HPO algorithm by name.
    
    Args:
        algorithm: Name of algorithm
        **kwargs: Algorithm-specific arguments
        
    Returns:
        BaseHPO: Created HPO algorithm
    """
    algorithms = {
        "random_search": RandomSearch,
        "grid_search": GridSearch,
        "optuna_tpe": lambda **kw: OptunaHPO(sampler="tpe", **kw),
        "optuna_random": lambda **kw: OptunaHPO(sampler="random", **kw),
        "optuna_cmaes": lambda **kw: OptunaHPO(sampler="cmaes", **kw),
        "hyperband": MultiFidelityHPO,
        "bohb": lambda **kw: OptunaHPO(sampler="tpe", pruner="hyperband", **kw)
    }
    
    if algorithm not in algorithms:
        raise ValueError(f"Unknown HPO algorithm: {algorithm}")
    
    return algorithms[algorithm](**kwargs)
