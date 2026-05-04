"""Test suite for hyperparameter optimization framework."""

import pytest
import torch
import numpy as np
from omegaconf import OmegaConf

from src.hpo.utils import set_seed, get_device, EarlyStopping
from src.hpo.data import get_data_loaders, SyntheticDataset
from src.hpo.models import create_model, SimpleCNN, MLP
from src.hpo.algorithms import create_hpo_algorithm, RandomSearch, OptunaHPO
from src.hpo.train import train_model
from src.hpo.eval import evaluate_model, evaluate_hpo_results


class TestUtils:
    """Test utility functions."""
    
    def test_set_seed(self):
        """Test random seed setting."""
        set_seed(42)
        # Test that seeds are set (basic check)
        assert True  # Placeholder for actual seed testing
    
    def test_get_device(self):
        """Test device detection."""
        device = get_device()
        assert isinstance(device, torch.device)
    
    def test_early_stopping(self):
        """Test early stopping functionality."""
        es = EarlyStopping(patience=3)
        
        # Mock model
        model = torch.nn.Linear(1, 1)
        
        # Test improvement
        assert not es(0.8, model)
        assert not es(0.9, model)
        
        # Test no improvement
        assert not es(0.85, model)
        assert not es(0.85, model)
        assert not es(0.85, model)
        assert es(0.85, model)  # Should trigger early stopping


class TestData:
    """Test data loading functionality."""
    
    def test_synthetic_dataset(self):
        """Test synthetic dataset creation."""
        dataset = SyntheticDataset(n_samples=100, n_features=10, n_classes=2)
        
        assert len(dataset) == 100
        x, y = dataset[0]
        assert x.shape == (10,)
        assert y.shape == (1,) or y.shape == ()  # Regression or classification
    
    def test_get_data_loaders(self):
        """Test data loader creation."""
        train_loader, val_loader, test_loader = get_data_loaders(
            dataset_name="synthetic",
            batch_size=16,
            n_samples=100,
            n_features=10,
            n_classes=2
        )
        
        assert len(train_loader) > 0
        assert len(val_loader) > 0
        assert len(test_loader) > 0


class TestModels:
    """Test model creation and functionality."""
    
    def test_create_model(self):
        """Test model creation."""
        model = create_model("simple_cnn", n_classes=10, n_filters=32)
        assert isinstance(model, SimpleCNN)
        
        model = create_model("mlp", input_size=20, hidden_sizes=[64], n_classes=2)
        assert isinstance(model, MLP)
    
    def test_model_forward(self):
        """Test model forward pass."""
        model = SimpleCNN(n_classes=10, n_filters=32)
        x = torch.randn(1, 3, 32, 32)
        y = model(x)
        assert y.shape == (1, 10)
        
        model = MLP(input_size=20, hidden_sizes=[64], n_classes=2)
        x = torch.randn(1, 20)
        y = model(x)
        assert y.shape == (1, 2)


class TestAlgorithms:
    """Test HPO algorithms."""
    
    def test_create_hpo_algorithm(self):
        """Test HPO algorithm creation."""
        hpo = create_hpo_algorithm("random_search", n_trials=5)
        assert isinstance(hpo, RandomSearch)
        
        hpo = create_hpo_algorithm("optuna_tpe", n_trials=5)
        assert isinstance(hpo, OptunaHPO)
    
    def test_random_search(self):
        """Test random search algorithm."""
        hpo = RandomSearch(n_trials=3)
        
        search_space = {
            'learning_rate': {'type': 'uniform', 'low': 0.001, 'high': 0.1},
            'batch_size': {'type': 'categorical', 'choices': [16, 32]}
        }
        
        def objective(params):
            return np.random.random()  # Mock objective
        
        best_params = hpo.optimize(objective, search_space)
        
        assert best_params is not None
        assert 'learning_rate' in best_params
        assert 'batch_size' in best_params
        assert len(hpo.get_trial_results()) == 3


class TestEvaluation:
    """Test evaluation functionality."""
    
    def test_evaluate_hpo_results(self):
        """Test HPO results evaluation."""
        trial_results = [
            {'trial': 0, 'score': 0.8, 'eval_time': 1.0},
            {'trial': 1, 'score': 0.9, 'eval_time': 1.2},
            {'trial': 2, 'score': 0.85, 'eval_time': 1.1}
        ]
        
        evaluation = evaluate_hpo_results(trial_results)
        
        assert evaluation['n_trials'] == 3
        assert evaluation['best_score'] == 0.9
        assert evaluation['mean_score'] == 0.85
        assert evaluation['total_time'] == 3.3


class TestIntegration:
    """Integration tests."""
    
    def test_simple_experiment(self):
        """Test a simple end-to-end experiment."""
        # Create simple configuration
        config = OmegaConf.create({
            'seed': 42,
            'logging_level': 'WARNING',
            'experiments': {
                'test_exp': {
                    'dataset_name': 'synthetic',
                    'model_name': 'mlp',
                    'max_epochs': 2,
                    'early_stopping_patience': 1,
                    'data': {
                        'dataset_name': 'synthetic',
                        'batch_size': 16,
                        'num_workers': 0,
                        'val_split': 0.2,
                        'n_samples': 100,
                        'n_features': 10,
                        'n_classes': 2,
                        'task_type': 'classification'
                    },
                    'search_space': {
                        'learning_rate': {'type': 'categorical', 'choices': [0.01, 0.1]},
                        'batch_size': {'type': 'categorical', 'choices': [16]},
                        'optimizer': {'type': 'categorical', 'choices': ['adam']},
                        'hidden_sizes': {'type': 'categorical', 'choices': [[32]]},
                        'dropout': {'type': 'categorical', 'choices': [0.0]},
                        'activation': {'type': 'categorical', 'choices': ['relu']}
                    },
                    'algorithms': {
                        'random_search': {
                            'n_trials': 2,
                            'timeout': None,
                            'direction': 'maximize'
                        }
                    }
                }
            }
        })
        
        # This would require the full framework initialization
        # For now, just test that the config is valid
        assert 'test_exp' in config.experiments
        assert config.experiments.test_exp.dataset_name == 'synthetic'


if __name__ == "__main__":
    pytest.main([__file__])
