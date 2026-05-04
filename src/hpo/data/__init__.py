"""Data loading and preprocessing utilities for hyperparameter optimization."""

import logging
from pathlib import Path
from typing import Dict, Tuple, Optional, Any
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms
from sklearn.datasets import make_classification, make_regression
from sklearn.preprocessing import StandardScaler


class CIFAR10Dataset(Dataset):
    """CIFAR-10 dataset wrapper for HPO experiments."""
    
    def __init__(self, root: str = "./data", train: bool = True, transform: Optional[Any] = None):
        """Initialize CIFAR-10 dataset.
        
        Args:
            root: Root directory for dataset
            train: Whether to use training set
            transform: Optional transforms to apply
        """
        self.dataset = datasets.CIFAR10(
            root=root, 
            train=train, 
            download=True, 
            transform=transform
        )
        
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return self.dataset[idx]


class SyntheticDataset(Dataset):
    """Synthetic dataset for testing HPO algorithms."""
    
    def __init__(self, n_samples: int = 1000, n_features: int = 20, n_classes: int = 2, 
                 task_type: str = "classification", noise: float = 0.1):
        """Initialize synthetic dataset.
        
        Args:
            n_samples: Number of samples
            n_features: Number of features
            n_classes: Number of classes (for classification)
            task_type: Type of task ('classification' or 'regression')
            noise: Amount of noise to add
        """
        self.task_type = task_type
        
        if task_type == "classification":
            X, y = make_classification(
                n_samples=n_samples,
                n_features=n_features,
                n_classes=n_classes,
                n_redundant=0,
                n_informative=n_features,
                random_state=42
            )
            # Convert to PyTorch tensors
            self.X = torch.FloatTensor(X)
            self.y = torch.LongTensor(y)
        else:  # regression
            X, y = make_regression(
                n_samples=n_samples,
                n_features=n_features,
                noise=noise,
                random_state=42
            )
            # Normalize features
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
            
            self.X = torch.FloatTensor(X)
            self.y = torch.FloatTensor(y).unsqueeze(1)
    
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def get_data_loaders(
    dataset_name: str = "cifar10",
    batch_size: int = 32,
    num_workers: int = 4,
    val_split: float = 0.2,
    **kwargs
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Get data loaders for training, validation, and testing.
    
    Args:
        dataset_name: Name of dataset to load
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes
        val_split: Fraction of training data to use for validation
        **kwargs: Additional arguments for dataset creation
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    logging.info(f"Loading dataset: {dataset_name}")
    
    if dataset_name == "cifar10":
        # Define transforms
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        # Load datasets
        train_dataset = CIFAR10Dataset(root="./data", train=True, transform=train_transform)
        test_dataset = CIFAR10Dataset(root="./data", train=False, transform=test_transform)
        
        # Split training data into train/val
        val_size = int(len(train_dataset) * val_split)
        train_size = len(train_dataset) - val_size
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
        
        # Apply transforms to validation set
        val_dataset.dataset.transform = test_transform
        
    elif dataset_name == "synthetic":
        # Create synthetic datasets
        train_dataset = SyntheticDataset(
            n_samples=kwargs.get('n_samples', 1000),
            n_features=kwargs.get('n_features', 20),
            n_classes=kwargs.get('n_classes', 2),
            task_type=kwargs.get('task_type', 'classification')
        )
        
        val_dataset = SyntheticDataset(
            n_samples=kwargs.get('n_samples', 200),
            n_features=kwargs.get('n_features', 20),
            n_classes=kwargs.get('n_classes', 2),
            task_type=kwargs.get('task_type', 'classification')
        )
        
        test_dataset = SyntheticDataset(
            n_samples=kwargs.get('n_samples', 200),
            n_features=kwargs.get('n_features', 20),
            n_classes=kwargs.get('n_classes', 2),
            task_type=kwargs.get('task_type', 'classification')
        )
        
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    logging.info(f"Train samples: {len(train_dataset)}")
    logging.info(f"Val samples: {len(val_dataset)}")
    logging.info(f"Test samples: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader


def get_dataset_info(dataset_name: str) -> Dict[str, Any]:
    """Get information about a dataset.
    
    Args:
        dataset_name: Name of dataset
        
    Returns:
        Dict containing dataset information
    """
    info = {
        "cifar10": {
            "n_classes": 10,
            "input_shape": (3, 32, 32),
            "task_type": "classification",
            "description": "CIFAR-10 image classification dataset"
        },
        "synthetic": {
            "n_classes": 2,  # Default
            "input_shape": (20,),  # Default
            "task_type": "classification",  # Default
            "description": "Synthetic dataset for testing"
        }
    }
    
    return info.get(dataset_name, {})
