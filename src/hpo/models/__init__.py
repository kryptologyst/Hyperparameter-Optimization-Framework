"""Model definitions for hyperparameter optimization experiments."""

import logging
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """Simple CNN model for image classification."""
    
    def __init__(self, n_classes: int = 10, n_filters: int = 32, dropout: float = 0.5):
        """Initialize SimpleCNN.
        
        Args:
            n_classes: Number of output classes
            n_filters: Number of filters in first conv layer
            dropout: Dropout rate
        """
        super().__init__()
        
        self.conv1 = nn.Conv2d(3, n_filters, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(n_filters, n_filters * 2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(n_filters * 2, n_filters * 4, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout)
        
        # Calculate flattened size
        self.flattened_size = n_filters * 4 * 4 * 4  # 4x4 after 3 pooling layers
        
        self.fc1 = nn.Linear(self.flattened_size, 128)
        self.fc2 = nn.Linear(128, n_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        x = x.view(-1, self.flattened_size)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        
        return x


class MLP(nn.Module):
    """Multi-layer perceptron for tabular data."""
    
    def __init__(self, input_size: int, hidden_sizes: list, n_classes: int = 2, 
                 dropout: float = 0.5, activation: str = "relu"):
        """Initialize MLP.
        
        Args:
            input_size: Input feature size
            hidden_sizes: List of hidden layer sizes
            n_classes: Number of output classes
            dropout: Dropout rate
            activation: Activation function ('relu', 'tanh', 'sigmoid')
        """
        super().__init__()
        
        self.layers = nn.ModuleList()
        
        # Input layer
        prev_size = input_size
        
        # Hidden layers
        for hidden_size in hidden_sizes:
            self.layers.append(nn.Linear(prev_size, hidden_size))
            prev_size = hidden_size
        
        # Output layer
        self.layers.append(nn.Linear(prev_size, n_classes))
        
        self.dropout = nn.Dropout(dropout)
        
        # Activation function
        if activation == "relu":
            self.activation = F.relu
        elif activation == "tanh":
            self.activation = torch.tanh
        elif activation == "sigmoid":
            self.activation = torch.sigmoid
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        for i, layer in enumerate(self.layers[:-1]):
            x = self.activation(layer(x))
            x = self.dropout(x)
        
        x = self.layers[-1](x)
        return x


class ResNetBlock(nn.Module):
    """Residual block for ResNet."""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        """Initialize ResNet block."""
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        residual = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        out += self.shortcut(residual)
        out = F.relu(out)
        
        return out


class ResNet(nn.Module):
    """ResNet model for image classification."""
    
    def __init__(self, n_classes: int = 10, n_blocks: int = 2, base_channels: int = 64):
        """Initialize ResNet.
        
        Args:
            n_classes: Number of output classes
            n_blocks: Number of residual blocks per layer
            base_channels: Number of base channels
        """
        super().__init__()
        
        self.in_channels = base_channels
        
        # Initial convolution
        self.conv1 = nn.Conv2d(3, base_channels, kernel_size=3, 
                              stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(base_channels)
        
        # Residual layers
        self.layer1 = self._make_layer(base_channels, n_blocks, stride=1)
        self.layer2 = self._make_layer(base_channels * 2, n_blocks, stride=2)
        self.layer3 = self._make_layer(base_channels * 4, n_blocks, stride=2)
        
        # Global average pooling and classifier
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(base_channels * 4, n_classes)
    
    def _make_layer(self, out_channels: int, n_blocks: int, stride: int) -> nn.Module:
        """Make a layer with residual blocks."""
        layers = []
        
        # First block might need to downsample
        layers.append(ResNetBlock(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        
        # Remaining blocks
        for _ in range(1, n_blocks):
            layers.append(ResNetBlock(self.in_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = F.relu(self.bn1(self.conv1(x)))
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x


def create_model(model_name: str, **kwargs) -> nn.Module:
    """Create a model by name.
    
    Args:
        model_name: Name of model to create
        **kwargs: Model-specific arguments
        
    Returns:
        nn.Module: Created model
    """
    models = {
        "simple_cnn": SimpleCNN,
        "mlp": MLP,
        "resnet": ResNet
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}")
    
    model = models[model_name](**kwargs)
    logging.info(f"Created {model_name} model with {sum(p.numel() for p in model.parameters())} parameters")
    
    return model


def get_model_info(model_name: str) -> Dict[str, Any]:
    """Get information about a model.
    
    Args:
        model_name: Name of model
        
    Returns:
        Dict containing model information
    """
    info = {
        "simple_cnn": {
            "description": "Simple CNN with 3 conv layers",
            "hyperparameters": ["n_filters", "dropout"],
            "suitable_for": ["cifar10"]
        },
        "mlp": {
            "description": "Multi-layer perceptron",
            "hyperparameters": ["hidden_sizes", "dropout", "activation"],
            "suitable_for": ["synthetic"]
        },
        "resnet": {
            "description": "ResNet with residual connections",
            "hyperparameters": ["n_blocks", "base_channels"],
            "suitable_for": ["cifar10"]
        }
    }
    
    return info.get(model_name, {})
