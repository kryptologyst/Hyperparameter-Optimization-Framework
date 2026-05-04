"""Training utilities for hyperparameter optimization."""

import logging
import time
from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..utils import EarlyStopping, get_device, count_parameters


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    params: Dict[str, Any],
    device: torch.device,
    max_epochs: int = 100,
    early_stopping_patience: int = 10,
    verbose: bool = True
) -> Tuple[float, Dict[str, Any]]:
    """Train a model with given hyperparameters.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        params: Hyperparameters
        device: Device to train on
        max_epochs: Maximum number of epochs
        early_stopping_patience: Early stopping patience
        verbose: Whether to print progress
        
    Returns:
        Tuple of (best_val_score, training_history)
    """
    # Setup optimizer
    optimizer_name = params.get('optimizer', 'adam').lower()
    learning_rate = params.get('learning_rate', 0.001)
    weight_decay = params.get('weight_decay', 0.0)
    
    if optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == 'sgd':
        momentum = params.get('momentum', 0.9)
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_name == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    # Setup loss function
    task_type = params.get('task_type', 'classification')
    if task_type == 'classification':
        criterion = nn.CrossEntropyLoss()
    elif task_type == 'regression':
        criterion = nn.MSELoss()
    else:
        raise ValueError(f"Unknown task type: {task_type}")
    
    # Setup scheduler
    scheduler_name = params.get('scheduler', None)
    if scheduler_name == 'step':
        step_size = params.get('step_size', 30)
        gamma = params.get('gamma', 0.1)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_name == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    elif scheduler_name == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5)
    else:
        scheduler = None
    
    # Setup early stopping
    early_stopping = EarlyStopping(patience=early_stopping_patience)
    
    # Move model to device
    model = model.to(device)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'learning_rate': []
    }
    
    best_val_score = float('-inf')
    
    if verbose:
        logging.info(f"Training model with {count_parameters(model)} parameters")
        logging.info(f"Optimizer: {optimizer_name}, LR: {learning_rate}")
    
    for epoch in range(max_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{max_epochs}', disable=not verbose)
        
        for batch_idx, (data, target) in enumerate(train_pbar):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            
            if task_type == 'classification':
                loss = criterion(output, target)
                pred = output.argmax(dim=1, keepdim=True)
                train_correct += pred.eq(target.view_as(pred)).sum().item()
            else:  # regression
                loss = criterion(output, target)
                pred = output
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_total += target.size(0)
            
            # Update progress bar
            if verbose:
                train_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100. * train_correct / train_total:.2f}%'
                })
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                
                if task_type == 'classification':
                    loss = criterion(output, target)
                    pred = output.argmax(dim=1, keepdim=True)
                    val_correct += pred.eq(target.view_as(pred)).sum().item()
                else:  # regression
                    loss = criterion(output, target)
                    pred = output
                
                val_loss += loss.item()
                val_total += target.size(0)
        
        # Calculate metrics
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total if task_type == 'classification' else avg_val_loss
        
        # Update history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['learning_rate'].append(optimizer.param_groups[0]['lr'])
        
        # Update best score
        if task_type == 'classification':
            current_score = val_acc
        else:
            current_score = -avg_val_loss  # Negative because we want to minimize loss
        
        if current_score > best_val_score:
            best_val_score = current_score
        
        # Learning rate scheduling
        if scheduler is not None:
            if scheduler_name == 'plateau':
                scheduler.step(current_score)
            else:
                scheduler.step()
        
        # Early stopping
        if early_stopping(current_score, model):
            if verbose:
                logging.info(f"Early stopping at epoch {epoch+1}")
            break
        
        if verbose and (epoch + 1) % 10 == 0:
            logging.info(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, "
                        f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")
    
    return best_val_score, history


def train_with_pruning(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    params: Dict[str, Any],
    device: torch.device,
    pruner: Optional[Any] = None,
    budget: int = 100
) -> float:
    """Train model with pruning support for multi-fidelity optimization.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        params: Hyperparameters
        device: Device to train on
        pruner: Optional pruner for early stopping
        budget: Training budget (e.g., epochs)
        
    Returns:
        Validation score
    """
    # Setup training
    optimizer_name = params.get('optimizer', 'adam').lower()
    learning_rate = params.get('learning_rate', 0.001)
    weight_decay = params.get('weight_decay', 0.0)
    
    if optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == 'sgd':
        momentum = params.get('momentum', 0.9)
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    else:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    task_type = params.get('task_type', 'classification')
    if task_type == 'classification':
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.MSELoss()
    
    model = model.to(device)
    
    # Training loop with pruning
    for epoch in range(budget):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            
            if task_type == 'classification':
                loss = criterion(output, target)
                pred = output.argmax(dim=1, keepdim=True)
                train_correct += pred.eq(target.view_as(pred)).sum().item()
            else:
                loss = criterion(output, target)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_total += target.size(0)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                
                if task_type == 'classification':
                    loss = criterion(output, target)
                    pred = output.argmax(dim=1, keepdim=True)
                    val_correct += pred.eq(target.view_as(pred)).sum().item()
                else:
                    loss = criterion(output, target)
                
                val_loss += loss.item()
                val_total += target.size(0)
        
        # Calculate score
        if task_type == 'classification':
            val_acc = 100. * val_correct / val_total
            score = val_acc
        else:
            avg_val_loss = val_loss / len(val_loader)
            score = -avg_val_loss  # Negative because we want to minimize loss
        
        # Check pruning
        if pruner is not None:
            if pruner.prune(epoch, score):
                return score
        
        # Report intermediate score for pruning
        if hasattr(pruner, 'report') and pruner is not None:
            pruner.report(score)
    
    return score
