"""Evaluation utilities for hyperparameter optimization."""

import logging
from typing import Dict, Any, List, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, mean_squared_error,
    mean_absolute_error, r2_score
)


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    task_type: str = "classification"
) -> Dict[str, float]:
    """Evaluate model on test set.
    
    Args:
        model: Model to evaluate
        test_loader: Test data loader
        device: Device to evaluate on
        task_type: Type of task ('classification' or 'regression')
        
    Returns:
        Dict containing evaluation metrics
    """
    model.eval()
    model = model.to(device)
    
    all_predictions = []
    all_targets = []
    all_probabilities = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            if task_type == "classification":
                # Get predictions and probabilities
                probabilities = torch.softmax(output, dim=1)
                predictions = output.argmax(dim=1)
                
                all_probabilities.append(probabilities.cpu().numpy())
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(target.cpu().numpy())
            else:  # regression
                predictions = output.squeeze()
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(target.cpu().numpy())
    
    # Concatenate all results
    all_predictions = np.concatenate(all_predictions)
    all_targets = np.concatenate(all_targets)
    
    if task_type == "classification":
        all_probabilities = np.concatenate(all_probabilities)
        metrics = evaluate_classification(all_targets, all_predictions, all_probabilities)
    else:
        metrics = evaluate_regression(all_targets, all_predictions)
    
    return metrics


def evaluate_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray
) -> Dict[str, float]:
    """Evaluate classification model.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities
        
    Returns:
        Dict containing classification metrics
    """
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # AUC metrics (for binary classification)
    if len(np.unique(y_true)) == 2:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob[:, 1])
            metrics['pr_auc'] = average_precision_score(y_true, y_prob[:, 1])
        except ValueError:
            metrics['roc_auc'] = 0.0
            metrics['pr_auc'] = 0.0
    else:
        # Multi-class AUC
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob, multi_class='ovr', average='weighted')
        except ValueError:
            metrics['roc_auc'] = 0.0
        metrics['pr_auc'] = 0.0  # Not well-defined for multi-class
    
    return metrics


def evaluate_regression(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """Evaluate regression model.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Dict containing regression metrics
    """
    metrics = {}
    
    # Basic metrics
    metrics['mse'] = mean_squared_error(y_true, y_pred)
    metrics['rmse'] = np.sqrt(metrics['mse'])
    metrics['mae'] = mean_absolute_error(y_true, y_pred)
    metrics['r2'] = r2_score(y_true, y_pred)
    
    # Additional metrics
    metrics['mape'] = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    
    return metrics


def evaluate_hpo_results(
    trial_results: List[Dict[str, Any]],
    metric_name: str = "score"
) -> Dict[str, Any]:
    """Evaluate HPO results and create leaderboard.
    
    Args:
        trial_results: List of trial results
        metric_name: Name of metric to evaluate
        
    Returns:
        Dict containing evaluation summary
    """
    if not trial_results:
        return {}
    
    scores = [result[metric_name] for result in trial_results]
    
    evaluation = {
        'n_trials': len(trial_results),
        'best_score': max(scores),
        'worst_score': min(scores),
        'mean_score': np.mean(scores),
        'std_score': np.std(scores),
        'median_score': np.median(scores),
        'q25_score': np.percentile(scores, 25),
        'q75_score': np.percentile(scores, 75),
        'total_time': sum(result.get('eval_time', 0) for result in trial_results),
        'avg_time_per_trial': np.mean([result.get('eval_time', 0) for result in trial_results])
    }
    
    # Find best trial
    best_trial_idx = np.argmax(scores)
    evaluation['best_trial'] = trial_results[best_trial_idx]
    
    # Convergence analysis
    evaluation['convergence'] = analyze_convergence(scores)
    
    return evaluation


def analyze_convergence(scores: List[float], window_size: int = 10) -> Dict[str, Any]:
    """Analyze convergence of HPO algorithm.
    
    Args:
        scores: List of scores from trials
        window_size: Window size for moving average
        
    Returns:
        Dict containing convergence analysis
    """
    if len(scores) < window_size:
        return {'converged': False, 'convergence_point': None}
    
    # Calculate moving average
    moving_avg = []
    for i in range(window_size, len(scores) + 1):
        moving_avg.append(np.mean(scores[i-window_size:i]))
    
    # Check for convergence (no significant improvement in last 20% of trials)
    convergence_threshold = 0.01  # 1% improvement threshold
    convergence_window = max(5, len(scores) // 5)
    
    if len(moving_avg) < convergence_window:
        return {'converged': False, 'convergence_point': None}
    
    recent_scores = moving_avg[-convergence_window:]
    max_recent = max(recent_scores)
    min_recent = min(recent_scores)
    
    improvement = (max_recent - min_recent) / max_recent if max_recent > 0 else 0
    
    converged = improvement < convergence_threshold
    
    convergence_point = None
    if converged:
        # Find when convergence started
        for i in range(len(moving_avg) - convergence_window, -1, -1):
            window_scores = moving_avg[i:i+convergence_window]
            window_improvement = (max(window_scores) - min(window_scores)) / max(window_scores) if max(window_scores) > 0 else 0
            if window_improvement >= convergence_threshold:
                convergence_point = i + window_size
                break
    
    return {
        'converged': converged,
        'convergence_point': convergence_point,
        'improvement': improvement,
        'moving_avg': moving_avg
    }


def create_leaderboard(
    algorithm_results: Dict[str, List[Dict[str, Any]]],
    metric_name: str = "score"
) -> List[Dict[str, Any]]:
    """Create leaderboard comparing different HPO algorithms.
    
    Args:
        algorithm_results: Results from different algorithms
        metric_name: Metric to rank by
        
    Returns:
        List of algorithm rankings
    """
    leaderboard = []
    
    for algorithm_name, trial_results in algorithm_results.items():
        if not trial_results:
            continue
        
        evaluation = evaluate_hpo_results(trial_results, metric_name)
        
        leaderboard.append({
            'algorithm': algorithm_name,
            'best_score': evaluation['best_score'],
            'mean_score': evaluation['mean_score'],
            'std_score': evaluation['std_score'],
            'n_trials': evaluation['n_trials'],
            'total_time': evaluation['total_time'],
            'converged': evaluation['convergence']['converged'],
            'convergence_point': evaluation['convergence']['convergence_point']
        })
    
    # Sort by best score (descending)
    leaderboard.sort(key=lambda x: x['best_score'], reverse=True)
    
    return leaderboard


def compute_ablation_results(
    base_results: List[Dict[str, Any]],
    ablation_results: Dict[str, List[Dict[str, Any]]],
    metric_name: str = "score"
) -> Dict[str, Any]:
    """Compute ablation study results.
    
    Args:
        base_results: Results from base configuration
        ablation_results: Results from ablation experiments
        metric_name: Metric to compare
        
    Returns:
        Dict containing ablation analysis
    """
    base_eval = evaluate_hpo_results(base_results, metric_name)
    
    ablation_analysis = {
        'base': {
            'best_score': base_eval['best_score'],
            'mean_score': base_eval['mean_score'],
            'std_score': base_eval['std_score']
        },
        'ablation_effects': {}
    }
    
    for ablation_name, ablation_trials in ablation_results.items():
        ablation_eval = evaluate_hpo_results(ablation_trials, metric_name)
        
        # Calculate effect size
        best_score_diff = ablation_eval['best_score'] - base_eval['best_score']
        mean_score_diff = ablation_eval['mean_score'] - base_eval['mean_score']
        
        ablation_analysis['ablation_effects'][ablation_name] = {
            'best_score': ablation_eval['best_score'],
            'mean_score': ablation_eval['mean_score'],
            'std_score': ablation_eval['std_score'],
            'best_score_diff': best_score_diff,
            'mean_score_diff': mean_score_diff,
            'best_score_improvement': (best_score_diff / base_eval['best_score']) * 100 if base_eval['best_score'] != 0 else 0,
            'mean_score_improvement': (mean_score_diff / base_eval['mean_score']) * 100 if base_eval['mean_score'] != 0 else 0
        }
    
    return ablation_analysis
