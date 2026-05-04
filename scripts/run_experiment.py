#!/usr/bin/env python3
"""Main script for running hyperparameter optimization experiments."""

import argparse
import logging
from pathlib import Path
from omegaconf import OmegaConf

from src.hpo import HPOFramework


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Hyperparameter Optimization Framework")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                       help="Path to configuration file")
    parser.add_argument("--experiment", type=str, required=True,
                       help="Name of experiment to run")
    parser.add_argument("--ablation", action="store_true",
                       help="Run ablation study")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Load configuration
    config = OmegaConf.load(args.config)
    
    # Override logging level if verbose
    if args.verbose:
        config.logging_level = "DEBUG"
    
    # Initialize framework
    framework = HPOFramework(config)
    
    if args.ablation:
        # Run ablation study
        ablation_configs = {
            "no_dropout": {"search_space.dropout": {"type": "categorical", "choices": [0.0]}},
            "high_dropout": {"search_space.dropout": {"type": "categorical", "choices": [0.7]}},
            "small_model": {"search_space.n_filters": {"type": "categorical", "choices": [16, 32]}},
            "large_model": {"search_space.n_filters": {"type": "categorical", "choices": [128, 256]}}
        }
        
        results = framework.run_ablation_study(args.experiment, ablation_configs)
        
        # Print ablation results
        print("\n" + "="*50)
        print("ABLATION STUDY RESULTS")
        print("="*50)
        
        ablation_analysis = results['ablation_analysis']
        print(f"Base Configuration:")
        print(f"  Best Score: {ablation_analysis['base']['best_score']:.4f}")
        print(f"  Mean Score: {ablation_analysis['base']['mean_score']:.4f}")
        
        print(f"\nAblation Effects:")
        for ablation_name, effect in ablation_analysis['ablation_effects'].items():
            print(f"  {ablation_name}:")
            print(f"    Best Score: {effect['best_score']:.4f} "
                  f"({effect['best_score_improvement']:+.2f}%)")
            print(f"    Mean Score: {effect['mean_score']:.4f} "
                  f"({effect['mean_score_improvement']:+.2f}%)")
    else:
        # Run single experiment
        results = framework.run_experiment(args.experiment)
        
        # Print results
        print("\n" + "="*50)
        print("EXPERIMENT RESULTS")
        print("="*50)
        
        print(f"Experiment: {args.experiment}")
        print(f"Dataset: {results['dataset_info']['description']}")
        print(f"Model: {results['config']['model_name']}")
        
        print(f"\nLeaderboard:")
        for i, entry in enumerate(results['leaderboard'], 1):
            print(f"  {i}. {entry['algorithm']}: {entry['best_score']:.4f} "
                  f"(±{entry['std_score']:.4f}) - {entry['n_trials']} trials")
        
        print(f"\nTest Set Results:")
        for algorithm, metrics in results['test_results'].items():
            print(f"  {algorithm}:")
            for metric, value in metrics.items():
                print(f"    {metric}: {value:.4f}")


if __name__ == "__main__":
    main()
