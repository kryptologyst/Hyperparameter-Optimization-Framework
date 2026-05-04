#!/usr/bin/env python3
"""Simple example script demonstrating hyperparameter optimization."""

import logging
from pathlib import Path
from omegaconf import OmegaConf

# Import safety module first
from src.hpo.safety import safety_disclaimer, check_safety_constraints

# Import main framework
from src.hpo import HPOFramework

def main():
    """Run a simple HPO example."""
    
    # Print safety disclaimer
    safety_disclaimer()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    print("\nHyperparameter Optimization Example")
    print("=" * 50)
    
    try:
        # Load configuration
        config_path = Path("configs/default.yaml")
        if not config_path.exists():
            print("Configuration file not found. Please ensure configs/default.yaml exists.")
            return
        
        config = OmegaConf.load(config_path)
        print("Configuration loaded successfully")
        
        # Check safety constraints
        warnings = check_safety_constraints(config)
        if warnings:
            print("\nSafety warnings:")
            for warning in warnings:
                print(f"  - {warning}")
        
        # Initialize framework
        print("\nInitializing HPO Framework...")
        framework = HPOFramework(config)
        print("Framework initialized")
        
        # Run a simple experiment
        print("\nRunning synthetic MLP experiment...")
        print("This may take a few minutes...")
        
        results = framework.run_experiment("synthetic_mlp")
        
        # Display results
        print("\nResults Summary:")
        print(f"Dataset: {results['dataset_info']['description']}")
        print(f"Model: {results['config']['model_name']}")
        
        print(f"\nAlgorithm Leaderboard:")
        for i, entry in enumerate(results['leaderboard'], 1):
            print(f"  {i}. {entry['algorithm']}: {entry['best_score']:.4f} "
                  f"(±{entry['std_score']:.4f}) - {entry['n_trials']} trials")
        
        print(f"\nTest Set Performance:")
        for algorithm, metrics in results['test_results'].items():
            print(f"  {algorithm}:")
            for metric, value in metrics.items():
                print(f"    {metric}: {value:.4f}")
        
        print(f"\nExample completed successfully!")
        print(f"Results saved to: {config.get('results_dir', './results')}")
        
    except Exception as e:
        print(f"\nError: {e}")
        print("This might be due to missing dependencies or configuration issues.")
        print("Please check the installation and configuration.")
    
    print(f"\nNext Steps:")
    print("1. Try the interactive demo: streamlit run demo/app.py")
    print("2. Modify configs/default.yaml for your own experiments")
    print("3. Explore different algorithms and search spaces")
    print("4. Read the documentation in README.md")
    
    print(f"\nRemember: This is for research/educational purposes only!")
    print("   Not intended for production use without proper validation.")

if __name__ == "__main__":
    main()
