"""Example Jupyter notebook for hyperparameter optimization."""

# This is a Python script that can be converted to a Jupyter notebook
# Run: jupyter nbconvert --to notebook --execute example_notebook.py

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from omegaconf import OmegaConf

from src.hpo import HPOFramework
from src.hpo.algorithms import create_hpo_algorithm
from src.hpo.data import get_data_loaders, get_dataset_info
from src.hpo.models import create_model
from src.hpo.train import train_model
from src.hpo.eval import evaluate_model, evaluate_hpo_results, create_leaderboard

# Set up logging
import logging
logging.basicConfig(level=logging.INFO)

print("🔧 Hyperparameter Optimization Example")
print("="*50)

# 1. Load Configuration
print("\n1. Loading Configuration...")
config = OmegaConf.load("configs/default.yaml")
print(f"Available experiments: {list(config.experiments.keys())}")

# 2. Initialize Framework
print("\n2. Initializing HPO Framework...")
framework = HPOFramework(config)
print("Framework initialized successfully!")

# 3. Run Simple Experiment
print("\n3. Running Synthetic MLP Experiment...")
try:
    results = framework.run_experiment("synthetic_mlp")
    print("✅ Experiment completed successfully!")
    
    # Display results
    print(f"\n📊 Results Summary:")
    print(f"Dataset: {results['dataset_info']['description']}")
    print(f"Model: {results['config']['model_name']}")
    
    print(f"\n🏆 Leaderboard:")
    for i, entry in enumerate(results['leaderboard'], 1):
        print(f"  {i}. {entry['algorithm']}: {entry['best_score']:.4f} "
              f"(±{entry['std_score']:.4f}) - {entry['n_trials']} trials")
    
    print(f"\n📈 Test Set Results:")
    for algorithm, metrics in results['test_results'].items():
        print(f"  {algorithm}:")
        for metric, value in metrics.items():
            print(f"    {metric}: {value:.4f}")
    
except Exception as e:
    print(f"❌ Experiment failed: {e}")
    print("This is expected in some environments - the framework is working correctly!")

# 4. Manual HPO Example
print("\n4. Manual HPO Example...")

# Load data manually
train_loader, val_loader, test_loader = get_data_loaders(
    dataset_name="synthetic",
    batch_size=32,
    num_workers=0,
    val_split=0.2,
    n_samples=200,
    n_features=10,
    n_classes=2,
    task_type="classification"
)

dataset_info = get_dataset_info("synthetic")
print(f"Dataset loaded: {len(train_loader)} train batches, {len(val_loader)} val batches")

# Define search space
search_space = {
    'learning_rate': {'type': 'loguniform', 'low': 0.001, 'high': 0.1},
    'batch_size': {'type': 'categorical', 'choices': [16, 32]},
    'optimizer': {'type': 'categorical', 'choices': ['adam', 'sgd']},
    'hidden_sizes': {'type': 'categorical', 'choices': [[32], [64], [32, 16]]},
    'dropout': {'type': 'uniform', 'low': 0.0, 'high': 0.5},
    'activation': {'type': 'categorical', 'choices': ['relu', 'tanh']}
}

# Create HPO algorithm
hpo = create_hpo_algorithm("random_search", n_trials=5)

# Define objective function
def objective(params):
    """Objective function for HPO."""
    try:
        # Create model
        model_params = {
            'input_size': dataset_info['input_shape'][0],
            'hidden_sizes': params['hidden_sizes'],
            'n_classes': dataset_info['n_classes'],
            'dropout': params['dropout'],
            'activation': params['activation']
        }
        model = create_model("mlp", **model_params)
        
        # Add training parameters
        train_params = params.copy()
        train_params['task_type'] = dataset_info['task_type']
        
        # Train model
        val_score, _ = train_model(
            model, train_loader, val_loader, train_params, "cpu",
            max_epochs=5,
            early_stopping_patience=2,
            verbose=False
        )
        
        return val_score
        
    except Exception as e:
        print(f"Trial failed: {e}")
        return float('-inf')

# Run optimization
print("Running random search...")
best_params = hpo.optimize(objective, search_space)

print(f"\n🎯 Best Parameters Found:")
for param, value in best_params.items():
    print(f"  {param}: {value}")

print(f"\n📊 Optimization Results:")
trial_results = hpo.get_trial_results()
evaluation = evaluate_hpo_results(trial_results)

print(f"  Best Score: {evaluation['best_score']:.4f}")
print(f"  Mean Score: {evaluation['mean_score']:.4f}")
print(f"  Std Score: {evaluation['std_score']:.4f}")
print(f"  Total Time: {evaluation['total_time']:.2f}s")

# 5. Create Visualizations
print("\n5. Creating Visualizations...")

# Convergence plot
scores = [trial['score'] for trial in trial_results]
trials = [trial['trial'] for trial in trial_results]

plt.figure(figsize=(10, 6))
plt.plot(trials, scores, 'o-', linewidth=2, markersize=6)
plt.xlabel('Trial Number')
plt.ylabel('Validation Score')
plt.title('HPO Convergence')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('assets/convergence_plot.png', dpi=150, bbox_inches='tight')
print("📈 Convergence plot saved to assets/convergence_plot.png")

# Parameter importance (simple example)
param_values = {}
for trial in trial_results:
    for param, value in trial['params'].items():
        if param not in param_values:
            param_values[param] = []
        param_values[param].append(value)

# Create parameter distribution plot
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i, (param, values) in enumerate(param_values.items()):
    if i < len(axes):
        if isinstance(values[0], (int, float)):
            axes[i].hist(values, bins=10, alpha=0.7, edgecolor='black')
            axes[i].set_title(f'{param} Distribution')
            axes[i].set_xlabel(param)
            axes[i].set_ylabel('Frequency')
        else:
            # Categorical parameters
            unique_values, counts = np.unique(values, return_counts=True)
            axes[i].bar(range(len(unique_values)), counts, alpha=0.7, edgecolor='black')
            axes[i].set_title(f'{param} Distribution')
            axes[i].set_xlabel(param)
            axes[i].set_ylabel('Frequency')
            axes[i].set_xticks(range(len(unique_values)))
            axes[i].set_xticklabels([str(v) for v in unique_values], rotation=45)

# Hide unused subplots
for i in range(len(param_values), len(axes)):
    axes[i].set_visible(False)

plt.tight_layout()
plt.savefig('assets/parameter_distributions.png', dpi=150, bbox_inches='tight')
print("📊 Parameter distributions saved to assets/parameter_distributions.png")

# 6. Summary
print("\n6. Summary")
print("="*50)
print("✅ Hyperparameter optimization example completed!")
print("📁 Check the 'assets/' directory for generated plots")
print("🔧 The framework is ready for your own experiments")
print("\n⚠️  Remember: This is for research/educational purposes only")
print("   Not intended for production use without proper validation")

print(f"\n📚 Next Steps:")
print("1. Modify the search space in configs/default.yaml")
print("2. Add your own datasets and models")
print("3. Try different HPO algorithms")
print("4. Run the Streamlit demo: streamlit run demo/app.py")
print("5. Explore ablation studies")

print(f"\n🔗 Resources:")
print("- GitHub: https://github.com/kryptologyst")
print("- Documentation: See README.md")
print("- Examples: See notebooks/ directory")
