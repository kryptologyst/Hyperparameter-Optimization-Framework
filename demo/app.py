"""Streamlit demo application for hyperparameter optimization visualization."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import yaml
from pathlib import Path
import time

from src.hpo import HPOFramework
from src.hpo.algorithms import create_hpo_algorithm
from src.hpo.data import get_data_loaders, get_dataset_info
from src.hpo.models import create_model, get_model_info
from src.hpo.train import train_model
from src.hpo.eval import evaluate_model, evaluate_hpo_results, create_leaderboard


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Hyperparameter Optimization Demo",
        page_icon="🔧",
        layout="wide"
    )
    
    st.title("🔧 Hyperparameter Optimization Framework")
    st.markdown("**Author:** [kryptologyst](https://github.com/kryptologyst)")
    
    # Sidebar
    st.sidebar.header("Configuration")
    
    # Dataset selection
    dataset = st.sidebar.selectbox(
        "Dataset",
        ["cifar10", "synthetic"],
        help="Choose the dataset for HPO experiments"
    )
    
    # Model selection
    model_options = {
        "cifar10": ["simple_cnn", "resnet"],
        "synthetic": ["mlp"]
    }
    model = st.sidebar.selectbox(
        "Model",
        model_options[dataset],
        help="Choose the model architecture"
    )
    
    # Algorithm selection
    algorithms = st.sidebar.multiselect(
        "HPO Algorithms",
        ["random_search", "grid_search", "optuna_tpe", "optuna_random", "hyperband"],
        default=["random_search", "optuna_tpe"],
        help="Select HPO algorithms to compare"
    )
    
    # Number of trials
    n_trials = st.sidebar.slider(
        "Number of Trials",
        min_value=5,
        max_value=50,
        value=20,
        help="Number of HPO trials to run"
    )
    
    # Max epochs
    max_epochs = st.sidebar.slider(
        "Max Epochs",
        min_value=10,
        max_value=100,
        value=30,
        help="Maximum training epochs"
    )
    
    # Run button
    run_experiment = st.sidebar.button("🚀 Run Experiment", type="primary")
    
    # Main content
    if run_experiment and algorithms:
        run_hpo_experiment(dataset, model, algorithms, n_trials, max_epochs)
    else:
        show_demo_info()


def run_hpo_experiment(dataset: str, model: str, algorithms: list, n_trials: int, max_epochs: int):
    """Run HPO experiment and display results."""
    
    # Create progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Initialize results storage
    all_results = {}
    
    # Load data
    status_text.text("Loading data...")
    progress_bar.progress(10)
    
    train_loader, val_loader, test_loader = get_data_loaders(
        dataset_name=dataset,
        batch_size=32,
        num_workers=0,  # Streamlit doesn't support multiprocessing
        val_split=0.2
    )
    
    dataset_info = get_dataset_info(dataset)
    
    # Run HPO algorithms
    total_algorithms = len(algorithms)
    
    for i, algorithm in enumerate(algorithms):
        status_text.text(f"Running {algorithm}...")
        progress_bar.progress(20 + (i * 60 // total_algorithms))
        
        # Create HPO algorithm
        hpo_algorithm = create_hpo_algorithm(
            algorithm,
            n_trials=n_trials,
            direction="maximize"
        )
        
        # Define search space
        search_space = get_search_space(dataset, model)
        
        # Define objective function
        def objective(params):
            return evaluate_hpo_trial(params, model, train_loader, val_loader, dataset_info, max_epochs)
        
        # Run optimization
        start_time = time.time()
        best_params = hpo_algorithm.optimize(objective, search_space)
        optimization_time = time.time() - start_time
        
        # Store results
        all_results[algorithm] = {
            'best_params': best_params,
            'trial_results': hpo_algorithm.get_trial_results(),
            'optimization_time': optimization_time,
            'best_score': hpo_algorithm.best_score
        }
    
    # Evaluate best models on test set
    status_text.text("Evaluating on test set...")
    progress_bar.progress(85)
    
    test_results = {}
    for algorithm, results in all_results.items():
        # Create and train best model
        best_model = create_and_train_model(
            results['best_params'], model, train_loader, val_loader, dataset_info, max_epochs
        )
        
        # Evaluate on test set
        test_metrics = evaluate_model(best_model, test_loader, "cpu", dataset_info['task_type'])
        test_results[algorithm] = test_metrics
    
    progress_bar.progress(100)
    status_text.text("Complete!")
    
    # Display results
    display_results(all_results, test_results, dataset, model)


def evaluate_hpo_trial(params, model_name, train_loader, val_loader, dataset_info, max_epochs):
    """Evaluate a single HPO trial."""
    try:
        # Create model
        model_params = extract_model_params(params, dataset_info)
        model = create_model(model_name, **model_params)
        
        # Add training parameters
        train_params = params.copy()
        train_params['task_type'] = dataset_info['task_type']
        
        # Train model
        val_score, _ = train_model(
            model, train_loader, val_loader, train_params, "cpu",
            max_epochs=max_epochs,
            early_stopping_patience=5,
            verbose=False
        )
        
        return val_score
        
    except Exception as e:
        st.warning(f"Trial failed: {e}")
        return float('-inf')


def create_and_train_model(params, model_name, train_loader, val_loader, dataset_info, max_epochs):
    """Create and train a model with given parameters."""
    # Create model
    model_params = extract_model_params(params, dataset_info)
    model = create_model(model_name, **model_params)
    
    # Add training parameters
    train_params = params.copy()
    train_params['task_type'] = dataset_info['task_type']
    
    # Train model
    train_model(
        model, train_loader, val_loader, train_params, "cpu",
        max_epochs=max_epochs,
        early_stopping_patience=5,
        verbose=False
    )
    
    return model


def extract_model_params(params, dataset_info):
    """Extract model-specific parameters from HPO params."""
    model_params = {}
    
    if 'n_filters' in params:
        model_params['n_filters'] = params['n_filters']
    if 'dropout' in params:
        model_params['dropout'] = params['dropout']
    if 'hidden_sizes' in params:
        model_params['hidden_sizes'] = params['hidden_sizes']
    if 'activation' in params:
        model_params['activation'] = params['activation']
    if 'n_blocks' in params:
        model_params['n_blocks'] = params['n_blocks']
    if 'base_channels' in params:
        model_params['base_channels'] = params['base_channels']
    
    # Add dataset-specific parameters
    model_params['n_classes'] = dataset_info['n_classes']
    if len(dataset_info['input_shape']) == 1:
        model_params['input_size'] = dataset_info['input_shape'][0]
    
    return model_params


def get_search_space(dataset: str, model: str) -> dict:
    """Get search space for given dataset and model."""
    if dataset == "cifar10" and model == "simple_cnn":
        return {
            'learning_rate': {'type': 'loguniform', 'low': 0.0001, 'high': 0.1},
            'batch_size': {'type': 'categorical', 'choices': [16, 32, 64]},
            'optimizer': {'type': 'categorical', 'choices': ['adam', 'sgd']},
            'n_filters': {'type': 'categorical', 'choices': [16, 32, 64]},
            'dropout': {'type': 'uniform', 'low': 0.0, 'high': 0.5}
        }
    elif dataset == "cifar10" and model == "resnet":
        return {
            'learning_rate': {'type': 'loguniform', 'low': 0.0001, 'high': 0.1},
            'batch_size': {'type': 'categorical', 'choices': [32, 64]},
            'optimizer': {'type': 'categorical', 'choices': ['adam', 'sgd']},
            'n_blocks': {'type': 'categorical', 'choices': [1, 2, 3]},
            'base_channels': {'type': 'categorical', 'choices': [32, 64]}
        }
    elif dataset == "synthetic" and model == "mlp":
        return {
            'learning_rate': {'type': 'loguniform', 'low': 0.0001, 'high': 0.1},
            'batch_size': {'type': 'categorical', 'choices': [16, 32, 64]},
            'optimizer': {'type': 'categorical', 'choices': ['adam', 'sgd']},
            'hidden_sizes': {'type': 'categorical', 'choices': [[64], [128], [64, 32], [128, 64]]},
            'dropout': {'type': 'uniform', 'low': 0.0, 'high': 0.5},
            'activation': {'type': 'categorical', 'choices': ['relu', 'tanh']}
        }
    else:
        return {}


def display_results(all_results: dict, test_results: dict, dataset: str, model: str):
    """Display HPO results."""
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Leaderboard", "📈 Convergence", "🔍 Best Parameters", "📋 Test Results"])
    
    with tab1:
        display_leaderboard(all_results)
    
    with tab2:
        display_convergence_plots(all_results)
    
    with tab3:
        display_best_parameters(all_results)
    
    with tab4:
        display_test_results(test_results)


def display_leaderboard(all_results: dict):
    """Display algorithm leaderboard."""
    st.subheader("Algorithm Performance Leaderboard")
    
    # Create leaderboard data
    leaderboard_data = []
    for algorithm, results in all_results.items():
        trial_results = results['trial_results']
        if trial_results:
            scores = [trial['score'] for trial in trial_results]
            leaderboard_data.append({
                'Algorithm': algorithm.replace('_', ' ').title(),
                'Best Score': max(scores),
                'Mean Score': np.mean(scores),
                'Std Score': np.std(scores),
                'Trials': len(trial_results),
                'Time (s)': results['optimization_time']
            })
    
    if leaderboard_data:
        df = pd.DataFrame(leaderboard_data)
        df = df.sort_values('Best Score', ascending=False)
        
        # Display table
        st.dataframe(df, use_container_width=True)
        
        # Create bar chart
        fig = px.bar(
            df, 
            x='Algorithm', 
            y='Best Score',
            title="Best Scores by Algorithm",
            color='Best Score',
            color_continuous_scale='viridis'
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No results to display")


def display_convergence_plots(all_results: dict):
    """Display convergence plots."""
    st.subheader("Convergence Analysis")
    
    # Create convergence plot
    fig = go.Figure()
    
    for algorithm, results in all_results.items():
        trial_results = results['trial_results']
        if trial_results:
            trials = [trial['trial'] for trial in trial_results]
            scores = [trial['score'] for trial in trial_results]
            
            fig.add_trace(go.Scatter(
                x=trials,
                y=scores,
                mode='lines+markers',
                name=algorithm.replace('_', ' ').title(),
                line=dict(width=2)
            ))
    
    fig.update_layout(
        title="Score Convergence Over Trials",
        xaxis_title="Trial Number",
        yaxis_title="Validation Score",
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Create box plot
    fig_box = go.Figure()
    
    for algorithm, results in all_results.items():
        trial_results = results['trial_results']
        if trial_results:
            scores = [trial['score'] for trial in trial_results]
            fig_box.add_trace(go.Box(
                y=scores,
                name=algorithm.replace('_', ' ').title(),
                boxpoints='outliers'
            ))
    
    fig_box.update_layout(
        title="Score Distribution by Algorithm",
        xaxis_title="Algorithm",
        yaxis_title="Validation Score"
    )
    
    st.plotly_chart(fig_box, use_container_width=True)


def display_best_parameters(all_results: dict):
    """Display best parameters for each algorithm."""
    st.subheader("Best Parameters by Algorithm")
    
    for algorithm, results in all_results.items():
        with st.expander(f"{algorithm.replace('_', ' ').title()} - Score: {results['best_score']:.4f}"):
            best_params = results['best_params']
            
            # Display parameters in columns
            col1, col2 = st.columns(2)
            
            with col1:
                for i, (param, value) in enumerate(best_params.items()):
                    if i % 2 == 0:
                        st.metric(param.replace('_', ' ').title(), f"{value:.4f}" if isinstance(value, float) else str(value))
            
            with col2:
                for i, (param, value) in enumerate(best_params.items()):
                    if i % 2 == 1:
                        st.metric(param.replace('_', ' ').title(), f"{value:.4f}" if isinstance(value, float) else str(value))


def display_test_results(test_results: dict):
    """Display test set results."""
    st.subheader("Test Set Performance")
    
    if test_results:
        # Create results dataframe
        test_data = []
        for algorithm, metrics in test_results.items():
            row = {'Algorithm': algorithm.replace('_', ' ').title()}
            row.update(metrics)
            test_data.append(row)
        
        df = pd.DataFrame(test_data)
        
        # Display table
        st.dataframe(df, use_container_width=True)
        
        # Create radar chart for classification metrics
        if 'accuracy' in df.columns:
            metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1']
            available_metrics = [m for m in metrics_to_plot if m in df.columns]
            
            if available_metrics:
                fig = go.Figure()
                
                for _, row in df.iterrows():
                    values = [row[metric] for metric in available_metrics]
                    fig.add_trace(go.Scatterpolar(
                        r=values,
                        theta=available_metrics,
                        fill='toself',
                        name=row['Algorithm']
                    ))
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 1]
                        )),
                    showlegend=True,
                    title="Test Set Metrics Comparison"
                )
                
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No test results to display")


def show_demo_info():
    """Show demo information and examples."""
    st.header("Welcome to the Hyperparameter Optimization Demo!")
    
    st.markdown("""
    This interactive demo allows you to:
    
    - **Compare HPO algorithms**: Random Search, Grid Search, Optuna TPE, and Hyperband
    - **Visualize convergence**: See how different algorithms improve over trials
    - **Analyze results**: View leaderboards, best parameters, and test performance
    - **Run experiments**: Test on CIFAR-10 or synthetic datasets
    
    ### 🚀 Getting Started
    
    1. **Select Configuration**: Choose your dataset and model from the sidebar
    2. **Pick Algorithms**: Select which HPO methods to compare
    3. **Set Parameters**: Adjust number of trials and training epochs
    4. **Run Experiment**: Click the "Run Experiment" button
    5. **Analyze Results**: Explore the results in the different tabs
    
    ### 📊 Available Algorithms
    
    - **Random Search**: Baseline random sampling
    - **Grid Search**: Exhaustive search over discrete parameters
    - **Optuna TPE**: Tree-structured Parzen Estimator (Bayesian optimization)
    - **Optuna Random**: Random sampling with Optuna framework
    - **Hyperband**: Multi-fidelity optimization with successive halving
    
    ### ⚠️ Important Notes
    
    - This is a **research/educational demo** - not for production use
    - Experiments run on CPU for compatibility
    - Results may vary due to random initialization
    - For faster experiments, use fewer trials or epochs
    """)
    
    # Show example results
    st.subheader("Example Results")
    
    example_data = {
        'Algorithm': ['Optuna TPE', 'Random Search', 'Grid Search', 'Hyperband'],
        'Best Score': [0.8234, 0.7891, 0.8012, 0.8156],
        'Mean Score': [0.8102, 0.7654, 0.7823, 0.7987],
        'Trials': [20, 20, 15, 20],
        'Time (s)': [45.2, 38.7, 52.1, 41.3]
    }
    
    df_example = pd.DataFrame(example_data)
    st.dataframe(df_example, use_container_width=True)
    
    # Create example plot
    fig_example = px.bar(
        df_example,
        x='Algorithm',
        y='Best Score',
        title="Example: Algorithm Performance Comparison",
        color='Best Score',
        color_continuous_scale='viridis'
    )
    st.plotly_chart(fig_example, use_container_width=True)


if __name__ == "__main__":
    main()
