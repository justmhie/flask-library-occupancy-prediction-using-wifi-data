"""
SHAP Analysis Module for Deep Learning Models
Provides comprehensive SHAP-based interpretability analysis for time series prediction models
"""

import numpy as np
import matplotlib.pyplot as plt
import shap
import warnings
warnings.filterwarnings('ignore')

def create_shap_explainer(model, X_train, model_type='deep'):
    """
    Create appropriate SHAP explainer for the model

    Args:
        model: Trained Keras model
        X_train: Training data for background samples
        model_type: Type of model ('deep', 'gradient', 'kernel')

    Returns:
        SHAP explainer object
    """
    print("      Creating SHAP explainer...")

    # Use a subset of training data as background for faster computation
    background_size = min(100, len(X_train))
    background_indices = np.random.choice(X_train.shape[0], background_size, replace=False)
    background_data = X_train[background_indices]

    # Reshape if needed for hybrid models
    if len(background_data.shape) == 4:
        # Hybrid CNN-LSTM: (samples, n_seq, n_steps, features)
        original_shape = background_data.shape
        background_data_flat = background_data.reshape(background_size, -1, 1)
    else:
        background_data_flat = background_data

    # Try multiple explainer types for compatibility
    explainer = None
    explainer_type = None

    # First try: GradientExplainer (more compatible with TensorFlow 2.x)
    try:
        print(f"      Trying GradientExplainer...")
        explainer = shap.GradientExplainer(model, background_data_flat)
        explainer_type = "GradientExplainer"
        print(f"      Using GradientExplainer with {background_size} background samples")
    except Exception as e1:
        print(f"      GradientExplainer failed: {str(e1)[:100]}")

        # Second try: KernelExplainer (model-agnostic, slower but reliable)
        try:
            print(f"      Trying KernelExplainer...")
            # Use even smaller background for KernelExplainer (it's slower)
            small_background_size = min(50, background_size)
            small_background = background_data_flat[:small_background_size]

            # Create prediction function
            def model_predict(x):
                return model.predict(x, verbose=0)

            explainer = shap.KernelExplainer(model_predict, small_background)
            explainer_type = "KernelExplainer"
            print(f"      Using KernelExplainer with {small_background_size} background samples")
        except Exception as e2:
            print(f"      KernelExplainer failed: {str(e2)[:100]}")
            raise Exception(f"All explainer types failed. Last error: {str(e2)}")

    return explainer, background_data_flat, explainer_type


def calculate_shap_values(explainer, X_test, explainer_type='gradient', max_samples=200):
    """
    Calculate SHAP values for test data

    Args:
        explainer: SHAP explainer object
        X_test: Test data
        explainer_type: Type of explainer being used
        max_samples: Maximum samples to analyze (for performance)

    Returns:
        SHAP values array
    """
    # Adjust sample size based on explainer type
    if explainer_type == "KernelExplainer":
        max_samples = min(50, max_samples)  # KernelExplainer is much slower

    print(f"      Computing SHAP values for {min(max_samples, len(X_test))} samples...")

    # Limit samples for performance
    test_samples = X_test[:max_samples]

    # Reshape if needed
    if len(test_samples.shape) == 4:
        original_shape = test_samples.shape
        test_samples = test_samples.reshape(test_samples.shape[0], -1, 1)

    # Calculate SHAP values
    if explainer_type == "KernelExplainer":
        # KernelExplainer needs special handling
        shap_values = explainer.shap_values(test_samples, nsamples=100)
    else:
        shap_values = explainer.shap_values(test_samples)

    # Handle output format (can be list or array)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    print(f"      SHAP values shape: {shap_values.shape}")
    return shap_values, test_samples


def plot_shap_summary(shap_values, X_test, model_name, location_name, save_path=None):
    """
    Create SHAP summary plot showing feature importance

    Args:
        shap_values: Computed SHAP values
        X_test: Test data
        model_name: Name of the model
        location_name: Library location name
        save_path: Path to save the plot
    """
    plt.figure(figsize=(12, 8))

    # Flatten data for visualization
    X_flat = X_test.reshape(X_test.shape[0], -1)
    shap_flat = shap_values.reshape(shap_values.shape[0], -1)

    # Create feature names (time steps)
    feature_names = [f't-{i}' for i in range(X_flat.shape[1]-1, -1, -1)]

    # Summary plot
    plt.subplot(2, 1, 1)
    shap.summary_plot(shap_flat, X_flat, feature_names=feature_names,
                      show=False, plot_type='bar')
    plt.title(f'SHAP Feature Importance - {model_name}\n{location_name}',
              fontsize=14, fontweight='bold')

    # Detailed summary plot
    plt.subplot(2, 1, 2)
    shap.summary_plot(shap_flat, X_flat, feature_names=feature_names,
                      show=False, plot_type='dot')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"      Saved SHAP summary: {save_path}")

    plt.close()


def plot_shap_waterfall(shap_values, X_test, explainer, model_name, location_name,
                        sample_idx=0, save_path=None):
    """
    Create SHAP waterfall plot for individual prediction

    Args:
        shap_values: Computed SHAP values
        X_test: Test data
        explainer: SHAP explainer with expected_value
        model_name: Name of the model
        location_name: Library location name
        sample_idx: Index of sample to explain
        save_path: Path to save the plot
    """
    plt.figure(figsize=(12, 8))

    # Get feature names
    n_features = shap_values.shape[1] if len(shap_values.shape) > 1 else 1
    feature_names = [f't-{i}' for i in range(n_features-1, -1, -1)]

    # Create Explanation object for waterfall plot
    base_value = explainer.expected_value if hasattr(explainer, 'expected_value') else 0
    if isinstance(base_value, np.ndarray):
        base_value = base_value[0]

    # Flatten for single sample
    shap_sample = shap_values[sample_idx].flatten()
    X_sample = X_test[sample_idx].flatten()

    # Create explanation object
    explanation = shap.Explanation(
        values=shap_sample,
        base_values=base_value,
        data=X_sample,
        feature_names=feature_names
    )

    # Waterfall plot
    shap.plots.waterfall(explanation, show=False)
    plt.title(f'SHAP Waterfall Plot - Sample {sample_idx}\n{model_name} - {location_name}',
              fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"      Saved SHAP waterfall: {save_path}")

    plt.close()


def plot_shap_force(shap_values, X_test, explainer, model_name, location_name,
                    sample_idx=0, save_path=None):
    """
    Create SHAP force plot for individual prediction

    Args:
        shap_values: Computed SHAP values
        X_test: Test data
        explainer: SHAP explainer with expected_value
        model_name: Name of the model
        location_name: Library location name
        sample_idx: Index of sample to explain
        save_path: Path to save the plot
    """
    # Get base value
    base_value = explainer.expected_value if hasattr(explainer, 'expected_value') else 0
    if isinstance(base_value, np.ndarray):
        base_value = base_value[0]

    # Flatten for single sample
    shap_sample = shap_values[sample_idx].flatten()
    X_sample = X_test[sample_idx].flatten()

    # Feature names
    feature_names = [f't-{i}h' for i in range(len(shap_sample)-1, -1, -1)]

    # Create force plot
    plt.figure(figsize=(20, 3))
    shap.force_plot(
        base_value,
        shap_sample,
        X_sample,
        feature_names=feature_names,
        matplotlib=True,
        show=False
    )
    plt.title(f'SHAP Force Plot - {model_name} - {location_name} (Sample {sample_idx})',
              fontsize=12, fontweight='bold', pad=20)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"      Saved SHAP force plot: {save_path}")

    plt.close()


def plot_shap_dependence(shap_values, X_test, model_name, location_name,
                         feature_idx=0, save_path=None):
    """
    Create SHAP dependence plot showing how a feature affects predictions

    Args:
        shap_values: Computed SHAP values
        X_test: Test data
        model_name: Name of the model
        location_name: Library location name
        feature_idx: Index of feature to analyze
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))

    # Flatten data
    X_flat = X_test.reshape(X_test.shape[0], -1)
    shap_flat = shap_values.reshape(shap_values.shape[0], -1)

    # Feature names
    feature_names = [f't-{i}h' for i in range(X_flat.shape[1]-1, -1, -1)]

    # Dependence plot
    shap.dependence_plot(
        feature_idx,
        shap_flat,
        X_flat,
        feature_names=feature_names,
        show=False
    )

    plt.title(f'SHAP Dependence Plot - {feature_names[feature_idx]}\n{model_name} - {location_name}',
              fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"      Saved SHAP dependence: {save_path}")

    plt.close()


def get_feature_importance_ranking(shap_values):
    """
    Get feature importance ranking from SHAP values

    Args:
        shap_values: Computed SHAP values

    Returns:
        Dictionary with feature importance metrics
    """
    # Flatten SHAP values
    shap_flat = shap_values.reshape(shap_values.shape[0], -1)

    # Calculate mean absolute SHAP values
    mean_abs_shap = np.abs(shap_flat).mean(axis=0)

    # Get top features
    top_indices = np.argsort(mean_abs_shap)[::-1]

    # Create ranking
    feature_importance = {
        'mean_abs_shap': mean_abs_shap.tolist(),
        'top_features_idx': top_indices[:10].tolist(),
        'top_features_importance': mean_abs_shap[top_indices[:10]].tolist()
    }

    return feature_importance


def create_comprehensive_shap_analysis(model, X_train, X_test, model_name,
                                       location_name, output_dir='model_results/shap'):
    """
    Create comprehensive SHAP analysis with all visualizations

    Args:
        model: Trained Keras model
        X_train: Training data
        X_test: Test data
        model_name: Name of the model
        location_name: Library location name
        output_dir: Directory to save plots

    Returns:
        Dictionary with SHAP analysis results
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n    Running SHAP Analysis for {model_name} - {location_name}...")

    try:
        # Create explainer
        explainer, background_data, explainer_type = create_shap_explainer(model, X_train)

        # Calculate SHAP values
        shap_values, test_samples = calculate_shap_values(
            explainer, X_test, explainer_type=explainer_type, max_samples=200
        )

        # Create safe filename
        safe_model = model_name.lower().replace(' ', '_').replace('-', '_')
        safe_location = location_name.lower().replace(' ', '_').replace('-', '_')

        # Generate all visualizations
        print("      Generating SHAP visualizations...")

        # 1. Summary plots
        summary_path = f"{output_dir}/{safe_model}_{safe_location}_summary.png"
        plot_shap_summary(shap_values, test_samples, model_name, location_name, summary_path)

        # 2. Waterfall plot for first prediction
        waterfall_path = f"{output_dir}/{safe_model}_{safe_location}_waterfall.png"
        plot_shap_waterfall(shap_values, test_samples, explainer, model_name,
                           location_name, sample_idx=0, save_path=waterfall_path)

        # 3. Force plot
        force_path = f"{output_dir}/{safe_model}_{safe_location}_force.png"
        plot_shap_force(shap_values, test_samples, explainer, model_name,
                       location_name, sample_idx=0, save_path=force_path)

        # 4. Dependence plot for most important feature
        dependence_path = f"{output_dir}/{safe_model}_{safe_location}_dependence.png"
        plot_shap_dependence(shap_values, test_samples, model_name, location_name,
                            feature_idx=0, save_path=dependence_path)

        # Get feature importance
        feature_importance = get_feature_importance_ranking(shap_values)

        print(f"      SHAP analysis complete!")

        return {
            'success': True,
            'feature_importance': feature_importance,
            'plots': {
                'summary': summary_path,
                'waterfall': waterfall_path,
                'force': force_path,
                'dependence': dependence_path
            }
        }

    except Exception as e:
        print(f"      Error in SHAP analysis: {str(e)}")
        return {
            'success': False,
            'error': str(e)
        }
