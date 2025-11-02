"""
Test SHAP Integration
Quick test to verify SHAP analysis works with existing trained models
"""

import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import pickle
from pathlib import Path
from shap_analysis import create_comprehensive_shap_analysis

print("=" * 80)
print("TEST SHAP INTEGRATION")
print("=" * 80)

# Check for existing models
models_dir = Path('saved_models')
if not models_dir.exists():
    print("\nNo saved models found. Please train models first using:")
    print("  python train_multiple_model_types.py")
    exit(1)

# Find first available model
model_files = list(models_dir.glob('*.keras'))
if not model_files:
    print("\nNo .keras model files found. Please train models first.")
    exit(1)

# Use first model for testing
model_path = model_files[0]
model_name_parts = model_path.stem.split('_')

print(f"\nTesting with model: {model_path.name}")
print(f"Loading model...")

# Load model
model = load_model(str(model_path))
print(f"  Model loaded successfully")

# Find corresponding scaler
scaler_name = model_path.stem + '_scaler.pkl'
scaler_path = Path('saved_scalers') / scaler_name

if not scaler_path.exists():
    print(f"\nWarning: Scaler not found at {scaler_path}")
    print("Creating synthetic test data instead...")

    # Create synthetic test data
    np.random.seed(42)
    n_samples = 300
    sequence_length = 24

    # Generate synthetic occupancy patterns
    t = np.linspace(0, 10 * np.pi, n_samples + sequence_length)
    base_pattern = 20 + 15 * np.sin(t / 12) + 5 * np.sin(t / 3)
    noise = np.random.normal(0, 2, len(t))
    occupancy = np.maximum(0, base_pattern + noise)

    # Create sequences
    X, y = [], []
    for i in range(len(occupancy) - sequence_length):
        X.append(occupancy[i:i+sequence_length])
        y.append(occupancy[i+sequence_length])

    X = np.array(X)
    y = np.array(y)

    # Normalize
    X_max = X.max()
    X = X / X_max
    y = y / X_max

    # Reshape for model input
    X = X.reshape(X.shape[0], sequence_length, 1)

    # Split
    train_size = int(0.8 * len(X))
    X_train = X[:train_size]
    X_test = X[train_size:]

else:
    print(f"Loading scaler from {scaler_path}")

    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    # Load actual data
    print("Loading data from all_data_cleaned.csv...")
    df = pd.read_csv('all_data_cleaned.csv')

    # Basic processing
    df['Start_dt'] = pd.to_datetime(df['Start_dt'])
    df.set_index('Start_dt', inplace=True)

    # Calculate occupancy
    occupancy = df['Client MAC'].resample('h').nunique().fillna(0)
    data = occupancy.values.reshape(-1, 1)
    scaled_data = scaler.transform(data)

    # Create sequences
    sequence_length = 24
    X, y = [], []
    for i in range(len(scaled_data) - sequence_length):
        X.append(scaled_data[i:i+sequence_length])
        y.append(scaled_data[i+sequence_length])

    X = np.array(X)
    y = np.array(y)

    # Split
    train_size = int(0.8 * len(X))
    X_train = X[:train_size]
    X_test = X[train_size:]

print(f"\nData prepared:")
print(f"  Training samples: {len(X_train)}")
print(f"  Test samples: {len(X_test)}")
print(f"  Input shape: {X_train.shape}")

# Extract model name and location from filename
if len(model_name_parts) >= 3:
    model_type = ' '.join(model_name_parts[:-2]).replace('_', ' ').title()
    location = ' '.join(model_name_parts[-2:]).replace('_', ' ').title()
else:
    model_type = "Test Model"
    location = "Test Location"

print(f"\nRunning SHAP analysis for: {model_type} - {location}")
print("-" * 80)

# Run SHAP analysis
shap_results = create_comprehensive_shap_analysis(
    model=model,
    X_train=X_train[:200],  # Limit for faster testing
    X_test=X_test[:100],     # Limit for faster testing
    model_name=model_type,
    location_name=location,
    output_dir='model_results/shap/test'
)

print("\n" + "=" * 80)
if shap_results['success']:
    print("SHAP ANALYSIS TEST: SUCCESS!")
    print("=" * 80)
    print("\nGenerated plots:")
    for plot_type, plot_path in shap_results['plots'].items():
        print(f"  - {plot_type}: {plot_path}")

    print("\nFeature importance top 5:")
    top_importance = shap_results['feature_importance']['top_features_importance'][:5]
    top_indices = shap_results['feature_importance']['top_features_idx'][:5]

    for i, (idx, imp) in enumerate(zip(top_indices, top_importance), 1):
        print(f"  {i}. t-{idx}: {imp:.4f}")

    print("\n" + "=" * 80)
    print("SHAP integration is working correctly!")
    print("You can now run full training with SHAP analysis:")
    print("  python train_multiple_model_types.py")
    print("=" * 80)

else:
    print("SHAP ANALYSIS TEST: FAILED")
    print("=" * 80)
    print(f"\nError: {shap_results.get('error', 'Unknown error')}")
    print("\nPlease check:")
    print("  1. SHAP library is installed: pip install shap")
    print("  2. TensorFlow compatibility")
    print("  3. Model architecture compatibility")
