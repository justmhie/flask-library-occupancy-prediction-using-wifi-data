"""
Train CNN-LSTM Models for All Libraries
Trains separate models for each library location + overall model
Generates comprehensive metrics and visualizations
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Flatten, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from math import sqrt
import matplotlib.pyplot as plt
import pickle
import os
import json
import sys
import io
from datetime import datetime
from ap_location_mapping import AP_LOCATION_MAP, LIBRARY_NAMES, get_location_from_ap

# Fix encoding for Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("=" * 80)
print("LIBRARY OCCUPANCY PREDICTION - TRAIN ALL MODELS")
print("=" * 80)

# ============================================
# CONFIGURATION
# ============================================

DATA_FILE = 'all_data_cleaned.csv'
SEQUENCE_LENGTH = 24
TEST_SIZE = 0.2
VALIDATION_SPLIT = 0.1
EPOCHS = 50  # Reduced for faster training of multiple models
BATCH_SIZE = 32
MIN_HOURS_REQUIRED = 100  # Minimum hours of data needed to train

# Create directories
os.makedirs('saved_models', exist_ok=True)
os.makedirs('saved_scalers', exist_ok=True)
os.makedirs('training_results', exist_ok=True)
os.makedirs('training_results/plots', exist_ok=True)

# ============================================
# HELPER FUNCTIONS
# ============================================

def calculate_metrics(y_true, y_pred):
    """Calculate comprehensive metrics"""
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1))) * 100  # +1 to avoid division by zero

    return {
        'r2': r2,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'mape': mape
    }

def build_cnn_lstm_model(sequence_length):
    """Build CNN-LSTM model architecture"""
    model = Sequential([
        Input(shape=(sequence_length, 1)),

        # CNN layers
        Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),

        Conv1D(filters=32, kernel_size=3, activation='relu', padding='same'),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),

        # LSTM layers
        LSTM(units=50, return_sequences=True, activation='relu'),
        Dropout(0.2),

        LSTM(units=50, activation='relu'),
        Dropout(0.2),

        # Dense layers
        Dense(25, activation='relu'),
        Dense(1)
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mean_squared_error',
        metrics=['mae']
    )

    return model

def plot_results(y_true, y_pred, history, location_name, metrics):
    """Create visualization plots"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Plot 1: Predictions vs Actual
    n_points = min(200, len(y_true))
    axes[0, 0].plot(y_true[:n_points], label='Actual', color='blue', linewidth=2, alpha=0.7)
    axes[0, 0].plot(y_pred[:n_points], label='Predicted', color='red', linewidth=2, linestyle='--', alpha=0.7)
    axes[0, 0].set_title(f'{location_name} - Actual vs Predicted', fontweight='bold')
    axes[0, 0].set_xlabel('Time Step')
    axes[0, 0].set_ylabel('Number of Users')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Training History
    axes[0, 1].plot(history.history['loss'], label='Training Loss', color='blue', alpha=0.7)
    axes[0, 1].plot(history.history['val_loss'], label='Validation Loss', color='red', alpha=0.7)
    axes[0, 1].set_title('Training History - Loss', fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss (MSE)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Error Distribution
    errors = y_true - y_pred
    axes[1, 0].hist(errors, bins=50, color='purple', alpha=0.7, edgecolor='black')
    axes[1, 0].axvline(x=0, color='red', linestyle='--', linewidth=2)
    axes[1, 0].set_title('Prediction Error Distribution', fontweight='bold')
    axes[1, 0].set_xlabel('Error (Actual - Predicted)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Metrics Summary
    axes[1, 1].axis('off')
    metrics_text = f"""
    MODEL PERFORMANCE METRICS
    {'-' * 40}

    R² Score:      {metrics['r2']:.4f}
    RMSE:          {metrics['rmse']:.2f} users
    MAE:           {metrics['mae']:.2f} users
    MAPE:          {metrics['mape']:.2f}%
    MSE:           {metrics['mse']:.2f}

    Higher R² is better (max 1.0)
    Lower RMSE/MAE/MAPE is better
    """
    axes[1, 1].text(0.1, 0.5, metrics_text, fontsize=12, family='monospace',
                    verticalalignment='center')

    plt.tight_layout()
    return fig

# ============================================
# LOAD AND PREPARE DATA
# ============================================

print("\n1. Loading data...")
df = pd.read_csv(DATA_FILE)
print(f"   ✓ Loaded {len(df):,} records")

# Add location column based on AP MAC
print("\n2. Mapping AP MACs to library locations...")
df['Location'] = df['AP MAC'].apply(get_location_from_ap)
location_counts = df['Location'].value_counts()
print(f"   ✓ Found {len(location_counts)} locations")
for loc, count in location_counts.items():
    if loc != 'unknown':
        print(f"      - {LIBRARY_NAMES.get(loc, loc)}: {count:,} records")

# Process timestamps
print("\n3. Processing timestamps...")
df['Start_dt'] = pd.to_datetime(df['Start_dt'])
df.set_index('Start_dt', inplace=True)
print(f"   ✓ Date range: {df.index.min()} to {df.index.max()}")

# ============================================
# TRAIN MODELS FOR EACH LIBRARY
# ============================================

all_results = {}
libraries_to_train = ['all'] + [loc for loc in AP_LOCATION_MAP.keys()]

for location_id in libraries_to_train:
    location_name = 'Overall (All Libraries)' if location_id == 'all' else LIBRARY_NAMES.get(location_id, location_id)

    print("\n" + "=" * 80)
    print(f"TRAINING MODEL: {location_name}")
    print("=" * 80)

    # Filter data for this library
    if location_id == 'all':
        df_lib = df.copy()
    else:
        df_lib = df[df['Location'] == location_id].copy()

    if len(df_lib) == 0:
        print(f"   ⚠ No data for {location_name}, skipping...")
        continue

    print(f"\n   Records: {len(df_lib):,}")

    # Calculate hourly occupancy
    occupancy = df_lib['Client MAC'].resample('h').nunique()
    occupancy = occupancy.fillna(0)

    print(f"   Hours of data: {len(occupancy)}")
    print(f"   Min users/hour: {int(occupancy.min())}")
    print(f"   Max users/hour: {int(occupancy.max())}")
    print(f"   Avg users/hour: {int(occupancy.mean())}")

    if len(occupancy) < MIN_HOURS_REQUIRED:
        print(f"   ⚠ Insufficient data (need {MIN_HOURS_REQUIRED}+ hours), skipping...")
        continue

    # Create sequences
    data = occupancy.values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    X, y = [], []
    for i in range(len(scaled_data) - SEQUENCE_LENGTH):
        X.append(scaled_data[i:i+SEQUENCE_LENGTH])
        y.append(scaled_data[i+SEQUENCE_LENGTH])

    X = np.array(X)
    y = np.array(y)

    if len(X) < 50:
        print(f"   ⚠ Not enough sequences ({len(X)}), skipping...")
        continue

    print(f"   Sequences created: {len(X):,}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, shuffle=False
    )
    print(f"   Training samples: {len(X_train):,}")
    print(f"   Testing samples: {len(X_test):,}")

    # Build and train model
    print(f"\n   Building CNN-LSTM model...")
    model = build_cnn_lstm_model(SEQUENCE_LENGTH)

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001, verbose=0)
    ]

    print(f"   Training for up to {EPOCHS} epochs...")
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        callbacks=callbacks,
        verbose=0
    )

    actual_epochs = len(history.history['loss'])
    print(f"   ✓ Training completed in {actual_epochs} epochs")

    # Evaluate model
    print(f"\n   Evaluating model...")
    predictions_scaled = model.predict(X_test, verbose=0)
    predictions = scaler.inverse_transform(predictions_scaled)
    y_test_actual = scaler.inverse_transform(y_test)

    y_true = y_test_actual.flatten()
    y_pred = predictions.flatten()

    metrics = calculate_metrics(y_true, y_pred)

    print(f"\n   {'=' * 40}")
    print(f"   MODEL PERFORMANCE")
    print(f"   {'=' * 40}")
    print(f"   R² Score:  {metrics['r2']:.4f}")
    print(f"   RMSE:      {metrics['rmse']:.2f} users")
    print(f"   MAE:       {metrics['mae']:.2f} users")
    print(f"   MAPE:      {metrics['mape']:.2f}%")

    # Save model and scaler
    model_path = f'saved_models/{location_id}_model.keras'
    scaler_path = f'saved_scalers/{location_id}_scaler.pkl'

    model.save(model_path)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)

    print(f"\n   ✓ Model saved: {model_path}")
    print(f"   ✓ Scaler saved: {scaler_path}")

    # Create visualization
    print(f"   Creating visualizations...")
    fig = plot_results(y_true, y_pred, history, location_name, metrics)
    plot_path = f'training_results/plots/{location_id}_results.png'
    fig.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"   ✓ Plot saved: {plot_path}")

    # Store results
    all_results[location_id] = {
        'location_name': location_name,
        'metrics': metrics,
        'data_stats': {
            'total_records': len(df_lib),
            'total_hours': len(occupancy),
            'min_occupancy': int(occupancy.min()),
            'max_occupancy': int(occupancy.max()),
            'avg_occupancy': float(occupancy.mean()),
            'training_samples': len(X_train),
            'testing_samples': len(X_test),
            'epochs_trained': actual_epochs
        },
        'model_path': model_path,
        'scaler_path': scaler_path,
        'trained_date': datetime.now().isoformat()
    }

# ============================================
# SAVE SUMMARY RESULTS
# ============================================

print("\n" + "=" * 80)
print("SAVING SUMMARY RESULTS")
print("=" * 80)

# Save detailed results as JSON
results_json_path = 'training_results/all_models_results.json'
with open(results_json_path, 'w') as f:
    json.dump(all_results, f, indent=2)
print(f"✓ Detailed results saved: {results_json_path}")

# Create comparison visualization
if len(all_results) > 1:
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    locations = list(all_results.keys())
    location_names = [all_results[loc]['location_name'] for loc in locations]

    # R² Scores
    r2_scores = [all_results[loc]['metrics']['r2'] for loc in locations]
    axes[0, 0].barh(location_names, r2_scores, color='steelblue')
    axes[0, 0].set_xlabel('R² Score')
    axes[0, 0].set_title('Model R² Scores (Higher is Better)', fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3, axis='x')

    # RMSE
    rmse_scores = [all_results[loc]['metrics']['rmse'] for loc in locations]
    axes[0, 1].barh(location_names, rmse_scores, color='coral')
    axes[0, 1].set_xlabel('RMSE (users)')
    axes[0, 1].set_title('Model RMSE (Lower is Better)', fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3, axis='x')

    # MAE
    mae_scores = [all_results[loc]['metrics']['mae'] for loc in locations]
    axes[1, 0].barh(location_names, mae_scores, color='mediumseagreen')
    axes[1, 0].set_xlabel('MAE (users)')
    axes[1, 0].set_title('Model MAE (Lower is Better)', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='x')

    # Average Occupancy
    avg_occ = [all_results[loc]['data_stats']['avg_occupancy'] for loc in locations]
    axes[1, 1].barh(location_names, avg_occ, color='mediumpurple')
    axes[1, 1].set_xlabel('Average Users')
    axes[1, 1].set_title('Average Occupancy by Library', fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    comparison_path = 'training_results/models_comparison.png'
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Comparison plot saved: {comparison_path}")

# Print summary table
print("\n" + "=" * 80)
print("TRAINING SUMMARY")
print("=" * 80)
print(f"\n{'Library':<30} {'R²':<8} {'RMSE':<10} {'MAE':<10} {'Avg Users':<12}")
print("-" * 80)
for loc_id, results in all_results.items():
    name = results['location_name'][:28]
    metrics = results['metrics']
    avg_users = results['data_stats']['avg_occupancy']
    print(f"{name:<30} {metrics['r2']:<8.4f} {metrics['rmse']:<10.2f} {metrics['mae']:<10.2f} {avg_users:<12.1f}")

print("\n" + "=" * 80)
print(f"✅ TRAINING COMPLETED - {len(all_results)} models trained successfully!")
print("=" * 80)
print("\nNext steps:")
print("  1. Restart the backend: python api_backend.py")
print("  2. Start the dashboard: cd library-dashboard && npm start")
print("  3. View results at: http://localhost:3000")
print("=" * 80)
