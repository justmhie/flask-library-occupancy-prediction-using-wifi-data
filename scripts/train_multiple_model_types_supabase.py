"""
Train Multiple Model Types for Library Occupancy Prediction - Supabase Version
Trains 4 different model architectures (LSTM, CNN, Hybrid, Advanced) for each library
Saves all models, scalers, and results to Supabase Storage
Run from project root: python scripts/train_multiple_model_types_supabase.py
"""
import os
import sys
if __name__ == "__main__":
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(_root)
    if _root not in sys.path:
        sys.path.insert(0, _root)

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Dense, LSTM, Conv1D, MaxPooling1D, Flatten,
                                     Dropout, Input, TimeDistributed, Attention,
                                     concatenate, Reshape)
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
from supabase_config import SupabaseStorage

# Fix encoding for Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("=" * 80)
print("TRAIN MULTIPLE MODEL TYPES - SUPABASE VERSION")
print("=" * 80)

# ============================================
# CONFIGURATION
# ============================================

DATA_FILE = 'all_data_cleaned.csv'
SEQUENCE_LENGTH = 24
TEST_SIZE = 0.2
VALIDATION_SPLIT = 0.1
EPOCHS = 30
BATCH_SIZE = 32

# Supabase Storage Configuration
STORAGE_BUCKET = 'models'  # You'll need to create this bucket in Supabase

# Model types to train
MODEL_TYPES = ['lstm_only', 'cnn_only', 'hybrid_cnn_lstm', 'advanced_cnn_lstm']

# Create temporary local directories (will upload to Supabase after)
os.makedirs('temp_models', exist_ok=True)
os.makedirs('temp_scalers', exist_ok=True)
os.makedirs('temp_results', exist_ok=True)

# ============================================
# HELPER FUNCTIONS
# ============================================

def calculate_metrics(y_true, y_pred):
    """Calculate comprehensive metrics"""
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1))) * 100

    return {
        'r2': r2,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'mape': mape
    }

def build_lstm_only_model(sequence_length):
    """LSTM-Only Model"""
    model = Sequential([
        Input(shape=(sequence_length, 1)),
        LSTM(units=50, return_sequences=True, activation='relu'),
        LSTM(units=50, activation='relu'),
        Dense(25, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mae'])
    return model

def build_cnn_only_model(sequence_length):
    """CNN-Only Model"""
    model = Sequential([
        Input(shape=(sequence_length, 1)),
        Conv1D(filters=64, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(50, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mae'])
    return model

def build_hybrid_cnn_lstm_model(sequence_length):
    """Hybrid CNN-LSTM Model"""
    n_seq = 4
    n_steps = sequence_length // n_seq

    model = Sequential([
        Input(shape=(n_seq, n_steps, 1)),
        TimeDistributed(Conv1D(filters=64, kernel_size=2, activation='relu')),
        TimeDistributed(MaxPooling1D(pool_size=2)),
        TimeDistributed(Flatten()),
        LSTM(units=50, activation='relu'),
        Dense(25, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mae'])
    return model

def build_advanced_cnn_lstm_model(sequence_length):
    """Advanced CNN-LSTM with Attention"""
    model = Sequential([
        Input(shape=(sequence_length, 1)),
        Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        Conv1D(filters=32, kernel_size=3, activation='relu', padding='same'),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        LSTM(units=50, return_sequences=True, activation='relu'),
        Dropout(0.2),
        LSTM(units=50, activation='relu'),
        Dropout(0.2),
        Dense(25, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mae'])
    return model

# ============================================
# LOAD AND PREPARE DATA
# ============================================

print("\n1. Loading data...")
df = pd.read_csv(DATA_FILE)
print(f"   ✓ Loaded {len(df):,} records")

# Add location column
print("\n2. Mapping AP MACs to libraries...")
df['Location'] = df['AP MAC'].apply(get_location_from_ap)
print(f"   ✓ Mapped locations")

# Process timestamps
df['Start_dt'] = pd.to_datetime(df['Start_dt'])
df.set_index('Start_dt', inplace=True)
print(f"   ✓ Date range: {df.index.min()} to {df.index.max()}")

# ============================================
# TRAIN MODELS
# ============================================

all_results = {}
uploaded_files = {
    'models': [],
    'scalers': [],
    'results': []
}

libraries_to_train = list(AP_LOCATION_MAP.keys())

for model_type in MODEL_TYPES:
    model_name = {
        'lstm_only': 'LSTM Only',
        'cnn_only': 'CNN Only',
        'hybrid_cnn_lstm': 'Hybrid CNN-LSTM',
        'advanced_cnn_lstm': 'Advanced CNN-LSTM'
    }[model_type]

    print("\n" + "=" * 80)
    print(f"TRAINING MODEL TYPE: {model_name}")
    print("=" * 80)

    model_results = {}

    for location_id in libraries_to_train:
        location_name = LIBRARY_NAMES.get(location_id, location_id)

        print(f"\n  Training {model_name} for {location_name}...")

        # Filter data
        df_lib = df[df['Location'] == location_id].copy()

        if len(df_lib) < 100:
            print(f"    ⚠ Insufficient data, skipping...")
            continue

        # Calculate occupancy
        occupancy = df_lib['Client MAC'].resample('h').nunique().fillna(0)

        if len(occupancy) < 100:
            print(f"    ⚠ Insufficient hours ({len(occupancy)}), skipping...")
            continue

        print(f"    Records: {len(df_lib):,}, Hours: {len(occupancy)}")

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

        # Reshape for hybrid model
        if model_type == 'hybrid_cnn_lstm':
            n_seq = 4
            n_steps = SEQUENCE_LENGTH // n_seq
            X = X.reshape((X.shape[0], n_seq, n_steps, 1))

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, shuffle=False)

        # Build model
        if model_type == 'lstm_only':
            model = build_lstm_only_model(SEQUENCE_LENGTH)
        elif model_type == 'cnn_only':
            model = build_cnn_only_model(SEQUENCE_LENGTH)
        elif model_type == 'hybrid_cnn_lstm':
            model = build_hybrid_cnn_lstm_model(SEQUENCE_LENGTH)
        else:
            model = build_advanced_cnn_lstm_model(SEQUENCE_LENGTH)

        # Train
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=0),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001, verbose=0)
        ]

        history = model.fit(
            X_train, y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_split=VALIDATION_SPLIT,
            callbacks=callbacks,
            verbose=0
        )

        # Evaluate
        predictions_scaled = model.predict(X_test, verbose=0)
        predictions = scaler.inverse_transform(predictions_scaled)
        y_test_actual = scaler.inverse_transform(y_test)

        y_true = y_test_actual.flatten()
        y_pred = predictions.flatten()

        metrics = calculate_metrics(y_true, y_pred)

        print(f"    ✓ R²: {metrics['r2']:.4f}, RMSE: {metrics['rmse']:.2f}")

        # Save model and scaler locally first
        model_filename = f'{model_type}_{location_id}_model.keras'
        scaler_filename = f'{model_type}_{location_id}_scaler.pkl'

        local_model_path = f'temp_models/{model_filename}'
        local_scaler_path = f'temp_scalers/{scaler_filename}'

        model.save(local_model_path)
        with open(local_scaler_path, 'wb') as f:
            pickle.dump(scaler, f)

        # Upload to Supabase Storage
        print(f"    Uploading model to Supabase...")
        try:
            model_url = SupabaseStorage.upload_file_to_storage(
                local_model_path,
                STORAGE_BUCKET,
                f'saved_models/{model_filename}'
            )
            uploaded_files['models'].append(model_filename)
            print(f"    ✓ Model uploaded")
        except Exception as e:
            print(f"    ⚠ Model upload failed: {e}")
            model_url = None

        print(f"    Uploading scaler to Supabase...")
        try:
            scaler_url = SupabaseStorage.upload_file_to_storage(
                local_scaler_path,
                STORAGE_BUCKET,
                f'saved_scalers/{scaler_filename}'
            )
            uploaded_files['scalers'].append(scaler_filename)
            print(f"    ✓ Scaler uploaded")
        except Exception as e:
            print(f"    ⚠ Scaler upload failed: {e}")
            scaler_url = None

        # Store results
        model_results[location_id] = {
            'library_name': location_name,
            'metrics': metrics,
            'data_stats': {
                'total_records': len(df_lib),
                'total_hours': len(occupancy),
                'avg_occupancy': float(occupancy.mean()),
                'max_occupancy': int(occupancy.max())
            },
            'supabase_model_path': f'saved_models/{model_filename}',
            'supabase_scaler_path': f'saved_scalers/{scaler_filename}',
            'model_url': model_url,
            'scaler_url': scaler_url,
            'trained_at': datetime.utcnow().isoformat()
        }

        # Save model metadata to Supabase database
        try:
            SupabaseStorage.upload_model_metadata(model_type, location_id, metrics)
            print(f"    ✓ Model metadata saved to database")
        except Exception as e:
            print(f"    ⚠ Metadata save failed: {e}")

    all_results[model_type] = {
        'model_name': model_name,
        'libraries': model_results
    }

# ============================================
# SAVE RESULTS TO SUPABASE
# ============================================

print("\n" + "=" * 80)
print("SAVING RESULTS TO SUPABASE")
print("=" * 80)

# Save results JSON locally first
results_filename = 'all_model_types_results.json'
local_results_path = f'temp_results/{results_filename}'

with open(local_results_path, 'w') as f:
    json.dump(all_results, f, indent=2)

# Upload results to Supabase Storage
print("\nUploading results JSON to Supabase...")
try:
    results_url = SupabaseStorage.upload_file_to_storage(
        local_results_path,
        STORAGE_BUCKET,
        f'model_results/{results_filename}'
    )
    uploaded_files['results'].append(results_filename)
    print(f"✓ Results uploaded: {results_url}")
except Exception as e:
    print(f"⚠ Results upload failed: {e}")

# Create comparison table
print("\n" + "=" * 80)
print("MODEL PERFORMANCE SUMMARY")
print("=" * 80)

for model_type, model_data in all_results.items():
    print(f"\n{model_data['model_name']}:")
    print(f"{'Library':<25} {'R²':<8} {'RMSE':<10} {'MAE':<10}")
    print("-" * 60)
    for lib_id, lib_data in model_data['libraries'].items():
        print(f"{lib_data['library_name']:<25} "
              f"{lib_data['metrics']['r2']:<8.4f} "
              f"{lib_data['metrics']['rmse']:<10.2f} "
              f"{lib_data['metrics']['mae']:<10.2f}")

# ============================================
# UPLOAD SUMMARY
# ============================================

print("\n" + "=" * 80)
print("SUPABASE UPLOAD SUMMARY")
print("=" * 80)
print(f"✓ Models uploaded: {len(uploaded_files['models'])}")
print(f"✓ Scalers uploaded: {len(uploaded_files['scalers'])}")
print(f"✓ Results uploaded: {len(uploaded_files['results'])}")

print("\nFiles in Supabase Storage:")
print(f"  Bucket: {STORAGE_BUCKET}")
print(f"  Location: saved_models/ ({len(uploaded_files['models'])} files)")
print(f"  Location: saved_scalers/ ({len(uploaded_files['scalers'])} files)")
print(f"  Location: model_results/ ({len(uploaded_files['results'])} files)")

print("\n" + "=" * 80)
print(f"✅ TRAINING COMPLETED - All files saved to Supabase!")
print("=" * 80)

# Clean up temporary files
print("\nCleaning up temporary files...")
import shutil
shutil.rmtree('temp_models', ignore_errors=True)
shutil.rmtree('temp_scalers', ignore_errors=True)
shutil.rmtree('temp_results', ignore_errors=True)
print("✓ Temporary files cleaned")
