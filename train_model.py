"""
Train CNN-LSTM Model for Library Occupancy Prediction
Trains on all_data_cleaned.csv and saves model + scaler for backend API
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Flatten, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import r2_score, mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt
import pickle
import os
from datetime import datetime

print("=" * 70)
print("LIBRARY OCCUPANCY PREDICTION - MODEL TRAINING")
print("=" * 70)

# ============================================
# CONFIGURATION
# ============================================

DATA_FILE = 'all_data_cleaned.csv'
SEQUENCE_LENGTH = 24  # Use past 24 hours to predict next hour
TEST_SIZE = 0.2
VALIDATION_SPLIT = 0.1
EPOCHS = 100
BATCH_SIZE = 32

# Create directories for saving
os.makedirs('saved_models', exist_ok=True)
os.makedirs('saved_scalers', exist_ok=True)
os.makedirs('training_results', exist_ok=True)

# ============================================
# 1. LOAD AND PREPROCESS DATA
# ============================================

print("\n1. Loading data...")
df = pd.read_csv(DATA_FILE)
print(f"   ✓ Loaded {len(df):,} records")
print(f"   ✓ Columns: {df.columns.tolist()[:5]}...")

# Convert timestamps and set index
print("\n2. Processing timestamps...")
df['Start_dt'] = pd.to_datetime(df['Start_dt'])
df.set_index('Start_dt', inplace=True)
print(f"   ✓ Date range: {df.index.min()} to {df.index.max()}")

# Calculate hourly occupancy (unique users per hour)
print("\n3. Calculating hourly occupancy...")
occupancy = df['Client MAC'].resample('h').nunique()
occupancy = occupancy.fillna(0)
print(f"   ✓ Total hours: {len(occupancy)}")
print(f"   ✓ Min users/hour: {int(occupancy.min())}")
print(f"   ✓ Max users/hour: {int(occupancy.max())}")
print(f"   ✓ Average users/hour: {int(occupancy.mean())}")

# ============================================
# 2. CREATE SEQUENCES
# ============================================

print(f"\n4. Creating sequences (sequence length = {SEQUENCE_LENGTH})...")
data = occupancy.values.reshape(-1, 1)

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Create sequences
X, y = [], []
for i in range(len(scaled_data) - SEQUENCE_LENGTH):
    X.append(scaled_data[i:i+SEQUENCE_LENGTH])
    y.append(scaled_data[i+SEQUENCE_LENGTH])

X = np.array(X)
y = np.array(y)

print(f"   ✓ Created {len(X):,} sequences")
print(f"   ✓ X shape: {X.shape}")
print(f"   ✓ y shape: {y.shape}")

# ============================================
# 3. SPLIT DATA
# ============================================

print(f"\n5. Splitting data (test size = {TEST_SIZE})...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, shuffle=False
)
print(f"   ✓ Training samples: {len(X_train):,}")
print(f"   ✓ Testing samples: {len(X_test):,}")

# ============================================
# 4. BUILD CNN-LSTM MODEL
# ============================================

print("\n6. Building CNN-LSTM model...")
model = Sequential([
    Input(shape=(SEQUENCE_LENGTH, 1)),

    # CNN layers to extract features
    Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'),
    MaxPooling1D(pool_size=2),
    Dropout(0.2),

    Conv1D(filters=32, kernel_size=3, activation='relu', padding='same'),
    MaxPooling1D(pool_size=2),
    Dropout(0.2),

    # Reshape for LSTM
    # After two MaxPooling1D(2), sequence_length becomes sequence_length//4
    # We'll flatten and reshape it for LSTM

    # LSTM layers to learn temporal patterns
    LSTM(units=50, return_sequences=True, activation='relu'),
    Dropout(0.2),

    LSTM(units=50, activation='relu'),
    Dropout(0.2),

    # Dense layers for prediction
    Dense(25, activation='relu'),
    Dense(1)  # Output: predicted occupancy for next hour
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='mean_squared_error',
    metrics=['mae']
)

print("\n   Model Architecture:")
model.summary()

# ============================================
# 5. TRAIN MODEL
# ============================================

print("\n7. Training model...")
print(f"   Epochs: {EPOCHS}")
print(f"   Batch size: {BATCH_SIZE}")
print(f"   This may take a few minutes...\n")

# Callbacks
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=0.00001,
    verbose=1
)

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=VALIDATION_SPLIT,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

print("\n   ✓ Training completed!")

# ============================================
# 6. EVALUATE MODEL
# ============================================

print("\n8. Evaluating model on test set...")

# Make predictions
predictions_scaled = model.predict(X_test, verbose=0)

# Inverse transform to get actual occupancy values
predictions = scaler.inverse_transform(predictions_scaled)
y_test_actual = scaler.inverse_transform(y_test)

# Flatten for metric calculation
y_true = y_test_actual.flatten()
y_pred = predictions.flatten()

# Calculate metrics
r2 = r2_score(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
rmse = sqrt(mse)
mae = np.mean(np.abs(y_true - y_pred))

print(f"\n   --- Model Performance Metrics ---")
print(f"   R² Score:  {r2:.4f} (1.0 is perfect)")
print(f"   MSE:       {mse:.2f}")
print(f"   RMSE:      {rmse:.2f} users")
print(f"   MAE:       {mae:.2f} users")

# ============================================
# 7. SAVE MODEL AND SCALER
# ============================================

print("\n9. Saving model and scaler...")

# Save model
model_path = 'saved_models/cnn_lstm_model.keras'
model.save(model_path)
print(f"   ✓ Model saved to: {model_path}")

# Save scaler
scaler_path = 'saved_scalers/occupancy_scaler.pkl'
with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)
print(f"   ✓ Scaler saved to: {scaler_path}")

# Save training history
history_path = 'training_results/training_history.pkl'
with open(history_path, 'wb') as f:
    pickle.dump(history.history, f)
print(f"   ✓ Training history saved to: {history_path}")

# Save metrics
metrics = {
    'r2_score': r2,
    'mse': mse,
    'rmse': rmse,
    'mae': mae,
    'training_date': datetime.now().isoformat(),
    'data_file': DATA_FILE,
    'sequence_length': SEQUENCE_LENGTH,
    'total_samples': len(X),
    'training_samples': len(X_train),
    'testing_samples': len(X_test)
}

metrics_path = 'training_results/model_metrics.pkl'
with open(metrics_path, 'wb') as f:
    pickle.dump(metrics, f)
print(f"   ✓ Metrics saved to: {metrics_path}")

# ============================================
# 8. VISUALIZE RESULTS
# ============================================

print("\n10. Creating visualizations...")

# Plot 1: Training History
fig, axes = plt.subplots(2, 1, figsize=(12, 10))

# Loss
axes[0].plot(history.history['loss'], label='Training Loss', color='blue', alpha=0.7)
axes[0].plot(history.history['val_loss'], label='Validation Loss', color='red', alpha=0.7)
axes[0].set_title('Model Loss Over Epochs', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss (MSE)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# MAE
axes[1].plot(history.history['mae'], label='Training MAE', color='blue', alpha=0.7)
axes[1].plot(history.history['val_mae'], label='Validation MAE', color='red', alpha=0.7)
axes[1].set_title('Model MAE Over Epochs', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('MAE')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_results/training_history.png', dpi=300, bbox_inches='tight')
print(f"   ✓ Training history plot saved")
plt.close()

# Plot 2: Predictions vs Actual
plt.figure(figsize=(15, 6))

# Plot only first 200 points for clarity
n_points = min(200, len(y_true))
x_axis = range(n_points)

plt.plot(x_axis, y_true[:n_points], label='Actual Occupancy',
         color='blue', linewidth=2, alpha=0.7)
plt.plot(x_axis, y_pred[:n_points], label='Predicted Occupancy',
         color='red', linewidth=2, linestyle='--', alpha=0.7)

plt.title(f'CNN-LSTM Model: Actual vs Predicted Occupancy (First {n_points} samples)',
          fontsize=14, fontweight='bold')
plt.xlabel('Time Step')
plt.ylabel('Number of Users')
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)

# Add metrics text box
textstr = f'R² = {r2:.4f}\nRMSE = {rmse:.2f} users\nMAE = {mae:.2f} users'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=11,
         verticalalignment='top', bbox=props)

plt.tight_layout()
plt.savefig('training_results/predictions_vs_actual.png', dpi=300, bbox_inches='tight')
print(f"   ✓ Predictions plot saved")
plt.close()

# Plot 3: Error Distribution
plt.figure(figsize=(10, 6))
errors = y_true - y_pred
plt.hist(errors, bins=50, color='purple', alpha=0.7, edgecolor='black')
plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
plt.title('Prediction Error Distribution', fontsize=14, fontweight='bold')
plt.xlabel('Error (Actual - Predicted)')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('training_results/error_distribution.png', dpi=300, bbox_inches='tight')
print(f"   ✓ Error distribution plot saved")
plt.close()

# ============================================
# 9. TEST PREDICTION
# ============================================

print("\n11. Testing prediction with latest data...")

# Use the last SEQUENCE_LENGTH hours to predict the next hour
latest_sequence = scaled_data[-SEQUENCE_LENGTH:].reshape(1, SEQUENCE_LENGTH, 1)
predicted_next_scaled = model.predict(latest_sequence, verbose=0)
predicted_next = scaler.inverse_transform(predicted_next_scaled)[0][0]

current_occupancy = occupancy.iloc[-1]

print(f"\n   Current occupancy: {int(current_occupancy)} users")
print(f"   Predicted next hour: {int(predicted_next)} users")
print(f"   Change: {int(predicted_next - current_occupancy):+d} users")

# ============================================
# SUMMARY
# ============================================

print("\n" + "=" * 70)
print("TRAINING COMPLETED SUCCESSFULLY!")
print("=" * 70)
print("\nSaved Files:")
print(f"  • Model:           {model_path}")
print(f"  • Scaler:          {scaler_path}")
print(f"  • Metrics:         {metrics_path}")
print(f"  • Training plots:  training_results/")
print("\nModel Performance:")
print(f"  • R² Score:  {r2:.4f}")
print(f"  • RMSE:      {rmse:.2f} users")
print(f"  • MAE:       {mae:.2f} users")
print("\nNext Steps:")
print("  1. Run 'python api_backend.py' to start the backend API")
print("  2. The API will automatically load this trained model")
print("  3. Access the dashboard at http://localhost:3000")
print("=" * 70)
