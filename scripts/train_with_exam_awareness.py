"""
Exam-Aware Model Training
Trains models that use historical exam period patterns for predictions during exams
Run from project root: python scripts/train_with_exam_awareness.py
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
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Dense, LSTM, Conv1D, MaxPooling1D,
                                     Dropout, Input, concatenate, Multiply,
                                     Lambda, BatchNormalization, Bidirectional)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from math import sqrt
import os
import json
from exam_period_tagger import ExamPeriodTagger
from ap_location_mapping import get_location_from_ap

print("=" * 80)
print("EXAM-AWARE MODEL TRAINING")
print("=" * 80)

# Configuration
SEQUENCE_LENGTH = 24
TEST_SIZE = 0.15
VALIDATION_SPLIT = 0.15
EPOCHS = 200
BATCH_SIZE = 64
INITIAL_LR = 0.002

# Create directories
os.makedirs('exam_aware_results', exist_ok=True)
os.makedirs('thesis_figures', exist_ok=True)

def calculate_metrics(y_true, y_pred):
    """Calculate comprehensive metrics"""
    r2 = r2_score(y_true, y_pred)
    if r2 < 0:
        r2 = 0
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

def build_exam_aware_model(sequence_length, num_exam_features):
    """
    Exam-Aware Model: Uses exam period indicator as auxiliary feature
    """
    # Sequence input branch
    sequence_input = Input(shape=(sequence_length, 1), name='sequence_input')

    # Exam context features
    exam_input = Input(shape=(num_exam_features,), name='exam_input')

    # CNN layers for sequence processing
    x = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(sequence_input)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.2)(x)

    x = Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.2)(x)

    # LSTM layers
    x = LSTM(units=50, return_sequences=True)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    lstm_out = LSTM(units=50, return_sequences=True)(x)
    lstm_out = BatchNormalization()(lstm_out)

    # Attention mechanism
    attention = Dense(1, activation='tanh')(lstm_out)
    attention = Lambda(lambda x: K.squeeze(x, -1))(attention)
    attention = Dense(lstm_out.shape[1], activation='softmax')(attention)
    attention = Lambda(lambda x: K.expand_dims(x, -1))(attention)

    # Apply attention
    context = Multiply()([lstm_out, attention])
    context = Lambda(lambda x: K.sum(x, axis=1))(context)

    # Process exam features
    exam_features = Dense(16, activation='relu')(exam_input)
    exam_features = BatchNormalization()(exam_features)
    exam_features = Dropout(0.2)(exam_features)

    # Combine sequence and exam context
    combined = concatenate([context, exam_features])

    # Final dense layers
    x = Dense(32, activation='relu')(combined)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    # Output layer
    output = Dense(1, name='output')(x)

    # Create and compile model
    model = Model(inputs=[sequence_input, exam_input], outputs=output)
    model.compile(
        optimizer=Adam(learning_rate=INITIAL_LR),
        loss='mean_squared_error',
        metrics=['mae', 'mse']
    )

    return model

def build_baseline_model(sequence_length):
    """Baseline model without exam awareness"""
    sequence_input = Input(shape=(sequence_length, 1), name='sequence_input')

    # CNN layers
    x = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(sequence_input)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.2)(x)

    x = Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.2)(x)

    # LSTM layers
    x = LSTM(units=50, return_sequences=True, activation='relu')(x)
    x = Dropout(0.2)(x)

    lstm_out = LSTM(units=50, return_sequences=True, activation='relu')(x)

    # Attention mechanism
    attention = Dense(1, activation='tanh')(lstm_out)
    attention = Lambda(lambda x: K.squeeze(x, -1))(attention)
    attention = Dense(lstm_out.shape[1], activation='softmax')(attention)
    attention = Lambda(lambda x: K.expand_dims(x, -1))(attention)

    # Apply attention
    context = Multiply()([lstm_out, attention])
    context = Lambda(lambda x: K.sum(x, axis=1))(context)

    # Output
    x = Dense(25, activation='relu')(context)
    output = Dense(1, name='output')(x)

    model = Model(inputs=sequence_input, outputs=output)
    model.compile(
        optimizer=Adam(learning_rate=INITIAL_LR),
        loss='mean_squared_error',
        metrics=['mae', 'mse']
    )

    return model

# Load and tag data
print("\n1. Loading and tagging data...")
tagger = ExamPeriodTagger()

df = pd.read_csv('all_data_cleaned.csv')
df['Location'] = df['AP MAC'].apply(get_location_from_ap)
df['Start_dt'] = pd.to_datetime(df['Start_dt'])

# Tag exam periods
print("\n2. Tagging exam periods...")
df = tagger.tag_dataframe(df, date_column='Start_dt')

# Set index after tagging
df.set_index('Start_dt', inplace=True)

print(f"   ✓ Loaded and tagged {len(df):,} records")

# Select library
print("\n3. Preparing data for Miguel Pro Library...")
df_lib = df[df['Location'] == 'miguel_pro'].copy()

# Calculate occupancy
occupancy = df_lib['Client MAC'].resample('h').nunique().fillna(0)
occupancy_df = occupancy.to_frame('occupancy')

# Resample exam tags (take max to preserve 1s)
exam_tags = df_lib[['is_exam_period', 'is_pre_exam_period']].resample('h').max().fillna(0)
occupancy_df = occupancy_df.join(exam_tags)

print(f"   Hours of data: {len(occupancy_df)}")
print(f"   Exam period hours: {occupancy_df['is_exam_period'].sum()}")
print(f"   Pre-exam period hours: {occupancy_df['is_pre_exam_period'].sum()}")

# Normalize occupancy
scaler = MinMaxScaler(feature_range=(0, 1))
occupancy_df['occupancy_scaled'] = scaler.fit_transform(occupancy_df[['occupancy']])

# Exam features
exam_features = ['is_exam_period', 'is_pre_exam_period']

# Create sequences
print("\n4. Creating sequences with exam context...")

X_seq, X_exam, y = [], [], []

for i in range(len(occupancy_df) - SEQUENCE_LENGTH):
    # Sequence features
    seq = occupancy_df['occupancy_scaled'].iloc[i:i+SEQUENCE_LENGTH].values
    X_seq.append(seq)

    # Exam features at prediction time
    exam = occupancy_df[exam_features].iloc[i+SEQUENCE_LENGTH].values
    X_exam.append(exam)

    # Target value
    y.append(occupancy_df['occupancy_scaled'].iloc[i+SEQUENCE_LENGTH])

X_seq = np.array(X_seq).reshape(-1, SEQUENCE_LENGTH, 1)
X_exam = np.array(X_exam)
y = np.array(y)

print(f"   ✓ Created {len(X_seq)} sequences")

# Split data
X_seq_train, X_seq_test, X_exam_train, X_exam_test, y_train, y_test = train_test_split(
    X_seq, X_exam, y, test_size=TEST_SIZE, shuffle=False
)

print(f"   Train: {len(X_seq_train)}, Test: {len(X_seq_test)}")

# Identify exam periods in test set
test_exam_mask = X_exam_test[:, 0] == 1  # is_exam_period
test_regular_mask = X_exam_test[:, 0] == 0

print(f"   Test exam periods: {test_exam_mask.sum()}")
print(f"   Test regular periods: {test_regular_mask.sum()}")

# Callbacks
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=30,
        restore_best_weights=True,
        verbose=1,
        min_delta=1e-4
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.3,
        patience=15,
        min_lr=0.0000001,
        verbose=1,
        cooldown=5
    )
]

# ============================================
# TRAIN BASELINE MODEL (NO exam awareness)
# ============================================

print("\n5. Training BASELINE Model (no exam awareness)...")

baseline_model = build_baseline_model(SEQUENCE_LENGTH)
print(f"   Model parameters: {baseline_model.count_params():,}")

baseline_history = baseline_model.fit(
    X_seq_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=VALIDATION_SPLIT,
    callbacks=callbacks,
    verbose=1
)

# Evaluate
baseline_pred_scaled = baseline_model.predict(X_seq_test, verbose=0)
baseline_pred = scaler.inverse_transform(baseline_pred_scaled)
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# Overall metrics
baseline_metrics = calculate_metrics(y_test_actual.flatten(), baseline_pred.flatten())

# Exam period specific metrics
if test_exam_mask.sum() > 0:
    baseline_exam_metrics = calculate_metrics(
        y_test_actual[test_exam_mask].flatten(),
        baseline_pred[test_exam_mask].flatten()
    )
else:
    baseline_exam_metrics = None

# Regular period metrics
baseline_regular_metrics = calculate_metrics(
    y_test_actual[test_regular_mask].flatten(),
    baseline_pred[test_regular_mask].flatten()
)

print(f"\n   BASELINE Model Results:")
print(f"   Overall - R²: {baseline_metrics['r2']:.4f}, RMSE: {baseline_metrics['rmse']:.2f}, MAE: {baseline_metrics['mae']:.2f}")
if baseline_exam_metrics:
    print(f"   Exam Periods - R²: {baseline_exam_metrics['r2']:.4f}, RMSE: {baseline_exam_metrics['rmse']:.2f}, MAE: {baseline_exam_metrics['mae']:.2f}")
print(f"   Regular Periods - R²: {baseline_regular_metrics['r2']:.4f}, RMSE: {baseline_regular_metrics['rmse']:.2f}, MAE: {baseline_regular_metrics['mae']:.2f}")

# ============================================
# TRAIN EXAM-AWARE MODEL
# ============================================

print("\n6. Training EXAM-AWARE Model...")

exam_aware_model = build_exam_aware_model(SEQUENCE_LENGTH, len(exam_features))
print(f"   Model parameters: {exam_aware_model.count_params():,}")

exam_aware_history = exam_aware_model.fit(
    [X_seq_train, X_exam_train], y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=VALIDATION_SPLIT,
    callbacks=callbacks,
    verbose=1
)

# Evaluate
exam_aware_pred_scaled = exam_aware_model.predict([X_seq_test, X_exam_test], verbose=0)
exam_aware_pred = scaler.inverse_transform(exam_aware_pred_scaled)

# Overall metrics
exam_aware_metrics = calculate_metrics(y_test_actual.flatten(), exam_aware_pred.flatten())

# Exam period specific metrics
if test_exam_mask.sum() > 0:
    exam_aware_exam_metrics = calculate_metrics(
        y_test_actual[test_exam_mask].flatten(),
        exam_aware_pred[test_exam_mask].flatten()
    )
else:
    exam_aware_exam_metrics = None

# Regular period metrics
exam_aware_regular_metrics = calculate_metrics(
    y_test_actual[test_regular_mask].flatten(),
    exam_aware_pred[test_regular_mask].flatten()
)

print(f"\n   EXAM-AWARE Model Results:")
print(f"   Overall - R²: {exam_aware_metrics['r2']:.4f}, RMSE: {exam_aware_metrics['rmse']:.2f}, MAE: {exam_aware_metrics['mae']:.2f}")
if exam_aware_exam_metrics:
    print(f"   Exam Periods - R²: {exam_aware_exam_metrics['r2']:.4f}, RMSE: {exam_aware_exam_metrics['rmse']:.2f}, MAE: {exam_aware_exam_metrics['mae']:.2f}")
print(f"   Regular Periods - R²: {exam_aware_regular_metrics['r2']:.4f}, RMSE: {exam_aware_regular_metrics['rmse']:.2f}, MAE: {exam_aware_regular_metrics['mae']:.2f}")

# ============================================
# SAVE RESULTS
# ============================================

results = {
    'baseline_model': {
        'description': 'CNN-LSTM with Attention (NO exam awareness)',
        'overall_metrics': baseline_metrics,
        'exam_metrics': baseline_exam_metrics,
        'regular_metrics': baseline_regular_metrics,
        'parameters': int(baseline_model.count_params())
    },
    'exam_aware_model': {
        'description': 'CNN-LSTM with Attention + Exam Context',
        'overall_metrics': exam_aware_metrics,
        'exam_metrics': exam_aware_exam_metrics,
        'regular_metrics': exam_aware_regular_metrics,
        'parameters': int(exam_aware_model.count_params())
    }
}

with open('exam_aware_results/exam_aware_comparison.json', 'w') as f:
    json.dump(results, f, indent=2)

# Save models
baseline_model.save('exam_aware_results/baseline_model.keras')
exam_aware_model.save('exam_aware_results/exam_aware_model.keras')

print("\n✓ Models and results saved to exam_aware_results/")

# ============================================
# GENERATE VISUALIZATIONS
# ============================================

print("\n7. Generating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Overall performance comparison
metrics_names = ['R²', 'RMSE', 'MAE']
baseline_values = [baseline_metrics['r2']*100, baseline_metrics['rmse'], baseline_metrics['mae']]
exam_aware_values = [exam_aware_metrics['r2']*100, exam_aware_metrics['rmse'], exam_aware_metrics['mae']]

x = np.arange(len(metrics_names))
width = 0.35

axes[0, 0].bar(x - width/2, baseline_values, width, label='Baseline', color='#3498db', alpha=0.8)
axes[0, 0].bar(x + width/2, exam_aware_values, width, label='Exam-Aware', color='#e74c3c', alpha=0.8)
axes[0, 0].set_ylabel('Value', fontsize=11, fontweight='bold')
axes[0, 0].set_title('Overall Performance Comparison', fontsize=12, fontweight='bold')
axes[0, 0].set_xticks(x)
axes[0, 0].set_xticklabels(metrics_names)
axes[0, 0].legend()
axes[0, 0].grid(axis='y', alpha=0.3)

# 2. Exam period performance (if available)
if baseline_exam_metrics and exam_aware_exam_metrics:
    baseline_exam_values = [baseline_exam_metrics['r2']*100, baseline_exam_metrics['rmse'], baseline_exam_metrics['mae']]
    exam_aware_exam_values = [exam_aware_exam_metrics['r2']*100, exam_aware_exam_metrics['rmse'], exam_aware_exam_metrics['mae']]

    axes[0, 1].bar(x - width/2, baseline_exam_values, width, label='Baseline', color='#3498db', alpha=0.8)
    axes[0, 1].bar(x + width/2, exam_aware_exam_values, width, label='Exam-Aware', color='#e74c3c', alpha=0.8)
    axes[0, 1].set_ylabel('Value', fontsize=11, fontweight='bold')
    axes[0, 1].set_title('Exam Period Performance', fontsize=12, fontweight='bold')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(metrics_names)
    axes[0, 1].legend()
    axes[0, 1].grid(axis='y', alpha=0.3)
else:
    axes[0, 1].text(0.5, 0.5, 'No exam periods in test set', ha='center', va='center', fontsize=12)
    axes[0, 1].set_title('Exam Period Performance', fontsize=12, fontweight='bold')

# 3. Training history
axes[1, 0].plot(baseline_history.history['loss'], label='Baseline - Training', color='#3498db', linewidth=2)
axes[1, 0].plot(baseline_history.history['val_loss'], label='Baseline - Validation', color='#3498db', linewidth=2, linestyle='--')
axes[1, 0].plot(exam_aware_history.history['loss'], label='Exam-Aware - Training', color='#e74c3c', linewidth=2)
axes[1, 0].plot(exam_aware_history.history['val_loss'], label='Exam-Aware - Validation', color='#e74c3c', linewidth=2, linestyle='--')
axes[1, 0].set_xlabel('Epoch', fontsize=11, fontweight='bold')
axes[1, 0].set_ylabel('Loss', fontsize=11, fontweight='bold')
axes[1, 0].set_title('Training History Comparison', fontsize=12, fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# 4. Prediction scatter plot
axes[1, 1].scatter(y_test_actual, baseline_pred, alpha=0.5, label='Baseline', s=20, color='#3498db')
axes[1, 1].scatter(y_test_actual, exam_aware_pred, alpha=0.5, label='Exam-Aware', s=20, color='#e74c3c')
axes[1, 1].plot([y_test_actual.min(), y_test_actual.max()],
                [y_test_actual.min(), y_test_actual.max()],
                'k--', linewidth=2, label='Perfect Prediction')
axes[1, 1].set_xlabel('Actual Occupancy', fontsize=11, fontweight='bold')
axes[1, 1].set_ylabel('Predicted Occupancy', fontsize=11, fontweight='bold')
axes[1, 1].set_title('Prediction Accuracy', fontsize=12, fontweight='bold')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('thesis_figures/Exam_Aware_Model_Comparison.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved: thesis_figures/Exam_Aware_Model_Comparison.png")
plt.close()

print("\n" + "=" * 80)
print("EXAM-AWARE TRAINING COMPLETE!")
print("=" * 80)
print("\nGenerated files:")
print("  - exam_aware_results/exam_aware_comparison.json")
print("  - exam_aware_results/baseline_model.keras")
print("  - exam_aware_results/exam_aware_model.keras")
print("  - thesis_figures/Exam_Aware_Model_Comparison.png")
print("=" * 80)
