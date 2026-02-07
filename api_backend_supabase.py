"""
Flask API Backend for Library Occupancy Predictions (Supabase Version)
Serves real-time predictions to React dashboard with Supabase storage
"""

from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pickle
import os
import json
from werkzeug.utils import secure_filename
from supabase_config import SupabaseStorage

# Upload configuration
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

ALLOWED_EXTENSIONS = {'csv'}
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from apscheduler.schedulers.background import BackgroundScheduler
import threading
import logging

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================
# CONFIGURATION
# ============================================

UPDATE_INTERVAL = 60  # seconds
SEQUENCE_LENGTH = 24  # Must match training

# Global cache for predictions
predictions_cache = {}
last_update_time = None
cache_lock = threading.Lock()

# Global models and scalers (keyed by model_type_location)
models_cache = {}
scalers_cache = {}

# Model types and libraries
MODEL_TYPES = {
    'lstm_only': 'LSTM Only',
    'cnn_only': 'CNN Only',
    'hybrid_cnn_lstm': 'Hybrid CNN-LSTM',
    'advanced_cnn_lstm': 'Advanced CNN-LSTM'
}

LIBRARY_IDS = {
    'miguel_pro': 'Miguel Pro Library',
    'american_corner': 'American Corner',
    'gisbert_2nd': 'Gisbert 2nd Floor',
    'gisbert_3rd': 'Gisbert 3rd Floor',
    'gisbert_4th': 'Gisbert 4th Floor',
    'gisbert_5th': 'Gisbert 5th Floor'
}

# ============================================
# MODEL LOADING (Still local - models are files)
# ============================================

def load_all_models():
    """Load all trained models and scalers for all model types and libraries"""
    global models_cache, scalers_cache

    logger.info("Loading all trained models...")
    loaded_count = 0

    for model_type in MODEL_TYPES.keys():
        for library_id in LIBRARY_IDS.keys():
            model_key = f"{model_type}_{library_id}"
            model_path = f"saved_models/{model_key}_model.keras"
            scaler_path = f"saved_scalers/{model_key}_scaler.pkl"

            try:
                if os.path.exists(model_path) and os.path.exists(scaler_path):
                    # Load model
                    model = load_model(model_path)
                    models_cache[model_key] = model

                    # Load scaler
                    with open(scaler_path, 'rb') as f:
                        scaler = pickle.load(f)
                    scalers_cache[model_key] = scaler

                    loaded_count += 1
                else:
                    logger.warning(f"Missing files for {model_key}")
            except Exception as e:
                logger.error(f"Error loading {model_key}: {e}")

    logger.info(f"✓ Loaded {loaded_count} models successfully")
    return loaded_count > 0

# ============================================
# DATA LOADING - USING SUPABASE
# ============================================

def load_historical_data():
    """Load and process historical data from Supabase"""
    try:
        # Get data from Supabase
        df = SupabaseStorage.get_wifi_data()

        if df.empty:
            logger.warning("No data in Supabase. Try uploading data first.")
            return None

        logger.info(f"Loaded {len(df)} rows from Supabase")

        df['Start_dt'] = pd.to_datetime(df['Start_dt'])
        df.set_index('Start_dt', inplace=True)

        # Add location mapping
        from ap_location_mapping import get_location_from_ap
        df['Location'] = df['AP MAC'].apply(get_location_from_ap)

        logger.info(f"Date range: {df.index.min()} to {df.index.max()}")
        return df
    except Exception as e:
        logger.error(f"Error loading data from Supabase: {e}")
        return None

def get_library_occupancy(df, library_id, hours=None):
    """Get occupancy time series for a specific library"""
    if df is None or df.empty:
        logger.error("No data available")
        return None

    if 'Location' not in df.columns:
        logger.error("Location column not found in data")
        return None

    df_lib = df[df['Location'] == library_id]

    if len(df_lib) == 0:
        logger.warning(f"No data for library {library_id}")
        return None

    occupancy = df_lib['Client MAC'].resample('h').nunique()
    occupancy = occupancy.fillna(0)

    logger.info(f"Library {library_id}: {len(occupancy)} hours, current={occupancy.iloc[-1] if len(occupancy) > 0 else 'N/A'}")

    return occupancy if hours is None else occupancy.tail(hours)

def predict_for_current_time(occupancy_series, library_id, hours_ahead=6):
    """Predict for the CURRENT real-world time using historical patterns"""
    if occupancy_series is None or len(occupancy_series) == 0:
        return None

    now = datetime.now()
    current_hour = now.hour
    current_day = now.weekday()

    logger.info(f"Predicting for {library_id} - Current time: {now.strftime('%A %I:%M %p')} (day={current_day}, hour={current_hour})")

    df = pd.DataFrame({'occupancy': occupancy_series})
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek

    predictions = []

    for i in range(hours_ahead):
        target_hour = (current_hour + i) % 24
        target_day = (current_day + (current_hour + i) // 24) % 7

        matching = df[(df['hour'] == target_hour) & (df['day_of_week'] == target_day)]

        if len(matching) > 0:
            recent_matching = matching.tail(8)
            values = recent_matching['occupancy'].values

            if len(values) >= 4:
                median_val = np.median(values)
                if median_val > 10:
                    filtered = values[values > median_val * 0.1]
                    if len(filtered) >= 3:
                        avg_occupancy = filtered.mean()
                    else:
                        avg_occupancy = recent_matching['occupancy'].mean()
                else:
                    avg_occupancy = recent_matching['occupancy'].mean()
            else:
                avg_occupancy = recent_matching['occupancy'].mean()

            predicted = max(0, int(avg_occupancy))
            predictions.append(predicted)
            logger.info(f"  Hour +{i} ({target_hour}:00): {predicted} users")
        else:
            hour_avg = df[df['hour'] == target_hour]['occupancy'].mean()
            predicted = max(0, int(hour_avg))
            predictions.append(predicted)
            logger.info(f"  Hour +{i} ({target_hour}:00): {predicted} users (fallback)")

    return predictions

def predict_with_specific_model(model_type, library_id, occupancy_series, hours_ahead=6):
    """Predict using pattern-based approach"""
    return predict_for_current_time(occupancy_series, library_id, hours_ahead)

# ============================================
# CACHE MANAGEMENT - USING SUPABASE
# ============================================

def update_predictions():
    """Update predictions for all libraries and model types - saves to Supabase"""
    global predictions_cache, last_update_time

    with cache_lock:
        logger.info("🔄 Starting predictions update...")

        df = load_historical_data()
        if df is None:
            logger.error("Cannot update predictions: no data available")
            return

        new_cache = {}
        update_time = datetime.now()

        for model_type in MODEL_TYPES.keys():
            new_cache[model_type] = {}

            for library_id in LIBRARY_IDS.keys():
                occupancy = get_library_occupancy(df, library_id)

                if occupancy is not None and len(occupancy) >= SEQUENCE_LENGTH:
                    predictions = predict_with_specific_model(
                        model_type, library_id, occupancy, hours_ahead=6
                    )

                    if predictions is not None:
                        new_cache[model_type][library_id] = {
                            'predictions': predictions,
                            'current_occupancy': int(occupancy.iloc[-1]),
                            'timestamp': update_time.isoformat()
                        }

        predictions_cache = new_cache
        last_update_time = update_time

        # Save to Supabase
        try:
            SupabaseStorage.save_predictions_cache(predictions_cache)
            logger.info(f"✅ Predictions updated and saved to Supabase at {update_time.strftime('%H:%M:%S')}")
        except Exception as e:
            logger.error(f"Error saving predictions to Supabase: {e}")

def load_predictions_cache():
    """Load predictions from Supabase on startup"""
    global predictions_cache, last_update_time

    try:
        cached = SupabaseStorage.get_predictions_cache()
        if cached:
            predictions_cache = cached
            last_update_time = datetime.now()
            logger.info("✅ Loaded predictions cache from Supabase")
            return True
    except Exception as e:
        logger.error(f"Error loading cache from Supabase: {e}")

    return False

# ============================================
# API ENDPOINTS
# ============================================

@app.route('/api/predictions', methods=['GET'])
def get_predictions():
    """Get all predictions for all model types and libraries"""
    model_type = request.args.get('model_type', 'advanced_cnn_lstm')

    with cache_lock:
        if model_type in predictions_cache:
            return jsonify({
                'predictions': predictions_cache[model_type],
                'last_update': last_update_time.isoformat() if last_update_time else None,
                'model_type': model_type
            })
        else:
            return jsonify({'error': f'No predictions for model type: {model_type}'}), 404

@app.route('/api/predictions/<library_id>', methods=['GET'])
def get_library_prediction(library_id):
    """Get predictions for a specific library"""
    model_type = request.args.get('model_type', 'advanced_cnn_lstm')

    with cache_lock:
        if model_type in predictions_cache and library_id in predictions_cache[model_type]:
            return jsonify({
                'library_id': library_id,
                'data': predictions_cache[model_type][library_id],
                'last_update': last_update_time.isoformat() if last_update_time else None
            })
        else:
            return jsonify({'error': f'No predictions for {library_id}'}), 404

@app.route('/api/model-types', methods=['GET'])
def get_model_types():
    """Get available model types"""
    return jsonify({'model_types': MODEL_TYPES})

@app.route('/api/libraries', methods=['GET'])
def get_libraries():
    """Get available libraries"""
    return jsonify({'libraries': LIBRARY_IDS})

@app.route('/api/upload', methods=['POST'])
def upload_data():
    """Upload new WiFi data CSV to Supabase"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if file and file.filename.endswith('.csv'):
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        try:
            # Read and validate
            df = pd.read_csv(filepath)
            required_cols = ['AP MAC', 'Client MAC', 'Start_dt']

            if not all(col in df.columns for col in required_cols):
                os.remove(filepath)
                return jsonify({'error': f'CSV must contain: {required_cols}'}), 400

            # Clean and upload to Supabase
            df['Start_dt'] = pd.to_datetime(df['Start_dt'])
            df = df.drop_duplicates()
            df = df.sort_values('Start_dt')

            count = SupabaseStorage.save_wifi_data(df)

            # Clean up local file
            os.remove(filepath)

            # Update predictions
            update_predictions()

            return jsonify({
                'message': f'Successfully uploaded {count} records to Supabase',
                'rows': count
            })

        except Exception as e:
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': str(e)}), 500

    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/api/status', methods=['GET'])
def get_status():
    """Get system status"""
    return jsonify({
        'status': 'running',
        'models_loaded': len(models_cache),
        'last_update': last_update_time.isoformat() if last_update_time else None,
        'storage': 'Supabase',
        'libraries': len(LIBRARY_IDS),
        'model_types': len(MODEL_TYPES)
    })

# ============================================
# INITIALIZATION
# ============================================

if __name__ == '__main__':
    logger.info("=" * 60)
    logger.info("LIBRARY OCCUPANCY PREDICTION API (Supabase Version)")
    logger.info("=" * 60)

    # Load models (still local files)
    load_all_models()

    # Try to load cached predictions from Supabase
    if not load_predictions_cache():
        logger.info("No cache found, generating initial predictions...")
        update_predictions()

    # Set up scheduler for automatic updates
    scheduler = BackgroundScheduler()
    scheduler.add_job(func=update_predictions, trigger="interval", seconds=UPDATE_INTERVAL)
    scheduler.start()

    logger.info(f"✓ Scheduler started (updates every {UPDATE_INTERVAL}s)")
    logger.info("=" * 60)

    # Run Flask app
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
