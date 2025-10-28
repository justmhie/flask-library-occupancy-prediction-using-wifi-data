"""
Flask API Backend for Library Occupancy Predictions
Serves real-time predictions to React dashboard
Includes automatic updates
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pickle
import os
import json
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

CLEANED_DATA_FILE = 'all_data_cleaned.csv'
MODEL_FILE = 'saved_models/cnn_lstm_model.keras'
SCALER_FILE = 'saved_scalers/occupancy_scaler.pkl'
CACHE_FILE = 'predictions_cache.pkl'
UPDATE_INTERVAL = 60  # seconds
SEQUENCE_LENGTH = 24  # Must match training

# Global cache for predictions
predictions_cache = {}
last_update_time = None
cache_lock = threading.Lock()

# Global model and scaler
model = None
scaler = None

# Library configurations
LIBRARIES = [
    'Miguel_Pro',
    'Gisbert_2nd_Floor',
    'American_Corner',
    'Gisbert_3rd_Floor',
    'Gisbert_4th_Floor',
    'Gisbert_5th_Floor'
]

# ============================================
# MODEL LOADING
# ============================================

def load_trained_model():
    """Load the trained CNN-LSTM model and scaler"""
    global model, scaler

    try:
        if os.path.exists(MODEL_FILE):
            logger.info(f"Loading trained model from {MODEL_FILE}...")
            model = load_model(MODEL_FILE)
            logger.info("✓ Model loaded successfully")
        else:
            logger.warning(f"Model file not found: {MODEL_FILE}")
            logger.warning("Will use simple statistical prediction instead")
            model = None

        if os.path.exists(SCALER_FILE):
            logger.info(f"Loading scaler from {SCALER_FILE}...")
            with open(SCALER_FILE, 'rb') as f:
                scaler = pickle.load(f)
            logger.info("✓ Scaler loaded successfully")
        else:
            logger.warning(f"Scaler file not found: {SCALER_FILE}")
            scaler = None

        return model is not None and scaler is not None

    except Exception as e:
        logger.error(f"Error loading model: {e}")
        model = None
        scaler = None
        return False

# ============================================
# PREDICTION FUNCTIONS
# ============================================

def load_historical_data():
    """Load and process historical data"""
    try:
        df = pd.read_csv(CLEANED_DATA_FILE)
        logger.info(f"Loaded {len(df)} rows from {CLEANED_DATA_FILE}")
        logger.info(f"Columns: {df.columns.tolist()}")

        df['Start_dt'] = pd.to_datetime(df['Start_dt'])
        df.set_index('Start_dt', inplace=True)

        logger.info(f"Date range: {df.index.min()} to {df.index.max()}")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return None

def get_library_occupancy(df, location=None, hours=168):
    """Get occupancy time series for a library"""
    if location and 'Location' in df.columns:
        df = df[df['Location'] == location]

    occupancy = df['Client MAC'].resample('h').nunique()
    occupancy = occupancy.fillna(0)

    logger.info(f"Generated occupancy series: {len(occupancy)} hours, min={occupancy.min()}, max={occupancy.max()}, current={occupancy.iloc[-1] if len(occupancy) > 0 else 'N/A'}")

    return occupancy.tail(hours)

def predict_with_model(occupancy_series, hours_ahead=6):
    """Predict using trained CNN-LSTM model"""
    global model, scaler

    if model is None or scaler is None:
        logger.warning("Model or scaler not loaded, using simple prediction")
        return predict_simple(occupancy_series, hours_ahead)

    if len(occupancy_series) < SEQUENCE_LENGTH:
        logger.warning(f"Need at least {SEQUENCE_LENGTH} hours, have {len(occupancy_series)}, using simple prediction")
        return predict_simple(occupancy_series, hours_ahead)

    try:
        # Get the last SEQUENCE_LENGTH hours
        recent_data = occupancy_series.tail(SEQUENCE_LENGTH).values.reshape(-1, 1)

        # Scale the data
        scaled_data = scaler.transform(recent_data)

        # Make predictions
        predictions = []
        current_sequence = scaled_data.copy()

        for _ in range(hours_ahead):
            # Reshape for model input: (1, SEQUENCE_LENGTH, 1)
            input_seq = current_sequence.reshape(1, SEQUENCE_LENGTH, 1)

            # Predict next hour
            pred_scaled = model.predict(input_seq, verbose=0)[0][0]

            # Inverse transform to get actual value
            pred_actual = scaler.inverse_transform([[pred_scaled]])[0][0]
            pred_actual = max(0, int(pred_actual))
            predictions.append(pred_actual)

            # Update sequence for next prediction
            current_sequence = np.roll(current_sequence, -1, axis=0)
            current_sequence[-1] = pred_scaled

        logger.info(f"Model prediction: {predictions}")
        return predictions

    except Exception as e:
        logger.error(f"Error in model prediction: {e}")
        return predict_simple(occupancy_series, hours_ahead)

def predict_simple(occupancy_series, hours_ahead=6):
    """Simple prediction using moving average with trend (fallback)"""
    if len(occupancy_series) == 0:
        return None

    predictions = []

    # Calculate trend
    if len(occupancy_series) >= 48:
        older_avg = occupancy_series.tail(48).head(24).mean()
        recent_avg = occupancy_series.tail(24).mean()
        trend = (recent_avg - older_avg) / 24  # hourly trend
    elif len(occupancy_series) >= 12:
        # Use half the available data for trend
        half = len(occupancy_series) // 2
        older_avg = occupancy_series.head(half).mean()
        recent_avg = occupancy_series.tail(half).mean()
        trend = (recent_avg - older_avg) / half
    else:
        trend = 0

    # Generate predictions
    current = occupancy_series.iloc[-1]
    for i in range(1, hours_ahead + 1):
        predicted = current + (trend * i)
        predicted = max(0, int(predicted))
        predictions.append(predicted)

    logger.info(f"Simple prediction: {predictions}")
    return predictions

def calculate_daily_pattern(occupancy_series, days=7):
    """Calculate average usage pattern by hour"""
    df = pd.DataFrame({'occupancy': occupancy_series})
    df['hour'] = df.index.hour
    df['date'] = df.index.date
    
    # Get last N days
    recent_dates = sorted(df['date'].unique())[-days:]
    df = df[df['date'].isin(recent_dates)]
    
    # Average by hour
    hourly_avg = df.groupby('hour')['occupancy'].agg(['mean', 'max']).reset_index()
    hourly_avg.columns = ['hour', 'average', 'peak']
    hourly_avg['average'] = hourly_avg['average'].round().astype(int)
    hourly_avg['peak'] = hourly_avg['peak'].astype(int)
    
    return hourly_avg.to_dict('records')

def generate_predictions_for_library(location):
    """Generate complete prediction data for a library"""
    df = load_historical_data()
    if df is None:
        return None

    # Get occupancy data
    occupancy = get_library_occupancy(df, location if location != 'all' else None)

    if len(occupancy) == 0:
        logger.warning(f"No data available for {location}")
        return None

    if len(occupancy) < 24:
        logger.warning(f"Insufficient data for {location}, only {len(occupancy)} hours available")
        # Still continue with whatever data we have
        pass
    
    # Current stats
    current = int(occupancy.iloc[-1])
    recent_24h = occupancy.tail(24).values
    avg_24h = int(recent_24h.mean())
    max_24h = int(recent_24h.max())
    peak_today = int(occupancy.tail(24).max())
    peak_time = occupancy.tail(24).idxmax().strftime('%I:%M %p')
    avg_7d = int(occupancy.tail(168).mean())
    
    # Predictions (use trained model if available)
    next_hour_predictions = predict_with_model(occupancy, hours_ahead=6)
    if next_hour_predictions is None:
        return None
    
    predicted_next = next_hour_predictions[0]
    change = predicted_next - current
    
    # Trend
    if len(occupancy) >= 48:
        recent_trend = occupancy.tail(24).mean() - occupancy.tail(48).head(24).mean()
        trend = 1 if recent_trend > 2 else -1 if recent_trend < -2 else 0
    else:
        trend = 0
    
    # Hourly data for chart (last 24 hours + predictions)
    hourly_data = []
    times = occupancy.tail(24).index
    for i, t in enumerate(times):
        hourly_data.append({
            'time': t.strftime('%I:%M %p'),
            'actual': int(occupancy.iloc[-(24-i)]),
            'predicted': None
        })
    
    # Add predicted hours
    current_time = occupancy.index[-1]
    for i, pred in enumerate(next_hour_predictions):
        future_time = current_time + timedelta(hours=i+1)
        hourly_data.append({
            'time': future_time.strftime('%I:%M %p'),
            'actual': None,
            'predicted': pred
        })
    
    # Next hours forecast
    next_hours = []
    for i, pred in enumerate(next_hour_predictions):
        future_time = current_time + timedelta(hours=i+1)
        # Simple confidence interval (±15%)
        next_hours.append({
            'time': future_time.strftime('%I %p'),
            'predicted': pred,
            'confidence_upper': int(pred * 1.15),
            'confidence_lower': max(0, int(pred * 0.85))
        })
    
    # Daily pattern
    daily_pattern = calculate_daily_pattern(occupancy)
    
    # Estimate max capacity (use historical max * 1.2)
    max_capacity = int(occupancy.max() * 1.2)
    
    return {
        'location': location,
        'current': current,
        'predicted': predicted_next,
        'change': change,
        'avg_24h': avg_24h,
        'max_24h': max_24h,
        'max_capacity': max_capacity,
        'peak_today': peak_today,
        'peak_time': peak_time,
        'avg_7d': avg_7d,
        'trend': trend,
        'model_accuracy': 85,  # Mock value - replace with actual R² score
        'hourly_data': hourly_data,
        'daily_pattern': daily_pattern,
        'next_hours': next_hours,
        'last_updated': datetime.now().isoformat()
    }

def update_predictions_cache():
    """Update predictions cache for all libraries"""
    global predictions_cache, last_update_time
    
    logger.info("Updating predictions cache...")
    
    df = load_historical_data()
    if df is None:
        logger.error("Failed to load data")
        return
    
    has_location = 'Location' in df.columns
    
    with cache_lock:
        new_cache = {}
        
        # Generate predictions for each library
        if has_location:
            for location in LIBRARIES:
                try:
                    pred = generate_predictions_for_library(location)
                    if pred:
                        new_cache[location] = pred
                        logger.info(f"Updated predictions for {location}")
                except Exception as e:
                    logger.error(f"Error generating predictions for {location}: {e}")
        else:
            # Single prediction for all data
            try:
                pred = generate_predictions_for_library('all')
                if pred:
                    new_cache['all'] = pred
                    logger.info("Updated predictions for all libraries")
            except Exception as e:
                logger.error(f"Error generating predictions: {e}")
        
        # Calculate overall stats
        if new_cache:
            total_current = sum(p['current'] for p in new_cache.values())
            total_predicted = sum(p['predicted'] for p in new_cache.values())
            
            new_cache['overall'] = {
                'total_current': total_current,
                'total_predicted': total_predicted,
                'total_change': total_predicted - total_current,
                'num_locations': len(new_cache),
                'last_updated': datetime.now().isoformat()
            }
        
        predictions_cache = new_cache
        last_update_time = datetime.now()
        
        # Save cache to file
        try:
            with open(CACHE_FILE, 'wb') as f:
                pickle.dump(predictions_cache, f)
            logger.info(f"Cache saved to {CACHE_FILE}")
        except Exception as e:
            logger.error(f"Error saving cache: {e}")
    
    logger.info(f"Cache updated successfully at {last_update_time}")

# ============================================
# API ENDPOINTS
# ============================================

@app.route('/api/predictions', methods=['GET'])
def get_predictions():
    """Get all predictions"""
    with cache_lock:
        if not predictions_cache:
            return jsonify({'error': 'No predictions available'}), 503
        
        # Convert to format expected by frontend
        response = {
            'libraries': {k: v for k, v in predictions_cache.items() if k != 'overall'},
            'overall': predictions_cache.get('overall', {}),
            'last_updated': last_update_time.isoformat() if last_update_time else None
        }
        
        return jsonify(response)

@app.route('/api/predictions/<library>', methods=['GET'])
def get_library_prediction(library):
    """Get prediction for specific library"""
    with cache_lock:
        if library not in predictions_cache:
            return jsonify({'error': f'Library {library} not found'}), 404
        
        return jsonify(predictions_cache[library])

@app.route('/api/refresh', methods=['POST'])
def force_refresh():
    """Force refresh of predictions"""
    update_predictions_cache()
    return jsonify({'status': 'success', 'message': 'Predictions updated'})

@app.route('/api/status', methods=['GET'])
def get_status():
    """Get API status"""
    return jsonify({
        'status': 'online',
        'last_update': last_update_time.isoformat() if last_update_time else None,
        'num_libraries': len([k for k in predictions_cache.keys() if k != 'overall']),
        'auto_update_enabled': scheduler.running if scheduler else False
    })

@app.route('/api/libraries', methods=['GET'])
def get_libraries():
    """Get list of available libraries"""
    with cache_lock:
        libraries = [k for k in predictions_cache.keys() if k != 'overall']
        return jsonify({'libraries': libraries})

@app.route('/api/models/metrics', methods=['GET'])
def get_model_metrics():
    """Get metrics for all trained models"""
    try:
        # Load metrics from training results
        metrics_file = 'training_results/all_models_results.json'
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                metrics_data = json.load(f)
            return jsonify({'models': metrics_data})
        else:
            return jsonify({'models': {}, 'message': 'No training metrics available. Please train models first.'})
    except Exception as e:
        logger.error(f"Error loading model metrics: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/models/retrain', methods=['POST'])
def retrain_models():
    """Trigger retraining of all models"""
    try:
        import subprocess
        # Run training script in background
        subprocess.Popen(['python', 'train_all_libraries.py'])
        return jsonify({'status': 'success', 'message': 'Model training started in background'})
    except Exception as e:
        logger.error(f"Error starting training: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/', methods=['GET'])
def home():
    """Home endpoint"""
    return jsonify({
        'message': 'Library Occupancy Prediction API',
        'version': '1.0',
        'endpoints': {
            'GET /api/predictions': 'Get all predictions',
            'GET /api/predictions/<library>': 'Get prediction for specific library',
            'GET /api/libraries': 'Get list of libraries',
            'GET /api/models/metrics': 'Get training metrics for all models',
            'POST /api/models/retrain': 'Retrain all models',
            'GET /api/status': 'Get API status',
            'POST /api/refresh': 'Force refresh predictions'
        }
    })

# ============================================
# SCHEDULER SETUP
# ============================================

def start_scheduler():
    """Start background scheduler for automatic updates"""
    global scheduler
    
    scheduler = BackgroundScheduler()
    scheduler.add_job(
        func=update_predictions_cache,
        trigger="interval",
        seconds=UPDATE_INTERVAL,
        id='update_predictions',
        name='Update predictions cache',
        replace_existing=True
    )
    scheduler.start()
    logger.info(f"Scheduler started - updating every {UPDATE_INTERVAL} seconds")

# ============================================
# MAIN
# ============================================

if __name__ == '__main__':
    # Load trained model and scaler
    logger.info("Loading trained model...")
    has_model = load_trained_model()
    if has_model:
        logger.info("✓ Using trained CNN-LSTM model for predictions")
    else:
        logger.info("→ Using simple statistical predictions (no trained model)")

    # Load cache from file if exists
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'rb') as f:
                predictions_cache = pickle.load(f)
            logger.info("Loaded predictions from cache file")
        except Exception as e:
            logger.error(f"Error loading cache: {e}")

    # Initial update
    logger.info("Performing initial prediction update...")
    update_predictions_cache()

    # Start scheduler
    start_scheduler()
    
    # Run Flask app
    logger.info("Starting Flask API server...")
    app.run(host='0.0.0.0', port=5000, debug=False)
