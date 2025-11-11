"""
Flask API Backend for Library Occupancy Predictions
Serves real-time predictions to React dashboard
Includes automatic updates and data management
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

CLEANED_DATA_FILE = 'all_data_cleaned.csv'
CACHE_FILE = 'predictions_cache.pkl'
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
# MODEL LOADING
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
# PREDICTION FUNCTIONS
# ============================================

def load_historical_data():
    """Load and process historical data with location mapping"""
    try:
        df = pd.read_csv(CLEANED_DATA_FILE)
        logger.info(f"Loaded {len(df)} rows from {CLEANED_DATA_FILE}")

        df['Start_dt'] = pd.to_datetime(df['Start_dt'])
        df.set_index('Start_dt', inplace=True)

        # Add location mapping
        from ap_location_mapping import get_location_from_ap
        df['Location'] = df['AP MAC'].apply(get_location_from_ap)

        logger.info(f"Date range: {df.index.min()} to {df.index.max()}")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return None

def get_library_occupancy(df, library_id, hours=None):
    """Get occupancy time series for a specific library

    Args:
        df: DataFrame with occupancy data
        library_id: Library identifier
        hours: Number of hours to return. If None, returns all available data.
    """
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

    # Return all data for pattern-based predictions (need full history)
    return occupancy if hours is None else occupancy.tail(hours)

def predict_for_current_time(occupancy_series, library_id, hours_ahead=6):
    """Predict for the CURRENT real-world time using historical patterns"""
    if len(occupancy_series) == 0:
        return None

    # Get current real-world time
    now = datetime.now()
    current_hour = now.hour
    current_day = now.weekday()  # Monday=0, Sunday=6

    logger.info(f"Predicting for {library_id} - Current time: {now.strftime('%A %I:%M %p')} (day={current_day}, hour={current_hour})")

    # Convert occupancy series to dataframe for easier manipulation
    df = pd.DataFrame({'occupancy': occupancy_series})
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek

    predictions = []

    for i in range(hours_ahead):
        target_hour = (current_hour + i) % 24
        target_day = (current_day + (current_hour + i) // 24) % 7

        # Get historical data for this day/hour combination
        matching = df[(df['hour'] == target_hour) & (df['day_of_week'] == target_day)]

        if len(matching) > 0:
            # Use recent history (last 8 weeks) if available
            recent_matching = matching.tail(8)  # Last 8 occurrences

            # Filter out extremely low values (likely closed days or errors)
            # Only if we have enough data points
            values = recent_matching['occupancy'].values
            if len(values) >= 4:
                # Remove values that are less than 10% of the median (likely closed)
                median_val = np.median(values)
                if median_val > 10:  # Only filter if median is reasonably high
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
            logger.info(f"  Hour +{i} ({target_hour}:00): {predicted} users (based on {len(recent_matching)} historical samples, values={values})")
        else:
            # Fallback to overall hour average
            hour_avg = df[df['hour'] == target_hour]['occupancy'].mean()
            predicted = max(0, int(hour_avg))
            predictions.append(predicted)
            logger.info(f"  Hour +{i} ({target_hour}:00): {predicted} users (fallback to hour average)")

    return predictions

def predict_with_specific_model(model_type, library_id, occupancy_series, hours_ahead=6):
    """Predict using a specific model type and library"""
    global models_cache, scalers_cache

    # IMPORTANT: Use pattern-based prediction for current real-world time
    # This gives accurate predictions based on day/hour patterns
    return predict_for_current_time(occupancy_series, library_id, hours_ahead)

    # Original model-based prediction (commented out but kept for reference)
    # model_key = f"{model_type}_{library_id}"
    # if model_key not in models_cache or model_key not in scalers_cache:
    #     logger.warning(f"Model {model_key} not loaded, using simple prediction")
    #     return predict_simple(occupancy_series, hours_ahead)
    # ... rest of model code ...

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

def generate_predictions_for_model_type(model_type, df=None):
    """Generate predictions for all libraries using a specific model type"""
    if df is None:
        df = load_historical_data()
        if df is None:
            return None

    results = {}

    # Get current real-world time for pattern-based "current" occupancy
    now = datetime.now()
    current_hour = now.hour
    current_day = now.weekday()

    for library_id, library_name in LIBRARY_IDS.items():
        try:
            # Get occupancy data for this library
            occupancy = get_library_occupancy(df, library_id)

            if occupancy is None or len(occupancy) < 24:
                logger.warning(f"Insufficient data for {library_id}")
                continue

            # Current stats - use pattern-based current for real-world time
            # Convert to dataframe for day/hour filtering
            occ_df = pd.DataFrame({'occupancy': occupancy})
            occ_df['hour'] = occ_df.index.hour
            occ_df['day_of_week'] = occ_df.index.dayofweek

            # Get historical average for current day/hour
            matching_current = occ_df[(occ_df['hour'] == current_hour) & (occ_df['day_of_week'] == current_day)]
            if len(matching_current) > 0:
                recent_values = matching_current.tail(8)['occupancy'].values
                # Filter out extremely low values (closed days)
                if len(recent_values) >= 4:
                    median_val = np.median(recent_values)
                    if median_val > 10:
                        filtered = recent_values[recent_values > median_val * 0.1]
                        if len(filtered) >= 3:
                            current = int(filtered.mean())
                        else:
                            current = int(recent_values.mean())
                    else:
                        current = int(recent_values.mean())
                else:
                    current = int(recent_values.mean())
            else:
                current = int(occupancy.iloc[-1])  # Fallback to last data point

            recent_24h = occupancy.tail(24).values
            avg_24h = int(recent_24h.mean())
            max_24h = int(recent_24h.max())

            # Predictions using specific model type
            next_hour_predictions = predict_with_specific_model(
                model_type, library_id, occupancy, hours_ahead=6
            )

            if next_hour_predictions is None:
                continue

            predicted_next = next_hour_predictions[0]
            change = predicted_next - current

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
                next_hours.append({
                    'time': future_time.strftime('%I %p'),
                    'predicted': pred,
                    'confidence_upper': int(pred * 1.15),
                    'confidence_lower': max(0, int(pred * 0.85))
                })

            results[library_id] = {
                'library_id': library_id,
                'library_name': library_name,
                'current': current,
                'predicted': predicted_next,
                'change': change,
                'avg_24h': avg_24h,
                'max_24h': max_24h,
                'hourly_data': hourly_data,
                'next_hours': next_hours
            }

        except Exception as e:
            logger.error(f"Error generating predictions for {library_id} with {model_type}: {e}")
            continue

    return results

def update_predictions_cache():
    """Update predictions cache for all model types"""
    global predictions_cache, last_update_time

    logger.info("Updating predictions cache for all model types...")

    df = load_historical_data()
    if df is None:
        logger.error("Failed to load data")
        return

    with cache_lock:
        new_cache = {}

        # Generate predictions for each model type
        for model_type in MODEL_TYPES.keys():
            try:
                predictions = generate_predictions_for_model_type(model_type, df)
                if predictions:
                    new_cache[model_type] = predictions
                    logger.info(f"Updated predictions for {model_type}: {len(predictions)} libraries")
            except Exception as e:
                logger.error(f"Error generating predictions for {model_type}: {e}")

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

@app.route('/api/model-types', methods=['GET'])
def get_model_types():
    """Get list of available model types"""
    return jsonify({
        'model_types': [
            {'id': k, 'name': v} for k, v in MODEL_TYPES.items()
        ]
    })

@app.route('/api/predictions', methods=['GET'])
def get_predictions():
    """Get predictions for a specific model type (default: cnn_only)"""
    model_type = request.args.get('model_type', 'cnn_only')

    with cache_lock:
        if not predictions_cache:
            return jsonify({'error': 'No predictions available'}), 503

        if model_type not in predictions_cache:
            return jsonify({'error': f'Model type {model_type} not found'}), 404

        return jsonify({
            'model_type': model_type,
            'model_name': MODEL_TYPES.get(model_type, model_type),
            'libraries': predictions_cache[model_type],
            'last_updated': last_update_time.isoformat() if last_update_time else None
        })

@app.route('/api/predictions/<model_type>/<library_id>', methods=['GET'])
def get_specific_prediction(model_type, library_id):
    """Get prediction for specific model type and library"""
    with cache_lock:
        if model_type not in predictions_cache:
            return jsonify({'error': f'Model type {model_type} not found'}), 404

        if library_id not in predictions_cache[model_type]:
            return jsonify({'error': f'Library {library_id} not found'}), 404

        return jsonify(predictions_cache[model_type][library_id])

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
        'num_model_types': len(MODEL_TYPES),
        'num_libraries': len(LIBRARY_IDS),
        'auto_update_enabled': scheduler.running if scheduler else False
    })

@app.route('/api/libraries', methods=['GET'])
def get_libraries():
    """Get list of available libraries"""
    return jsonify({
        'libraries': [
            {'id': k, 'name': v} for k, v in LIBRARY_IDS.items()
        ]
    })

@app.route('/api/models/metrics', methods=['GET'])
def get_model_metrics():
    """Get metrics for all trained model types"""
    try:
        # Load metrics from model_results (new format)
        metrics_file = 'model_results/all_model_types_results.json'
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                metrics_data = json.load(f)
            return jsonify({
                'model_types': metrics_data,
                'last_updated': last_update_time.isoformat() if last_update_time else None
            })
        else:
            return jsonify({'model_types': {}, 'message': 'No training metrics available. Please train models first.'})
    except Exception as e:
        logger.error(f"Error loading model metrics: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/models/retrain', methods=['POST'])
def retrain_models():
    """Trigger retraining of all model types"""
    try:
        import subprocess
        # Run training script in background
        subprocess.Popen(['python', 'train_multiple_model_types.py'])
        return jsonify({'status': 'success', 'message': 'Model training started in background for all model types'})
    except Exception as e:
        logger.error(f"Error starting training: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/shap/results', methods=['GET'])
def get_shap_results():
    """Get SHAP analysis results for all models"""
    try:
        results_path = 'model_results/all_model_types_results.json'
        if os.path.exists(results_path):
            with open(results_path, 'r') as f:
                results = json.load(f)
            return jsonify(results)
        else:
            return jsonify({'error': 'SHAP results not available. Please train models first.'}), 404
    except Exception as e:
        logger.error(f"Error loading SHAP results: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/shap/images/<filename>', methods=['GET'])
def get_shap_image(filename):
    """Serve SHAP visualization images"""
    try:
        image_path = os.path.join('model_results', 'shap', filename)
        if os.path.exists(image_path):
            return send_file(image_path, mimetype='image/png')
        else:
            # Return a placeholder image or 404
            logger.warning(f"SHAP image not found: {filename}")
            return jsonify({'error': 'Image not found'}), 404
    except Exception as e:
        logger.error(f"Error serving SHAP image {filename}: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/shap/report', methods=['GET'])
def get_shap_report():
    """Get SHAP analysis text report"""
    try:
        report_path = 'model_results/shap/SHAP_ANALYSIS_REPORT.txt'
        if os.path.exists(report_path):
            with open(report_path, 'r') as f:
                report = f.read()
            return jsonify({'report': report})
        else:
            return jsonify({'error': 'SHAP report not available'}), 404
    except Exception as e:
        logger.error(f"Error loading SHAP report: {e}")
        return jsonify({'error': str(e)}), 500

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/admin/upload-csv', methods=['POST'])
def upload_csv():
    """Upload a new CSV file"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        return jsonify({
            'message': 'File uploaded successfully',
            'filename': filename
        })
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/api/admin/clean-data', methods=['POST'])
def clean_data():
    """Clean and process uploaded data"""
    try:
        data = request.get_json()
        filename = data.get('filename')
        if not filename:
            return jsonify({'error': 'Filename not provided'}), 400

        file_path = os.path.join(UPLOAD_FOLDER, filename)
        
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Basic validation
        required_columns = {'AP MAC', 'Client MAC', 'Start_dt'}
        missing_cols = required_columns - set(df.columns)
        if missing_cols:
            return jsonify({
                'error': f'Missing required columns: {", ".join(missing_cols)}'
            }), 400

        # Clean the data
        df['Start_dt'] = pd.to_datetime(df['Start_dt'])
        df = df.sort_values('Start_dt')
        
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Save cleaned data
        cleaned_path = 'all_data_cleaned.csv'
        df.to_csv(cleaned_path, index=False)
        
        # Clean up the upload
        os.remove(file_path)
        
        # Update predictions cache
        update_predictions_cache()
        
        return jsonify({
            'message': 'Data cleaned successfully',
            'rows_processed': len(df)
        })

    except Exception as e:
        logger.error(f"Error cleaning data: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/', methods=['GET'])
def home():
    """Home endpoint"""
    return jsonify({
        'message': 'Library Occupancy Prediction API - Model Comparison with SHAP Analysis',
        'version': '2.1',
        'endpoints': {
            'GET /api/model-types': 'Get list of available model types',
            'GET /api/predictions?model_type=<type>': 'Get predictions for all libraries using specific model type',
            'GET /api/predictions/<model_type>/<library_id>': 'Get prediction for specific model and library',
            'GET /api/libraries': 'Get list of libraries',
            'GET /api/models/metrics': 'Get training metrics for all model types',
            'POST /api/models/retrain': 'Retrain all model types',
            'GET /api/status': 'Get API status',
            'POST /api/refresh': 'Force refresh predictions',
            'GET /api/shap/results': 'Get SHAP analysis results',
            'GET /api/shap/images/<filename>': 'Get SHAP visualization image',
            'GET /api/shap/report': 'Get SHAP analysis text report',
            'POST /api/admin/upload-csv': 'Upload new CSV data file',
            'POST /api/admin/clean-data': 'Clean and process uploaded data'
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
    # Load all trained models
    logger.info("Loading all trained models...")
    has_models = load_all_models()
    if has_models:
        logger.info(f"✓ Loaded {len(models_cache)} models for predictions")
    else:
        logger.warning("→ No trained models loaded, predictions may use simple statistical methods")

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
