# Real-Time Library Occupancy Prediction System

A machine learning-powered system for predicting library occupancy in real-time using WiFi access point data. The system compares multiple deep learning architectures (LSTM, CNN, Hybrid, and Advanced CNN-LSTM) across multiple library locations.

## Features

- **Multiple Model Architectures**: Compare 4 different neural network models:
  - LSTM Only
  - CNN Only
  - Hybrid CNN-LSTM
  - Advanced CNN-LSTM

- **Multi-Library Support**: Tracks 6 library locations:
  - Miguel Pro Library
  - American Corner
  - Gisbert 2nd Floor
  - Gisbert 3rd Floor
  - Gisbert 4th Floor
  - Gisbert 5th Floor

- **Real-Time Predictions**: Hourly occupancy predictions with 6-hour forecasts
- **Interactive Dashboard**: React-based admin dashboard with model comparison
- **Performance Metrics**: R², RMSE, MAE, and MAPE for each model/library combination
- **Auto-Refresh**: Automatic updates every 60 seconds

## System Architecture

```
├── Backend (Flask API)
│   ├── Model inference (24 trained models)
│   ├── Data processing and predictions
│   └── REST API endpoints
│
├── Frontend (React Dashboard)
│   ├── Model type selector
│   ├── Performance visualization
│   └── Real-time predictions display
│
└── Training Pipeline
    ├── AP MAC to library mapping
    ├── Data preprocessing
    └── Model training for all architectures
```

## Prerequisites

- **Python**: 3.8 or higher
- **Node.js**: 14.x or higher
- **npm**: 6.x or higher
- **Git**: For cloning the repository

## First-Time Setup

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/real-time-prediction.git
cd real-time-prediction
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

Required packages include:
- tensorflow
- scikit-learn
- pandas
- numpy
- flask
- flask-cors
- apscheduler
- matplotlib

### 3. Prepare Your Data

Place your WiFi data CSV file in the project root:
- File name: `all_data_cleaned.csv`
- Required columns:
  - `Start_dt`: Timestamp
  - `Client MAC`: Unique device identifier
  - `AP MAC`: Access point MAC address

### 4. Configure AP-to-Library Mapping

Edit `ap_location_mapping.py` to map your access point MAC addresses to library locations:

```python
AP_LOCATION_MAP = {
    'miguel_pro': [
        '10:F0:68:29:66:70',  # Your AP MAC addresses
        '10:F0:68:28:3C:D0',
        # Add more...
    ],
    'american_corner': [
        '34:15:93:01:25:40',
        # Add more...
    ],
    # Configure other libraries...
}
```

### 5. Train the Models

Train all 24 models (4 architectures × 6 libraries):

```bash
python train_multiple_model_types.py
```

This will:
- Process the WiFi data
- Map APs to library locations
- Train 4 different model architectures for each library
- Save models to `saved_models/`
- Save scalers to `saved_scalers/`
- Generate metrics in `model_results/all_model_types_results.json`

Training time: ~1 hour (depending on hardware)

### 6. Install Frontend Dependencies

```bash
cd library-dashboard
npm install
cd ..
```

## Running the System

### Start the Backend API

```bash
python api_backend.py
```

The backend will:
- Load all 24 trained models
- Start Flask server on `http://localhost:5000`
- Begin generating predictions every 60 seconds

Output should show:
```
✓ Loaded 24 models successfully
✓ Loaded 24 models for predictions
Cache updated successfully
Starting Flask API server...
Running on http://127.0.0.1:5000
```

### Start the Frontend Dashboard

In a new terminal:

```bash
cd library-dashboard
npm start
```

The dashboard will open automatically at `http://localhost:3000`

## Using the Dashboard

### Admin Dashboard (Default View)

1. **Select Model Type**: Choose which model architecture to view
   - LSTM Only
   - CNN Only
   - Hybrid CNN-LSTM
   - Advanced CNN-LSTM

2. **Select View Mode**:
   - **Overview**: Current predictions for all libraries
   - **Performance Metrics**: Detailed metrics for each library
   - **Model Comparison**: Compare all model types side-by-side

3. **Features**:
   - Real-time occupancy display
   - 6-hour prediction forecasts
   - Hourly trend charts
   - Performance metrics tables
   - Auto-refresh toggle

### Switching to User View

Click the "Switch to User View" button for a simplified dashboard showing current occupancy only.

## API Endpoints

The backend provides the following REST API endpoints:

### Get Model Types
```
GET /api/model-types
```
Returns list of available model architectures.

### Get Predictions
```
GET /api/predictions?model_type=<type>
```
Returns predictions for all libraries using specified model type.

Parameters:
- `model_type`: lstm_only, cnn_only, hybrid_cnn_lstm, advanced_cnn_lstm

### Get Model Metrics
```
GET /api/models/metrics
```
Returns training performance metrics for all model types.

### Get Libraries
```
GET /api/libraries
```
Returns list of available library locations.

### Force Refresh
```
POST /api/refresh
```
Manually trigger prediction cache update.

### Retrain Models
```
POST /api/models/retrain
```
Start background model retraining.

## Project Structure

```
real-time-prediction/
├── api_backend.py              # Flask API server
├── train_multiple_model_types.py  # Model training script
├── ap_location_mapping.py      # AP-to-library mapping
├── requirements.txt            # Python dependencies
├── all_data_cleaned.csv        # WiFi data (not in git)
│
├── saved_models/               # Trained model files (.keras)
├── saved_scalers/             # Data scalers (.pkl)
├── model_results/             # Training metrics (JSON)
│
└── library-dashboard/         # React frontend
    ├── package.json
    ├── public/
    └── src/
        ├── App.js             # Main app with view switcher
        ├── AdminDashboard.js  # Model comparison dashboard
        └── SimpleDashboard.js # User-facing dashboard
```

## Model Performance

The system trains and compares 4 different neural network architectures:

### 1. LSTM Only
- Pure LSTM layers for time series prediction
- Best for capturing long-term dependencies

### 2. CNN Only
- Convolutional layers with max pooling
- **Best overall performer** (R² = 0.9679 for Miguel Pro)
- Efficient at pattern recognition

### 3. Hybrid CNN-LSTM
- Combines CNN feature extraction with LSTM temporal modeling
- Good balance of spatial and temporal features

### 4. Advanced CNN-LSTM
- Multi-layer architecture with attention mechanisms
- Higher complexity, good generalization

## Training New Models

To retrain models with new data:

1. Update `all_data_cleaned.csv` with new WiFi records
2. Run training:
   ```bash
   python train_multiple_model_types.py
   ```
3. Restart the backend to load new models:
   ```bash
   python api_backend.py
   ```

Or use the API:
```bash
curl -X POST http://localhost:5000/api/models/retrain
```

## Troubleshooting

### Backend won't start
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check if port 5000 is available
- Verify models exist in `saved_models/` directory

### Frontend shows errors
- Check backend is running on port 5000
- Clear browser cache
- Reinstall dependencies: `cd library-dashboard && npm install`

### Models not loading
- Run training script first: `python train_multiple_model_types.py`
- Check `saved_models/` and `saved_scalers/` directories exist
- Verify 24 .keras files and 24 .pkl files are present

### Predictions show 0 or incorrect values
- Check AP MAC mapping in `ap_location_mapping.py`
- Verify data file has correct columns
- Check backend logs for errors

## Data Requirements

Your `all_data_cleaned.csv` should contain:

| Column | Type | Description |
|--------|------|-------------|
| Start_dt | datetime | Connection timestamp |
| Client MAC | string | Unique device identifier |
| AP MAC | string | Access point MAC address |

Example:
```csv
Start_dt,Client MAC,AP MAC
2025-07-20 22:18:57,AA:BB:CC:DD:EE:FF,10:F0:68:29:66:70
2025-07-20 22:19:15,11:22:33:44:55:66,34:15:93:01:25:40
```

## Performance Metrics Explained

- **R² (Coefficient of Determination)**: 0-1 scale, higher is better. Measures prediction accuracy.
- **RMSE (Root Mean Squared Error)**: Lower is better. Average prediction error in users.
- **MAE (Mean Absolute Error)**: Lower is better. Average absolute difference from actual.
- **MAPE (Mean Absolute Percentage Error)**: Lower is better. Percentage error.

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -m "Add feature"`
4. Push to branch: `git push origin feature-name`
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Acknowledgments

- Built with TensorFlow/Keras for deep learning models
- React and Recharts for visualization
- Flask for backend API
- APScheduler for automatic updates

## Support

For issues or questions:
1. Check the Troubleshooting section
2. Review API logs in terminal
3. Open an issue on GitHub

---

**Generated with Claude Code** 🤖
