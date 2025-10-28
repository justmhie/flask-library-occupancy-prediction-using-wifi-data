# 🎛️ Library Occupancy Admin Dashboard

## Overview

This is a comprehensive admin dashboard for monitoring and managing library occupancy predictions across multiple library locations using CNN-LSTM deep learning models.

## Features

### 📊 **Multi-Library Support**
- Train separate models for each library location
- Track performance metrics for all models
- Compare model accuracy across libraries

### 🔮 **Real-Time Predictions**
- Live occupancy data for all libraries
- Next 6-hour forecasts with confidence intervals
- Hourly and daily usage patterns

### 📈 **Interactive Visualizations**
- Overview mode: Current stats and trends
- Predictions mode: Detailed forecasts
- Comparison mode: Side-by-side model performance
- Metrics mode: Comprehensive training statistics

### 🤖 **ML Model Management**
- CNN-LSTM architecture for each library
- Automated training pipeline
- Model performance tracking
- API-triggered retraining

## Quick Start

### 1. **First Time Setup**

Update the AP MAC to Library mapping in `ap_location_mapping.py`:

```python
AP_LOCATION_MAP = {
    'miguel_pro': [
        '80:BC:37:20:8A:20',  # Add your actual AP MAC addresses
        '5C:DF:89:07:62:A0',
    ],
    'gisbert_2nd': [
        '5C:DF:89:07:69:30',
        # Add more...
    ],
    # ... other libraries
}
```

### 2. **Train All Models**

```bash
# Train models for all libraries (this will take 10-30 minutes)
python train_all_libraries.py
```

This will:
- ✅ Train CNN-LSTM models for each library
- ✅ Save trained models and scalers
- ✅ Generate performance metrics
- ✅ Create visualization plots
- ✅ Save results to `training_results/`

### 3. **Start the Backend**

```bash
# Start Flask API server
python api_backend.py
```

The backend will:
- Load all trained models
- Serve predictions via REST API
- Update every 60 seconds
- Run on http://localhost:5000

### 4. **Start the Dashboard**

```bash
# Navigate to dashboard directory
cd library-dashboard

# Install dependencies (first time only)
npm install

# Start development server
npm start
```

Dashboard opens at: **http://localhost:3000**

## Dashboard Modes

### 📊 Overview Mode
- Current occupancy for selected library
- Predicted next hour
- 24-hour average
- Peak usage times
- Hourly trends chart
- Daily usage patterns

### 🔮 Predictions Mode
- Next 6-hour forecast with confidence intervals
- Detailed predictions table
- Upper and lower prediction bounds
- Visual forecast chart

### 📈 Model Comparison Mode
- R² scores for all libraries
- RMSE comparison
- Average occupancy by library
- Comprehensive metrics table
- Performance rankings

### 📉 Detailed Metrics Mode
- R², RMSE, MAE, MAPE for selected library
- Dataset statistics
- Training information
- Model file locations
- Training date and epochs

## File Structure

```
real-time-prediction/
├── ap_location_mapping.py          # AP MAC to library mapping
├── train_all_libraries.py          # Train all library models
├── train_model.py                  # Train single overall model
├── api_backend.py                  # Flask API server
├── test_backend.py                 # Backend validation
│
├── saved_models/                   # Trained model files
│   ├── all_model.keras            # Overall model
│   ├── miguel_pro_model.keras     # Miguel Pro model
│   └── ...
│
├── saved_scalers/                  # Data scalers
│   ├── all_scaler.pkl
│   └── ...
│
├── training_results/               # Training outputs
│   ├── all_models_results.json    # Metrics for all models
│   ├── models_comparison.png      # Comparison chart
│   └── plots/                      # Individual result plots
│       ├── all_results.png
│       ├── miguel_pro_results.png
│       └── ...
│
└── library-dashboard/              # React dashboard
    └── src/
        ├── AdminDashboard.js      # Admin dashboard component
        ├── SimpleDashboard.js     # User-facing dashboard
        └── App.js                  # Main app with switcher
```

## API Endpoints

### Get Predictions
```bash
GET http://localhost:5000/api/predictions
```
Returns predictions for all libraries

### Get Model Metrics
```bash
GET http://localhost:5000/api/models/metrics
```
Returns training metrics for all models

### Trigger Retraining
```bash
POST http://localhost:5000/api/models/retrain
```
Starts background training of all models

### Get Specific Library
```bash
GET http://localhost:5000/api/predictions/{library_id}
```
Returns predictions for one library

## Model Performance Metrics

### R² Score (Coefficient of Determination)
- Range: 0 to 1
- Higher is better
- Measures how well predictions match actual data
- **Good:** > 0.8
- **Excellent:** > 0.9

### RMSE (Root Mean Squared Error)
- Units: number of users
- Lower is better
- Measures average prediction error
- **Good:** < 20 users
- **Excellent:** < 10 users

### MAE (Mean Absolute Error)
- Units: number of users
- Lower is better
- Average absolute deviation from actual
- **Good:** < 15 users
- **Excellent:** < 5 users

### MAPE (Mean Absolute Percentage Error)
- Units: percentage
- Lower is better
- Percentage deviation from actual
- **Good:** < 10%
- **Excellent:** < 5%

## Configuration

### Training Parameters (train_all_libraries.py)
```python
SEQUENCE_LENGTH = 24        # Hours of history to use
EPOCHS = 50                 # Training iterations
BATCH_SIZE = 32            # Training batch size
TEST_SIZE = 0.2            # 20% for testing
MIN_HOURS_REQUIRED = 100   # Minimum data needed
```

### Backend Settings (api_backend.py)
```python
UPDATE_INTERVAL = 60        # Prediction update frequency (seconds)
SEQUENCE_LENGTH = 24        # Must match training
```

## Switching Between User and Admin Views

Click the button in the top-right corner:
- **🎛️ Switch to Admin View** - Full metrics and comparisons
- **👤 Switch to User View** - Simple occupancy display

## Troubleshooting

### "No metrics available"
- Run `python train_all_libraries.py` first
- Check that `training_results/all_models_results.json` exists

### "Using simple statistical predictions"
- Models not trained yet, or
- Model files missing in `saved_models/`
- Run training script to generate models

### API returning 503 errors
- Backend not running
- Start with `python api_backend.py`

### Dashboard shows "Demo data"
- Backend API not accessible
- Check backend is running on port 5000
- Check firewall settings

### Low model accuracy
- Need more data (minimum 100 hours per library)
- Adjust AP MAC mappings
- Increase training epochs
- Check data quality

## Advanced Usage

### Retrain Specific Library
1. Modify `train_all_libraries.py`
2. Set `libraries_to_train = ['miguel_pro']`
3. Run training script

### Add New Library
1. Update `ap_location_mapping.py`:
   ```python
   'new_library': ['AP:MAC:ADDRESS'],
   ```
2. Add to `LIBRARY_NAMES` dict
3. Retrain all models
4. Restart backend

### Export Metrics
Metrics are saved as JSON in:
```
training_results/all_models_results.json
```

### Schedule Auto-Retraining
Add to cron (Linux) or Task Scheduler (Windows):
```bash
# Retrain every week
0 0 * * 0 python train_all_libraries.py
```

## System Requirements

- **Python:** 3.8+
- **Node.js:** 14+
- **RAM:** 8GB+ (for model training)
- **Disk:** 1GB+ free space
- **Data:** Minimum 100 hours per library

## Performance Tips

- **Faster Training:** Reduce `EPOCHS` to 30
- **Better Accuracy:** Increase `EPOCHS` to 100
- **Less Memory:** Reduce `BATCH_SIZE` to 16
- **Faster API:** Reduce `UPDATE_INTERVAL` to 30 seconds

## Support

For issues or questions:
1. Check logs in backend console
2. Verify data file exists
3. Check browser console for frontend errors
4. Review API responses at http://localhost:5000/api/predictions

---

**Built with:** React, Recharts, Flask, TensorFlow/Keras, Scikit-learn
