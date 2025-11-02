# SHAP Analysis Web Dashboard Setup Guide

## Overview

This guide shows you how to view SHAP analysis visualizations in your web browser through the React dashboard.

## What You'll See

The web dashboard provides:
- **Interactive SHAP visualizations** for all models and libraries
- **4 visualization types**: Summary, Waterfall, Force, and Dependence plots
- **Cross-model comparisons**: Heatmaps and trends across all models
- **Feature importance tables**: Ranked list of most important time steps
- **Real-time switching**: Change models, libraries, and views instantly

## Step-by-Step Setup

### 1. Generate SHAP Analysis (if not done already)

First, train your models with SHAP analysis:

```bash
# Navigate to project directory
cd c:\Users\User\Documents\codes\real-time-prediction

# Train models with SHAP analysis (this will take time)
python train_multiple_model_types.py
```

This creates:
- `model_results/shap/*.png` - Individual SHAP plots for each model
- `model_results/all_model_types_results.json` - Results with SHAP data

### 2. Start the Backend API

The backend serves SHAP images and data:

```bash
# Start Flask API server
python api_backend.py
```

You should see:
```
✓ Loaded X models for predictions
Starting Flask API server...
* Running on http://0.0.0.0:5000
```

**Leave this terminal running!**

### 3. Start the React Dashboard

Open a **new terminal** and start the React app:

```bash
# Navigate to dashboard directory
cd library-dashboard

# Install dependencies (first time only)
npm install

# Start development server
npm start
```

The browser should automatically open to `http://localhost:3000`

### 4. View SHAP Analysis

In the web dashboard:

1. **Click "🔍 SHAP Analysis"** in the navigation
2. **Select model type**: LSTM Only, CNN Only, Hybrid, or Advanced
3. **Select library**: Miguel Pro, American Corner, Gisbert floors
4. **Choose visualization**: Summary, Waterfall, Force, or Dependence
5. **Switch tabs**: Individual Model Analysis or Cross-Model Comparison

## Dashboard Features

### Individual Model Analysis Tab

#### Summary View (📊 Feature Importance)
- **Bar chart**: Shows which time steps are most important
- **Dot plot**: Shows how feature values affect predictions
- **Table**: Top 10 features with importance scores

**What to look for:**
- Recent hours (t-0 to t-5) typically have highest importance
- Red dots = high occupancy in past → higher predictions

#### Waterfall View (💧 Prediction Breakdown)
- Shows how a single prediction is built step-by-step
- Red bars = time steps that increase prediction
- Blue bars = time steps that decrease prediction

**What to look for:**
- Base value (average prediction)
- Which specific hours pushed prediction up or down
- Final prediction value

#### Force Plot (⚡)
- Alternative view of single prediction
- Shows forces pushing prediction from base to final value

#### Dependence Plot (🔗 Feature Interactions)
- Shows relationship between feature value and impact
- X-axis = past occupancy at specific time
- Y-axis = impact on prediction (SHAP value)

**What to look for:**
- Positive slope = higher past → higher prediction
- Scatter indicates interactions with other features

### Cross-Model Comparison Tab

Compare SHAP analysis across all models:

1. **Feature Importance Heatmap**
   - All models side-by-side
   - Which features each model prioritizes

2. **Average Feature Importance**
   - Trends across all libraries
   - Model consensus on important features

3. **Library-wise Comparison**
   - How feature importance differs by location
   - Library-specific patterns

4. **Summary Statistics**
   - Top features across all models
   - Model consistency analysis

## Troubleshooting

### Problem: "SHAP plot not available" message

**Solution:**
1. Check that training with SHAP completed successfully
2. Look for files in `model_results/shap/`
3. Run comparison script: `python visualize_shap_comparison.py`

### Problem: Can't connect to backend

**Solution:**
1. Ensure `api_backend.py` is running
2. Check it's on port 5000: `http://localhost:5000`
3. Visit `http://localhost:5000` to see API info

### Problem: CORS errors in browser console

**Solution:**
Backend already has CORS enabled. If issues persist:
```python
# In api_backend.py, CORS is already configured:
from flask_cors import CORS
app = Flask(__name__)
CORS(app)  # Already present
```

### Problem: Images not loading

**Solution:**
1. Check browser console for errors
2. Verify image paths: `http://localhost:5000/api/shap/images/cnn_only_miguel_pro_summary.png`
3. Ensure files exist in `model_results/shap/`

## API Endpoints

The backend provides these SHAP endpoints:

```
GET /api/shap/results
  → Returns all SHAP analysis results (JSON)

GET /api/shap/images/<filename>
  → Serves SHAP visualization images
  → Example: /api/shap/images/cnn_only_miguel_pro_summary.png

GET /api/shap/report
  → Returns text report with insights
```

### Testing Endpoints

Open in browser or use curl:

```bash
# Check if API is running
curl http://localhost:5000

# Get SHAP results
curl http://localhost:5000/api/shap/results

# View an image
# Open in browser:
http://localhost:5000/api/shap/images/feature_importance_heatmap.png
```

## File Structure

```
real-time-prediction/
├── api_backend.py                          # Backend with SHAP endpoints
├── train_multiple_model_types.py          # Training with SHAP
├── visualize_shap_comparison.py           # Comparison plots
├── shap_analysis.py                       # SHAP helper functions
│
├── model_results/
│   ├── shap/                              # SHAP visualizations
│   │   ├── Individual plots (96 files)
│   │   ├── feature_importance_heatmap.png
│   │   ├── average_feature_importance.png
│   │   ├── library_comparison.png
│   │   └── SHAP_ANALYSIS_REPORT.txt
│   └── all_model_types_results.json       # Includes SHAP data
│
└── library-dashboard/
    └── src/
        ├── App.jsx                        # Main app with navigation
        ├── App.css                        # Main app styles
        ├── ShapAnalysisViz.jsx            # SHAP component
        └── ShapAnalysisViz.css            # SHAP styles
```

## Customization

### Change Default Model/Library

Edit [ShapAnalysisViz.jsx](library-dashboard/src/ShapAnalysisViz.jsx:8-10):

```javascript
const [selectedModel, setSelectedModel] = useState('cnn_only');  // Change default model
const [selectedLibrary, setSelectedLibrary] = useState('miguel_pro');  // Change default library
const [selectedView, setSelectedView] = useState('summary');  // Change default view
```

### Add New Visualizations

1. Generate new SHAP plots in `shap_analysis.py`
2. Save to `model_results/shap/`
3. Add new view type to `viewTypes` array in `ShapAnalysisViz.jsx`
4. Add rendering logic in tab content

### Modify Styling

Edit [ShapAnalysisViz.css](library-dashboard/src/ShapAnalysisViz.css):
- Colors: Change hex values (e.g., `#667eea`)
- Spacing: Adjust padding, margins, gaps
- Sizes: Modify font sizes, image widths
- Responsive: Edit media queries for mobile

## Production Deployment

### Build for Production

```bash
cd library-dashboard
npm run build
```

Creates optimized build in `library-dashboard/build/`

### Serve with Flask

Update `api_backend.py` to serve React build:

```python
from flask import send_from_directory
import os

# Add after other routes
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_react(path):
    if path and os.path.exists(f"library-dashboard/build/{path}"):
        return send_from_directory('library-dashboard/build', path)
    else:
        return send_from_directory('library-dashboard/build', 'index.html')
```

Then visit: `http://localhost:5000`

## Next Steps

1. **Explore Different Models**: Compare LSTM vs CNN vs Hybrid
2. **Analyze Patterns**: Look for consistent important features
3. **Library Comparison**: See how locations differ
4. **Export Insights**: Screenshot key visualizations
5. **Share Reports**: Use SHAP_ANALYSIS_REPORT.txt for documentation

## Useful Commands

```bash
# Full workflow
python train_multiple_model_types.py       # Train with SHAP
python visualize_shap_comparison.py        # Create comparisons
python api_backend.py                      # Start backend (terminal 1)
cd library-dashboard && npm start          # Start frontend (terminal 2)

# Quick checks
ls model_results/shap/                     # List SHAP files
curl http://localhost:5000/api/shap/results  # Test API
```

## Support

For issues:
1. Check browser console (F12) for errors
2. Check terminal logs for backend errors
3. Verify files exist in `model_results/shap/`
4. Review [SHAP_ANALYSIS_GUIDE.md](SHAP_ANALYSIS_GUIDE.md) for details

---

**Last Updated**: 2025-10-30
**Version**: 1.0
**Dashboard**: React + Flask + SHAP
