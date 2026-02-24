# 🚀 React Dashboard Setup Guide

## Complete System for Library Occupancy Dashboard with Graphs & Automation

You wanted: *"show the graph and users of each library so maybe just choose a button or dropdown, not limited to google colab, can use react or js, how to automate this easily"*

Here's the **complete solution**! 🎯

---

## 📦 What You're Getting

### 1. **React Frontend Dashboard**
- ✅ Beautiful modern UI with graphs
- ✅ Dropdown to select each library
- ✅ Real-time charts (line, bar, area charts)
- ✅ Auto-refresh every 60 seconds
- ✅ Mobile responsive

### 2. **Flask API Backend**
- ✅ Serves predictions via REST API
- ✅ Automatic updates every 60 seconds
- ✅ Caches predictions for fast response
- ✅ Background scheduler (no manual work!)

### 3. **Complete Automation**
- ✅ Auto-updates predictions
- ✅ No manual intervention needed
- ✅ Deploy once, runs forever

---

## 🎯 Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                   USER BROWSER                          │
│  (React Dashboard - Graphs, Dropdown, Real-time data)   │
└────────────────────┬────────────────────────────────────┘
                     │ HTTP Requests every 60s
                     ▼
┌─────────────────────────────────────────────────────────┐
│              FLASK API BACKEND                          │
│  • REST API endpoints (/api/predictions)                │
│  • Background scheduler (auto-updates)                  │
│  • ML predictions generation                            │
└────────────────────┬────────────────────────────────────┘
                     │ Reads data
                     ▼
┌─────────────────────────────────────────────────────────┐
│           all_data_cleaned.csv                          │
│  (Your cleaned WiFi data)                               │
└─────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start (3 Steps)

### Step 1: Setup Backend (5 minutes)

```bash
# 1. Install dependencies
pip install flask flask-cors pandas numpy scikit-learn tensorflow apscheduler

# 2. Place your files:
#    - api_backend.py
#    - all_data_cleaned.csv
#    (in same directory)

# 3. Run the backend
python api_backend.py

# Output:
# INFO: Performing initial prediction update...
# INFO: Cache updated successfully
# INFO: Scheduler started - updating every 60 seconds
# INFO: Starting Flask API server...
# * Running on http://0.0.0.0:5000
```

**Done!** Backend is now running and auto-updating! ✅

### Step 2: Setup Frontend (5 minutes)

```bash
# 1. Create React app
npx create-react-app library-dashboard
cd library-dashboard

# 2. Install dependencies
npm install recharts

# For shadcn/ui components:
npx shadcn-ui@latest init
npx shadcn-ui@latest add card select alert

# 3. Copy react_dashboard.jsx to src/App.js

# 4. Update API endpoint in App.js:
# Change: const response = await fetch('/api/predictions');
# To: const response = await fetch('http://localhost:5000/api/predictions');

# 5. Start dev server
npm start
```

**Done!** Dashboard opens at http://localhost:3000 ✅

### Step 3: View Dashboard

1. Open browser to `http://localhost:3000`
2. Select library from dropdown
3. See real-time graphs!
4. Auto-refreshes every 60 seconds

---

## 📊 Features Breakdown

### Dashboard Features:

1. **Library Selector**
   - Dropdown menu
   - Choose individual library or "All Libraries"
   - Instant graph update

2. **Current Stats Cards**
   - Current users (right now)
   - Predicted users (next hour)
   - 24-hour average
   - Current capacity percentage

3. **Hourly Trend Chart** (Area Chart)
   - Shows last 24 hours actual data
   - Shows next 6 hours predictions
   - Smooth gradient visualization
   - Hover for exact values

4. **Daily Pattern Chart** (Bar Chart)
   - Average usage by hour (last 7 days)
   - Shows peak times
   - Helps identify busy periods

5. **Next 6 Hours Forecast** (Line Chart)
   - Hour-by-hour predictions
   - Confidence intervals (upper/lower bounds)
   - Helps plan ahead

6. **Additional Stats**
   - Peak today and time
   - 7-day average
   - Trend indicator (up/down/stable)
   - Model accuracy

7. **Auto-Refresh**
   - Updates every 60 seconds automatically
   - Toggle on/off button
   - Shows last update time

8. **Capacity Alert**
   - Red alert when > 80% capacity
   - Suggests redirecting users

---

## 🤖 Complete Automation

### Backend Auto-Updates:

```python
# api_backend.py automatically:
1. Updates predictions every 60 seconds
2. Caches results for fast response
3. Saves cache to disk (survives restarts)
4. Runs in background (no manual intervention)
```

### How It Works:

```python
# Background scheduler runs this every 60 seconds:
def update_predictions_cache():
    1. Load all_data_cleaned.csv
    2. Calculate current occupancy
    3. Generate predictions for each library
    4. Calculate trends and stats
    5. Save to cache
    6. Serve via API
```

### Frontend Auto-Refresh:

```javascript
// React dashboard automatically:
useEffect(() => {
    // Fetch on mount
    fetchPredictions();
    
    // Auto-refresh every 60 seconds
    const interval = setInterval(fetchPredictions, 60000);
    return () => clearInterval(interval);
}, [autoRefresh]);
```

---

## 🌐 Deployment Options

### Option 1: Simple Local (Development)

```bash
# Terminal 1: Backend
python api_backend.py

# Terminal 2: Frontend
cd library-dashboard
npm start

# Access: http://localhost:3000
```

**Best for:** Testing, development

### Option 2: Deploy Backend to Cloud

```bash
# Deploy backend to Heroku, AWS, Google Cloud, etc.

# Heroku example:
heroku create library-occupancy-api
git push heroku main

# Your API is now at: https://your-app.herokuapp.com

# Update frontend API URL to deployed backend
```

**Best for:** Production, remote access

### Option 3: Deploy Everything (Full Production)

```bash
# Backend: Deploy to cloud (Heroku/AWS/GCP)
# Frontend: Deploy to Vercel/Netlify (free!)

# Vercel:
cd library-dashboard
npm run build
vercel deploy

# Your dashboard is now at: https://your-dashboard.vercel.app
```

**Best for:** Public access, sharing with stakeholders

### Option 4: Single Server (All-in-One)

```bash
# Build React for production
cd library-dashboard
npm run build

# Serve from Flask:
# Add to api_backend.py:
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_frontend(path):
    if path != "" and os.path.exists(f"build/{path}"):
        return send_from_directory('build', path)
    return send_from_directory('build', 'index.html')

# Run single server:
python api_backend.py

# Access everything at: http://localhost:5000
```

**Best for:** Simple deployment, one server

---

## ⚙️ Advanced Automation

### 1. Automatic Data Updates

```python
# Create update_data.py:
import pandas as pd
from datetime import datetime

def append_new_data():
    """Add new WiFi data to all_data_cleaned.csv"""
    # Load new raw data
    new_data = pd.read_csv('new_wifi_data.csv')
    
    # Process it (use cleandata2_fixed.py logic)
    # ...
    
    # Append to existing cleaned data
    existing = pd.read_csv('all_data_cleaned.csv')
    updated = pd.concat([existing, new_data])
    updated.to_csv('all_data_cleaned.csv', index=False)
    
    print(f"Data updated: {datetime.now()}")

# Schedule to run daily
from apscheduler.schedulers.blocking import BlockingScheduler

scheduler = BlockingScheduler()
scheduler.add_job(append_new_data, 'interval', hours=24)
scheduler.start()
```

### 2. Email Alerts

```python
# Add to api_backend.py:
import smtplib
from email.mime.text import MIMEText

def send_alert(library, occupancy, capacity):
    """Send email when capacity > 80%"""
    if occupancy / capacity > 0.8:
        msg = MIMEText(f"High occupancy at {library}: {occupancy} users")
        msg['Subject'] = f'Capacity Alert: {library}'
        msg['From'] = 'alerts@library.com'
        msg['To'] = 'admin@library.com'
        
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login('your_email@gmail.com', 'password')
            server.send_message(msg)
```

### 3. Logging and Monitoring

```python
# Add to api_backend.py:
import logging
from logging.handlers import RotatingFileHandler

# Setup file logging
handler = RotatingFileHandler(
    'predictions.log',
    maxBytes=10000000,  # 10MB
    backupCount=5
)
handler.setFormatter(logging.Formatter(
    '%(asctime)s %(levelname)s: %(message)s'
))
logger.addHandler(handler)

# Now all updates are logged to predictions.log
```

### 4. Database Storage (Optional)

```python
# Instead of CSV, use PostgreSQL/MongoDB:
from sqlalchemy import create_engine

engine = create_engine('postgresql://user:pass@localhost/library_db')

# Save predictions to database
def save_predictions_to_db(predictions):
    df = pd.DataFrame([predictions])
    df.to_sql('predictions', engine, if_exists='append')

# Query historical predictions
def get_prediction_history(library, days=7):
    query = f"""
        SELECT * FROM predictions 
        WHERE location = '{library}' 
        AND timestamp > NOW() - INTERVAL '{days} days'
    """
    return pd.read_sql(query, engine)
```

---

## 🔧 Configuration

### Backend Configuration (api_backend.py):

```python
# Update these variables:
CLEANED_DATA_FILE = 'all_data_cleaned.csv'  # Your data file
UPDATE_INTERVAL = 60  # Update frequency (seconds)
MODEL_PATH = 'saved_models'  # Path to ML models
PORT = 5000  # API port
```

### Frontend Configuration (react_dashboard.jsx):

```javascript
// Update API endpoint:
const API_URL = 'http://localhost:5000';  // Development
// const API_URL = 'https://your-api.com';  // Production

// Update refresh interval:
const REFRESH_INTERVAL = 60000;  // 60 seconds
```

### Auto-Update Frequency:

```python
# In api_backend.py, change:
UPDATE_INTERVAL = 30  # Update every 30 seconds (faster)
UPDATE_INTERVAL = 300  # Update every 5 minutes (slower)
```

---

## 📁 Complete File Structure

```
library-dashboard/
├── backend/
│   ├── api_backend.py           ✨ Flask API with auto-updates
│   ├── all_data_cleaned.csv     📊 Your data
│   ├── requirements.txt         📋 Python dependencies
│   └── predictions_cache.pkl    💾 Auto-generated cache
│
├── frontend/
│   ├── package.json
│   ├── src/
│   │   ├── App.js              ✨ React dashboard
│   │   ├── index.js
│   │   └── components/
│   │       └── ui/             📦 shadcn/ui components
│   └── public/
│
└── docs/
    └── REACT_DASHBOARD_SETUP.md  📖 This file
```

---

## 🎨 Customization

### Change Colors:

```javascript
// In react_dashboard.jsx:

// Gradient colors:
<div className="bg-gradient-to-br from-blue-500 to-blue-600">
// Change to:
<div className="bg-gradient-to-br from-purple-500 to-purple-600">

// Chart colors:
fill="#3b82f6"  // Blue
// Change to:
fill="#8b5cf6"  // Purple
```

### Add More Charts:

```javascript
// Add pie chart for distribution:
<PieChart width={400} height={400}>
  <Pie 
    data={libraryData.distribution} 
    dataKey="value" 
    nameKey="name"
    fill="#8884d8"
  />
</PieChart>
```

### Custom Alerts:

```javascript
// Add custom threshold alert:
{libraryData.predicted > 100 && (
  <Alert className="bg-yellow-50">
    <AlertDescription>
      ⚠️ High traffic expected! Predicted: {libraryData.predicted} users
    </AlertDescription>
  </Alert>
)}
```

---

## 🆘 Troubleshooting

### Backend Issues:

**Q: Backend won't start**
```bash
A: Check if all dependencies installed:
   pip install -r requirements.txt
   
   Check if port 5000 is available:
   lsof -i :5000
   
   Kill process if needed:
   kill -9 <PID>
```

**Q: Predictions not updating**
```bash
A: Check logs:
   # Backend prints logs to console
   # Look for: "Cache updated successfully"
   
   Force refresh:
   curl -X POST http://localhost:5000/api/refresh
```

**Q: CORS errors in browser**
```python
A: Make sure flask-cors is installed:
   pip install flask-cors
   
   And CORS is enabled in api_backend.py:
   from flask_cors import CORS
   CORS(app)
```

### Frontend Issues:

**Q: Dropdown not working**
```bash
A: Install shadcn/ui components:
   npx shadcn-ui@latest add select
```

**Q: Charts not showing**
```bash
A: Install recharts:
   npm install recharts
   
   Check import:
   import { LineChart, Line, ... } from 'recharts';
```

**Q: API connection failed**
```javascript
A: Check API_URL in react_dashboard.jsx:
   const response = await fetch('http://localhost:5000/api/predictions');
   
   Test API directly:
   Open http://localhost:5000/api/predictions in browser
```

---

## 🚀 Performance Optimization

### Backend:

1. **Use Redis for caching:**
```python
import redis
r = redis.Redis(host='localhost', port=6379, db=0)

def cache_predictions(data):
    r.setex('predictions', 60, pickle.dumps(data))

def get_cached_predictions():
    cached = r.get('predictions')
    return pickle.loads(cached) if cached else None
```

2. **Use ML models for better accuracy:**
```python
# Train models once, use saved models for predictions
# See realtime_predictions.py for implementation
```

### Frontend:

1. **Lazy loading:**
```javascript
import React, { lazy, Suspense } from 'react';
const Dashboard = lazy(() => import('./Dashboard'));

<Suspense fallback={<Loading />}>
  <Dashboard />
</Suspense>
```

2. **Memoization:**
```javascript
import { useMemo } from 'react';

const processedData = useMemo(() => {
  return libraryData.hourly_data.filter(...);
}, [libraryData]);
```

---

## 📊 Sample API Responses

### GET /api/predictions

```json
{
  "libraries": {
    "Miguel_Pro": {
      "location": "Miguel_Pro",
      "current": 45,
      "predicted": 48,
      "change": 3,
      "avg_24h": 42,
      "max_capacity": 100,
      "hourly_data": [...],
      "daily_pattern": [...],
      "next_hours": [...]
    },
    ...
  },
  "overall": {
    "total_current": 123,
    "total_predicted": 128,
    "total_change": 5
  }
}
```

---

## ✅ Deployment Checklist

- [ ] Backend running (`python api_backend.py`)
- [ ] Frontend running (`npm start`)
- [ ] Can access dashboard (http://localhost:3000)
- [ ] Dropdown shows all libraries
- [ ] Charts display correctly
- [ ] Auto-refresh working (check every 60s)
- [ ] API responds (`http://localhost:5000/api/predictions`)
- [ ] Data updating automatically (check logs)
- [ ] Mobile responsive (test on phone)
- [ ] Production build works (`npm run build`)

---

## 🎉 You're Done!

You now have a **complete, automated, production-ready dashboard** with:
- ✅ Beautiful React UI with graphs
- ✅ Dropdown to select libraries
- ✅ Real-time predictions
- ✅ Automatic updates (no manual work!)
- ✅ Multiple charts (line, bar, area)
- ✅ Mobile responsive
- ✅ Easy deployment

**Just run the backend and frontend - everything else is automatic!** 🚀

---

**Next Steps:**
1. Run backend: `python api_backend.py`
2. Run frontend: `npm start`
3. Open browser: `http://localhost:3000`
4. Select library from dropdown
5. Watch real-time predictions! 📊

---

For questions or issues, check the troubleshooting section above!
