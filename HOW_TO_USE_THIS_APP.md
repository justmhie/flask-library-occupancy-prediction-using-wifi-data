# How to Use This App - Simple Guide

## What This App Does

Predicts library occupancy using WiFi access point data and displays it on a dashboard.

---

## What You Need

### 1. Main Data File (REQUIRED)
**File:** `all_data_cleaned.csv`  
**Location:** Put it in the project root (same folder as this README)

**Must have these 3 columns:**
```csv
Start_dt,Client MAC,AP MAC
2025-07-20 22:18:57,AA:BB:CC:DD:EE:FF,10:F0:68:29:66:70
```

### 2. Supabase Account (OPTIONAL - for cloud storage)
If you want to store models in the cloud instead of locally.

---

## How to Run (3 Simple Steps)

### Step 1: Train Models

Choose ONE option:

**Option A: Save to Local Folders** (simpler)
```bash
python train_multiple_model_types.py
```
Creates folders: `saved_models/`, `saved_scalers/`, `model_results/`

**Option B: Save to Supabase Cloud** (for backup/sharing)
```bash
python setup_supabase_storage.py          # First time only
python train_multiple_model_types_supabase.py
```
Uploads to Supabase Storage

---

### Step 2: Start Backend API

```bash
python api_backend.py
```

Should show:
```
✓ Loaded 24 models successfully
Running on http://127.0.0.1:5000
```

Keep this terminal running!

---

### Step 3: Start Dashboard

Open a NEW terminal:

```bash
cd library-dashboard
npm install          # First time only
npm start
```

Dashboard opens automatically at: **http://localhost:3000**

---

## That's It!

You now have:
- Backend running on port 5000
- Dashboard running on port 3000
- Real-time occupancy predictions updating every 60 seconds

---

## Commands Cheat Sheet

### First Time Setup
```bash
pip install -r requirements.txt
cd library-dashboard && npm install && cd ..
```

### Training Models
```bash
# Local storage
python train_multiple_model_types.py

# Supabase storage (cloud)
python setup_supabase_storage.py
python train_multiple_model_types_supabase.py
```

### Running the App
```bash
# Terminal 1: Backend
python api_backend.py

# Terminal 2: Frontend
cd library-dashboard
npm start
```

### Download from Supabase (if using cloud)
```bash
python download_models_from_supabase.py
```

---

## Folder Structure

### What You Provide
```
your-project/
└── all_data_cleaned.csv    ← YOUR WiFi DATA (required)
```

### What Gets Created (Local Mode)
```
your-project/
├── saved_models/           ← 24 model files (created by training)
├── saved_scalers/          ← 24 scaler files (created by training)
└── model_results/          ← Results JSON (created by training)
```

### What Gets Created (Supabase Mode)
```
Supabase Storage (Cloud)
└── models/
    ├── saved_models/       ← 24 model files
    ├── saved_scalers/      ← 24 scaler files
    └── model_results/      ← Results JSON
```

---

## What Each File Does

### Main Scripts
- `train_multiple_model_types.py` - Train models, save locally
- `train_multiple_model_types_supabase.py` - Train models, upload to cloud
- `api_backend.py` - Backend API server
- `library-dashboard/` - React frontend dashboard

### Supabase Scripts (Optional)
- `setup_supabase_storage.py` - Create cloud storage bucket
- `download_models_from_supabase.py` - Download models from cloud
- `migrate_to_supabase.py` - Upload existing data to cloud

### Configuration
- `ap_location_mapping.py` - Maps WiFi access points to libraries
- `.env` - Supabase credentials (if using cloud)

---

## Common Questions

### Q: Do I need WiFi data in a specific format?
**A:** Yes, CSV file with columns: `Start_dt`, `Client MAC`, `AP MAC`

### Q: Can I use a zip file?
**A:** No, extract the zip and use the CSV file directly.

### Q: Do I need RFID data?
**A:** No, RFID data is optional (only for validation studies).

### Q: What's the difference between local and Supabase storage?
**A:** 
- **Local**: Models saved on your computer (faster, offline)
- **Supabase**: Models saved in cloud (backup, sharing, access from anywhere)

### Q: How long does training take?
**A:** About 30-60 minutes depending on your data size and computer.

### Q: Can I train models again with new data?
**A:** Yes! Just update `all_data_cleaned.csv` and run the training script again.

---

## Troubleshooting

### Problem: "File not found: all_data_cleaned.csv"
**Solution:** Make sure the CSV file is in the project root folder.

### Problem: "No models found"
**Solution:** Run training first: `python train_multiple_model_types.py`

### Problem: Backend won't start
**Solutions:**
1. Install dependencies: `pip install -r requirements.txt`
2. Train models first
3. Check if port 5000 is available

### Problem: Frontend shows errors
**Solutions:**
1. Make sure backend is running first
2. Install dependencies: `cd library-dashboard && npm install`
3. Clear browser cache

### Problem: Supabase errors
**Solutions:**
1. Check `.env` file has correct credentials
2. Run setup: `python setup_supabase_storage.py`
3. Verify bucket exists in Supabase dashboard

---

## Need More Help?

Read these guides in order:

1. **[USAGE_SUMMARY.md](USAGE_SUMMARY.md)** - Quick overview (you are here!)
2. **[README.md](README.md)** - Complete documentation
3. **[QUICK_SUPABASE_COMMANDS.md](QUICK_SUPABASE_COMMANDS.md)** - Supabase commands
4. **[SUPABASE_STORAGE_GUIDE.md](SUPABASE_STORAGE_GUIDE.md)** - Detailed Supabase guide

---

## Quick Workflow Summary

```
1. Get WiFi data → all_data_cleaned.csv
2. Train models → python train_multiple_model_types.py
3. Start backend → python api_backend.py
4. Start frontend → cd library-dashboard && npm start
5. Open browser → http://localhost:3000
```

Done! 🎉
