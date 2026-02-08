# Library Occupancy Prediction - Usage Summary

## Required Data File

You need **ONE** main file: `all_data_cleaned.csv`

**Columns required:**
- `Start_dt` - Timestamp
- `Client MAC` - Device MAC address
- `AP MAC` - Access Point MAC address

**Example:**
```csv
Start_dt,Client MAC,AP MAC
2025-07-20 22:18:57,AA:BB:CC:DD:EE:FF,10:F0:68:29:66:70
```

---

## Quick Start (Local Storage)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train models (creates local folders)
python train_multiple_model_types.py
   → Creates: saved_models/ (24 .keras files)
   → Creates: saved_scalers/ (24 .pkl files)
   → Creates: model_results/ (JSON files)

# 3. Start backend
python api_backend.py

# 4. Start frontend (new terminal)
cd library-dashboard
npm install
npm start
```

**Result:** Dashboard opens at http://localhost:3000

---

## Quick Start (Supabase Cloud Storage)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Setup Supabase storage bucket
python setup_supabase_storage.py

# 3. Train models (uploads to Supabase)
python train_multiple_model_types_supabase.py
   → Uploads to: Supabase Storage/models/saved_models/
   → Uploads to: Supabase Storage/models/saved_scalers/
   → Uploads to: Supabase Storage/models/model_results/

# 4. Download models from Supabase
python download_models_from_supabase.py

# 5. Start backend
python api_backend.py

# 6. Start frontend (new terminal)
cd library-dashboard
npm install
npm start
```

**Result:** Dashboard opens at http://localhost:3000

---

## Main Commands

### Training
```bash
# Local storage
python train_multiple_model_types.py

# Supabase storage
python train_multiple_model_types_supabase.py
```

### Running
```bash
# Backend
python api_backend.py

# Frontend
cd library-dashboard && npm start
```

### Supabase Operations
```bash
# Setup
python setup_supabase_storage.py

# Download from cloud
python download_models_from_supabase.py

# Migrate data
python migrate_to_supabase.py
```

---

## What Gets Created

### Local Storage Mode
```
saved_models/          ← 24 .keras model files
saved_scalers/         ← 24 .pkl scaler files
model_results/         ← JSON results
```

### Supabase Storage Mode
```
Supabase Storage
└── models (bucket)
    ├── saved_models/      ← 24 .keras files (cloud)
    ├── saved_scalers/     ← 24 .pkl files (cloud)
    └── model_results/     ← JSON results (cloud)
```

---

## File Requirements Checklist

Required:
- [x] `all_data_cleaned.csv` (WiFi data)
- [x] `.env` file with Supabase credentials (if using Supabase)

Optional:
- [ ] `rfid_logs.csv` (for RFID validation)
- [ ] `survey_responses.csv` (for survey validation)

---

## Storage Comparison

| Feature | Local Storage | Supabase Storage |
|---------|--------------|------------------|
| Setup | Simple | Requires Supabase account |
| Speed | Fast | Network dependent |
| Backup | Manual | Automatic |
| Sharing | Hard | Easy |
| Space | Local disk | Cloud (1GB free) |
| Offline | Yes | No |

---

## Common Issues

### "No such file: all_data_cleaned.csv"
**Solution:** Place your WiFi data CSV in the project root

### "No models found"
**Solution:** Run training first: `python train_multiple_model_types.py`

### "Bucket does not exist"
**Solution:** Run setup: `python setup_supabase_storage.py`

### "Connection refused on port 5000"
**Solution:** Start backend: `python api_backend.py`

---

## Documentation Files

- [README.md](README.md) - Complete project documentation
- [SUPABASE_STORAGE_GUIDE.md](SUPABASE_STORAGE_GUIDE.md) - Detailed Supabase guide
- [QUICK_SUPABASE_COMMANDS.md](QUICK_SUPABASE_COMMANDS.md) - Quick command reference
- [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) - Research recommendations
- **[USAGE_SUMMARY.md](USAGE_SUMMARY.md)** - This file (quick overview)

---

## Need Help?

1. Check the README.md
2. Check SUPABASE_STORAGE_GUIDE.md for cloud storage
3. Check QUICK_SUPABASE_COMMANDS.md for commands
4. Look at the troubleshooting sections

---

**Quick tip:** If you forget what to do, just run:
```bash
ls *.md
```
And read the documentation files! 📚
