# Supabase Storage Guide - Save Models to Cloud

This guide explains how to save your trained models, scalers, and results to Supabase Storage instead of local folders.

## Why Use Supabase Storage?

- **Cloud Storage**: Models stored in the cloud, accessible from anywhere
- **No Local Files**: No need for local `saved_models/` and `saved_scalers/` folders
- **Team Sharing**: Multiple team members can access the same models
- **Backup**: Automatic backup and version control
- **Free Tier**: 1GB storage included in Supabase free tier

---

## Setup (One-Time)

### Step 1: Create Storage Bucket in Supabase

1. Go to your Supabase dashboard: https://supabase.com/dashboard
2. Select your project: `library-occupancy-prediction`
3. Click **Storage** in the left sidebar
4. Click **"New bucket"**
5. Enter bucket details:
   - **Name**: `models`
   - **Public**: ✓ Check this (for easy access)
   - Click **"Create bucket"**

Alternatively, run the setup script:

```bash
python setup_supabase_storage.py
```

### Step 2: Verify Your `.env` File

Make sure your `.env` file has Supabase credentials:

```env
SUPABASE_URL=https://jkversfdfvvyyspptolg.supabase.co
SUPABASE_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

---

## Training Models (Save to Supabase)

### Use the Supabase Training Script

Instead of the regular training script, use the Supabase version:

```bash
# Regular version (saves to local folders)
python train_multiple_model_types.py

# Supabase version (saves to cloud)
python train_multiple_model_types_supabase.py
```

### What It Does

The Supabase training script will:

1. Train all 24 models (4 architectures × 6 libraries)
2. Save each model temporarily to local disk
3. **Upload each model to Supabase Storage** → `models/saved_models/`
4. **Upload each scaler to Supabase Storage** → `models/saved_scalers/`
5. **Upload results JSON to Supabase Storage** → `models/model_results/`
6. Save model metadata to Supabase database
7. Clean up temporary local files

### Expected Output

```
================================================================================
TRAIN MULTIPLE MODEL TYPES - SUPABASE VERSION
================================================================================

1. Loading data...
   ✓ Loaded 125,000 records

2. Mapping AP MACs to libraries...
   ✓ Mapped locations

================================================================================
TRAINING MODEL TYPE: LSTM Only
================================================================================

  Training LSTM Only for Miguel Pro Library...
    Records: 45,230, Hours: 2,156
    ✓ R²: 0.9562, RMSE: 12.34
    Uploading model to Supabase...
    ✓ Model uploaded
    Uploading scaler to Supabase...
    ✓ Scaler uploaded
    ✓ Model metadata saved to database

  Training LSTM Only for American Corner...
    Records: 32,104, Hours: 1,842
    ✓ R²: 0.9423, RMSE: 8.92
    Uploading model to Supabase...
    ✓ Model uploaded
    Uploading scaler to Supabase...
    ✓ Scaler uploaded
    ✓ Model metadata saved to database

... (continues for all libraries and model types)

================================================================================
SUPABASE UPLOAD SUMMARY
================================================================================
✓ Models uploaded: 24
✓ Scalers uploaded: 24
✓ Results uploaded: 1

Files in Supabase Storage:
  Bucket: models
  Location: saved_models/ (24 files)
  Location: saved_scalers/ (24 files)
  Location: model_results/ (1 files)

================================================================================
✅ TRAINING COMPLETED - All files saved to Supabase!
================================================================================
```

---

## Using Models from Supabase

### Option 1: Download Models to Local Folders

If you want to use the regular API backend, download models first:

```bash
python download_models_from_supabase.py
```

This will:
- Download all `.keras` files to `saved_models/`
- Download all `.pkl` files to `saved_scalers/`
- Download all `.json` files to `model_results/`

Then run the regular backend:

```bash
python api_backend.py
```

### Option 2: Use Supabase Backend (Direct Cloud Access)

Modify your `api_backend_supabase.py` to load models directly from Supabase (requires additional implementation).

---

## File Structure in Supabase

After training, your Supabase Storage will look like this:

```
Supabase Storage
└── models (bucket)
    ├── saved_models/
    │   ├── lstm_only_miguel_pro_model.keras
    │   ├── lstm_only_american_corner_model.keras
    │   ├── lstm_only_gisbert_2nd_model.keras
    │   ├── lstm_only_gisbert_3rd_model.keras
    │   ├── lstm_only_gisbert_4th_model.keras
    │   ├── lstm_only_gisbert_5th_model.keras
    │   ├── cnn_only_miguel_pro_model.keras
    │   ├── cnn_only_american_corner_model.keras
    │   ... (24 total .keras files)
    │
    ├── saved_scalers/
    │   ├── lstm_only_miguel_pro_scaler.pkl
    │   ├── lstm_only_american_corner_scaler.pkl
    │   ├── lstm_only_gisbert_2nd_scaler.pkl
    │   ... (24 total .pkl files)
    │
    └── model_results/
        └── all_model_types_results.json
```

---

## Viewing Files in Supabase Dashboard

1. Go to Supabase dashboard: https://supabase.com/dashboard
2. Select your project
3. Click **Storage** → **models** bucket
4. Browse folders:
   - `saved_models/` - See all 24 model files
   - `saved_scalers/` - See all 24 scaler files
   - `model_results/` - See results JSON

You can also:
- **Download** individual files
- **Delete** old models
- **View** file sizes and upload dates

---

## Commands Reference

### Setup (One-Time)

```bash
# Create storage bucket in Supabase
python setup_supabase_storage.py
```

### Training

```bash
# Train and save to Supabase
python train_multiple_model_types_supabase.py
```

### Download Models

```bash
# Download all models from Supabase to local folders
python download_models_from_supabase.py
```

### Run API

```bash
# Option 1: Use local models (download first)
python download_models_from_supabase.py
python api_backend.py

# Option 2: Use Supabase backend (cloud models)
python api_backend_supabase.py
```

---

## Workflow Comparison

### Before (Local Storage)

```bash
# Train models
python train_multiple_model_types.py
# ✓ Creates saved_models/ folder locally
# ✓ Creates saved_scalers/ folder locally
# ✓ Creates model_results/ folder locally

# Run API
python api_backend.py
# ✓ Loads models from local folders
```

### After (Supabase Storage)

```bash
# Train models
python train_multiple_model_types_supabase.py
# ✓ Uploads to Supabase Storage
# ✓ No local folders needed
# ✓ Accessible from anywhere

# Run API - Option 1: Download first
python download_models_from_supabase.py
python api_backend.py

# Run API - Option 2: Direct from Supabase
python api_backend_supabase.py
```

---

## Advantages

### Local Storage (Original)
- ✓ Fast access (no network)
- ✓ Works offline
- ✗ Large files (24 models × ~5MB = 120MB+)
- ✗ Not backed up
- ✗ Can't share with team

### Supabase Storage (New)
- ✓ Cloud backup
- ✓ Share with team
- ✓ Access from anywhere
- ✓ Version control
- ✓ Free 1GB storage
- ✗ Requires internet
- ✗ Slightly slower access

---

## Storage Usage

### File Sizes (Approximate)

- Each `.keras` model: ~3-5 MB
- Each `.pkl` scaler: ~1-2 KB
- Results JSON: ~10-20 KB

**Total storage needed:**
- 24 models: ~72-120 MB
- 24 scalers: ~24-48 KB
- Results: ~20 KB

**Total: ~120 MB** (well within 1GB free tier!)

---

## Troubleshooting

### Error: "Bucket does not exist"

**Solution:** Run the setup script to create the bucket:

```bash
python setup_supabase_storage.py
```

Or create it manually in the Supabase dashboard.

### Error: "Failed to upload file"

**Possible causes:**
1. Bucket doesn't exist → Run setup script
2. File too large → Increase bucket size limit
3. Network issue → Check internet connection
4. Wrong credentials → Verify `.env` file

### Models Not Showing in Dashboard

**Check:**
1. Go to Storage → models bucket
2. Look in `saved_models/` folder
3. Files are uploaded with full path: `saved_models/lstm_only_miguel_pro_model.keras`

### Download Script Shows "No files found"

**Solution:**
- Make sure you've run the training script first
- Check the Supabase dashboard to verify files exist
- Verify bucket name is correct (`models`)

---

## Advanced: Custom Bucket Configuration

If you want to use a different bucket name or make it private:

1. Edit `train_multiple_model_types_supabase.py`:

```python
# Change this line
STORAGE_BUCKET = 'models'  # Change to your bucket name
```

2. Create bucket with custom settings:

```python
# In setup_supabase_storage.py, modify:
BUCKETS = [
    {
        'name': 'my-custom-bucket',
        'public': False,  # Set to False for private
        'file_size_limit': 104857600,  # 100MB
    }
]
```

---

## Next Steps

1. **First time setup:**
   ```bash
   python setup_supabase_storage.py
   ```

2. **Train and upload models:**
   ```bash
   python train_multiple_model_types_supabase.py
   ```

3. **Download models when needed:**
   ```bash
   python download_models_from_supabase.py
   ```

4. **Run your API:**
   ```bash
   python api_backend.py
   ```

---

## Summary

You now have three new scripts:

1. **setup_supabase_storage.py** - Creates storage bucket (one-time)
2. **train_multiple_model_types_supabase.py** - Trains and uploads to Supabase
3. **download_models_from_supabase.py** - Downloads models from Supabase

All your models, scalers, and results are now backed up in the cloud! 🎉
