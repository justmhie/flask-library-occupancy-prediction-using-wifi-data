# Quick Commands - Supabase Storage

## Setup (First Time Only)

```bash
# 1. Create storage bucket
python setup_supabase_storage.py
```

---

## Train Models & Save to Supabase

```bash
# Train all models and upload to Supabase Storage
python train_multiple_model_types_supabase.py
```

**What happens:**
- Trains 24 models (4 types × 6 libraries)
- Uploads to Supabase Storage:
  - `models/saved_models/` ← 24 .keras files
  - `models/saved_scalers/` ← 24 .pkl files
  - `models/model_results/` ← results JSON
- Saves metadata to Supabase database
- Cleans up temporary local files

---

## Download Models from Supabase

```bash
# Download all models to local folders
python download_models_from_supabase.py
```

**What happens:**
- Downloads from Supabase Storage:
  - `saved_models/` ← 24 .keras files
  - `saved_scalers/` ← 24 .pkl files
  - `model_results/` ← results JSON

---

## Run the API

### Option 1: Use Local Models (After Download)

```bash
python download_models_from_supabase.py
python api_backend.py
```

### Option 2: Use Supabase Backend

```bash
python api_backend_supabase.py
```

---

## Complete Workflow

```bash
# First time setup
python setup_supabase_storage.py

# Train and upload
python train_multiple_model_types_supabase.py

# Download when needed
python download_models_from_supabase.py

# Run API
python api_backend.py
```

---

## View Files in Supabase

1. Go to: https://supabase.com/dashboard
2. Select your project
3. Click: **Storage** → **models** bucket
4. Browse folders:
   - `saved_models/` (24 model files)
   - `saved_scalers/` (24 scaler files)
   - `model_results/` (results JSON)

---

## File Locations

### In Supabase Storage:
```
models/
├── saved_models/
│   ├── lstm_only_miguel_pro_model.keras
│   ├── cnn_only_miguel_pro_model.keras
│   └── ... (24 total)
├── saved_scalers/
│   ├── lstm_only_miguel_pro_scaler.pkl
│   └── ... (24 total)
└── model_results/
    └── all_model_types_results.json
```

### After Download (Local):
```
your-project/
├── saved_models/
│   └── ... (24 .keras files)
├── saved_scalers/
│   └── ... (24 .pkl files)
└── model_results/
    └── all_model_types_results.json
```

---

## Troubleshooting

### Bucket doesn't exist
```bash
python setup_supabase_storage.py
```

### Need to re-download models
```bash
python download_models_from_supabase.py
```

### Check what's in Supabase
- Go to Supabase dashboard
- Storage → models bucket
- Browse folders

---

That's it! Your models are now in the cloud. ☁️
