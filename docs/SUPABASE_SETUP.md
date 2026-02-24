# Supabase Setup Guide

This guide will help you migrate from local file storage to Supabase for the Library Occupancy Prediction system.

## Why Supabase?

- **Zero Installation**: Cloud-based, no database software to install
- **Free Tier**: 500MB database, 1GB file storage, 2GB bandwidth
- **Easy Setup**: Takes ~5 minutes to get started
- **Real-time**: Built-in subscriptions for live updates
- **Scalable**: Can handle millions of rows

---

## Step 1: Create Supabase Account

1. Go to [supabase.com](https://supabase.com)
2. Click **"Start your project"**
3. Sign up with GitHub, Google, or email
4. Create a new organization (or use existing)

---

## Step 2: Create a New Project

1. Click **"New Project"**
2. Enter project details:
   - **Name**: library-occupancy-prediction
   - **Database Password**: (generate a strong password - save it!)
   - **Region**: Choose closest to your location
3. Click **"Create new project"**
4. Wait 2-3 minutes for project setup

---

## Step 3: Get API Credentials

1. In your Supabase dashboard, go to **Settings** (⚙️) → **API**
2. Copy these values:
   - **Project URL** (e.g., `https://xxxxx.supabase.co`)
   - **anon public key** (starts with `eyJhbG...`)

---

## Step 4: Create Database Tables

1. In Supabase dashboard, go to **SQL Editor**
2. Click **"New query"**
3. Paste and run this SQL:

```sql
-- WiFi Data Table
CREATE TABLE wifi_data (
    id BIGSERIAL PRIMARY KEY,
    "AP MAC" TEXT,
    "Client MAC" TEXT,
    "Start_dt" TIMESTAMP,
    library_id TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Create index for faster queries
CREATE INDEX idx_wifi_start_dt ON wifi_data("Start_dt");
CREATE INDEX idx_wifi_library ON wifi_data(library_id);
CREATE INDEX idx_wifi_ap_mac ON wifi_data("AP MAC");

-- Predictions Cache Table
CREATE TABLE predictions_cache (
    id INTEGER PRIMARY KEY DEFAULT 1,
    predictions JSONB,
    updated_at TIMESTAMP DEFAULT NOW(),
    CHECK (id = 1)  -- Only allow one row
);

-- Model Metadata Table
CREATE TABLE model_metadata (
    id BIGSERIAL PRIMARY KEY,
    model_type TEXT NOT NULL,
    library_id TEXT NOT NULL,
    metrics JSONB,
    trained_at TIMESTAMP DEFAULT NOW()
);

-- Create index for model metadata
CREATE INDEX idx_model_type ON model_metadata(model_type);
CREATE INDEX idx_model_library ON model_metadata(library_id);

-- Insert initial predictions cache row
INSERT INTO predictions_cache (id, predictions)
VALUES (1, '{}'::jsonb)
ON CONFLICT (id) DO NOTHING;
```

4. Click **"Run"** to execute
5. Verify tables created by going to **Table Editor**

---

## Step 5: Configure Environment Variables

1. Create a `.env` file in your project root:

```bash
touch .env
```

2. Add your Supabase credentials to `.env`:

```env
SUPABASE_URL=https://xxxxx.supabase.co
SUPABASE_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.xxxxx
```

3. Make sure `.env` is in your `.gitignore`:

```bash
echo ".env" >> .gitignore
```

---

## Step 6: Install Dependencies

```bash
uv pip install -r requirements.txt
```

This will install:
- `supabase` - Python client for Supabase
- `python-dotenv` - Load environment variables from .env

---

## Step 7: Migrate Existing Data (Optional)

If you have existing data in `all_data_cleaned.csv`, upload it to Supabase:

```bash
python scripts/migrate_to_supabase.py
```

This will:
- Upload WiFi data from CSV to Supabase
- Upload predictions cache (if exists)
- Upload model metadata from JSON results

**Expected output:**
```
============================================================
SUPABASE MIGRATION TOOL
============================================================
Migrating WiFi data to Supabase...
Found 125000 records in all_data_cleaned.csv
✅ Uploaded 125000 WiFi records to Supabase

Migrating predictions cache to Supabase...
Found 24 predictions in cache
✅ Uploaded predictions cache to Supabase

Migrating model metadata to Supabase...
✅ Uploaded 24 model metadata records to Supabase

============================================================
✅ MIGRATION COMPLETED SUCCESSFULLY!
============================================================
```

---

## Step 8: Start the API

Now use the Supabase-enabled backend:

```bash
python api_backend_supabase.py
```

**Expected output:**
```
============================================================
LIBRARY OCCUPANCY PREDICTION API (Supabase Version)
============================================================
Loading all trained models...
✓ Loaded 24 models successfully
✅ Loaded predictions cache from Supabase
✓ Scheduler started (updates every 60s)
============================================================
 * Running on http://0.0.0.0:5000
```

---

## Step 9: Verify Everything Works

1. **Check API status:**
   ```bash
   curl http://localhost:5000/api/status
   ```

2. **Get predictions:**
   ```bash
   curl http://localhost:5000/api/predictions
   ```

3. **Check Supabase dashboard:**
   - Go to **Table Editor** → `wifi_data`
   - You should see your data rows
   - Go to `predictions_cache` to see cached predictions

---

## Usage Comparison

### Before (Local Files)
```python
# Read from CSV
df = pd.read_csv('all_data_cleaned.csv')

# Save predictions
pickle.dump(cache, open('predictions_cache.pkl', 'wb'))
```

### After (Supabase)
```python
# Read from Supabase
df = SupabaseStorage.get_wifi_data()

# Save predictions
SupabaseStorage.save_predictions_cache(cache)
```

---

## Benefits You Get

1. **No Local Storage**: Data stored securely in the cloud
2. **Easy Backup**: Automatic backups in Supabase
3. **Scalability**: Can handle millions of records
4. **Multi-Server**: Multiple servers can share the same database
5. **Real-time Updates**: Can add live subscriptions later
6. **Query Power**: Full SQL support for analytics

---

## File Structure After Migration

```
your-project/
├── .env                        # Supabase credentials (DO NOT COMMIT)
├── supabase_config.py          # Supabase helper functions
├── api_backend_supabase.py     # New API using Supabase
├── scripts/migrate_to_supabase.py      # One-time migration script
├── saved_models/               # Models still stored locally
├── saved_scalers/              # Scalers still stored locally
└── all_data_cleaned.csv        # Can delete after migration
```

**Note**: Model files (.keras) and scalers (.pkl) remain as local files since they're binary objects and work fine on disk.

---

## Troubleshooting

### Error: "No solution found when resolving dependencies"
- Make sure you updated `requirements.txt` with TensorFlow 2.20.0+ and numpy 2.1.0+

### Error: "SUPABASE_URL and SUPABASE_KEY must be set"
- Check that `.env` file exists in project root
- Verify credentials are correct (no extra spaces)

### Error: "relation 'wifi_data' does not exist"
- Run the SQL table creation script in Step 4

### Data not appearing in Supabase
- Check Supabase dashboard → Table Editor
- Verify migration script completed successfully
- Check API logs for errors

### Slow queries
- Make sure indexes were created (Step 4)
- For large datasets (>1M rows), consider partitioning by date

---

## Next Steps

1. **Set up Row Level Security (RLS)** for production:
   ```sql
   ALTER TABLE wifi_data ENABLE ROW LEVEL SECURITY;
   CREATE POLICY "Allow service role" ON wifi_data FOR ALL USING (true);
   ```

2. **Enable Realtime** (optional):
   - Go to Database → Replication
   - Enable for `predictions_cache` table
   - Get live updates in your frontend

3. **Deploy to Production**:
   - Add `SUPABASE_URL` and `SUPABASE_KEY` to your hosting environment variables
   - Deploy `api_backend_supabase.py` instead of `api_backend.py`

---

## Cost Estimate

**Free Tier Limits:**
- 500 MB database storage
- 1 GB file storage
- 2 GB bandwidth/month
- 50,000 monthly active users

**Your Usage (estimated):**
- WiFi data: ~50-100 MB (assuming 1 year of hourly data)
- Predictions cache: <1 MB
- Monthly bandwidth: ~500 MB (for typical usage)

✅ **You should comfortably fit within the free tier!**

---

## Support

- Supabase Docs: [supabase.com/docs](https://supabase.com/docs)
- Community: [github.com/supabase/supabase/discussions](https://github.com/supabase/supabase/discussions)
- Status: [status.supabase.com](https://status.supabase.com)
