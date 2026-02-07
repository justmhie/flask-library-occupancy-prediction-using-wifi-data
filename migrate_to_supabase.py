"""
Migration script to move local data to Supabase.
Run this once to upload your existing data.
"""
import pandas as pd
from supabase_config import SupabaseStorage
import os
import pickle

def migrate_wifi_data():
    """Migrate CSV data to Supabase."""
    print("Migrating WiFi data to Supabase...")

    if not os.path.exists('all_data_cleaned.csv'):
        print("❌ all_data_cleaned.csv not found. Skipping WiFi data migration.")
        return

    df = pd.read_csv('all_data_cleaned.csv')
    print(f"Found {len(df)} records in all_data_cleaned.csv")

    # Upload data
    count = SupabaseStorage.save_wifi_data(df)
    print(f"✅ Uploaded {count} WiFi records to Supabase")

def migrate_predictions_cache():
    """Migrate predictions cache to Supabase."""
    print("\nMigrating predictions cache to Supabase...")

    if not os.path.exists('predictions_cache.pkl'):
        print("❌ predictions_cache.pkl not found. Skipping cache migration.")
        return

    with open('predictions_cache.pkl', 'rb') as f:
        cache = pickle.load(f)

    print(f"Found {len(cache)} predictions in cache")

    # Upload cache
    SupabaseStorage.save_predictions_cache(cache)
    print(f"✅ Uploaded predictions cache to Supabase")

def migrate_model_metadata():
    """Migrate model results to Supabase."""
    print("\nMigrating model metadata to Supabase...")

    if not os.path.exists('model_results/all_model_types_results.json'):
        print("❌ Model results not found. Skipping metadata migration.")
        return

    import json
    with open('model_results/all_model_types_results.json', 'r') as f:
        results = json.load(f)

    uploaded = 0
    for model_type, model_data in results.items():
        if 'libraries' in model_data:
            for library_id, library_data in model_data['libraries'].items():
                if 'metrics' in library_data:
                    SupabaseStorage.upload_model_metadata(
                        model_type=model_type,
                        library_id=library_id,
                        metrics=library_data['metrics']
                    )
                    uploaded += 1

    print(f"✅ Uploaded {uploaded} model metadata records to Supabase")

if __name__ == "__main__":
    print("=" * 60)
    print("SUPABASE MIGRATION TOOL")
    print("=" * 60)

    try:
        migrate_wifi_data()
        migrate_predictions_cache()
        migrate_model_metadata()

        print("\n" + "=" * 60)
        print("✅ MIGRATION COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Verify data in Supabase dashboard")
        print("2. Update api_backend.py to use Supabase")
        print("3. Test the API endpoints")

    except Exception as e:
        print(f"\n❌ Migration failed: {str(e)}")
        print("\nPlease check:")
        print("1. SUPABASE_URL and SUPABASE_KEY are set in .env")
        print("2. Tables exist in Supabase (see setup instructions)")
        print("3. Network connection is working")
