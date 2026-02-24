"""
Setup Supabase Storage Buckets
Creates the necessary storage buckets for models, scalers, and results
Run from project root: python scripts/setup_supabase_storage.py
"""
import os
import sys
if __name__ == "__main__":
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(_root)
    if _root not in sys.path:
        sys.path.insert(0, _root)

from supabase_config import supabase

print("=" * 80)
print("SUPABASE STORAGE SETUP")
print("=" * 80)

# Bucket configuration
BUCKETS = [
    {
        'name': 'models',
        'public': True,  # Set to True for easy access, False for private
        'file_size_limit': 52428800,  # 50MB
        'allowed_mime_types': None  # Allow all file types
    }
]

print("\nCreating storage buckets...")

for bucket_config in BUCKETS:
    bucket_name = bucket_config['name']

    try:
        # Try to create the bucket
        result = supabase.storage.create_bucket(
            bucket_name,
            options={
                'public': bucket_config['public'],
                'file_size_limit': bucket_config['file_size_limit']
            }
        )
        print(f"✓ Created bucket: {bucket_name}")
    except Exception as e:
        error_msg = str(e)
        if 'already exists' in error_msg.lower() or 'duplicate' in error_msg.lower():
            print(f"✓ Bucket already exists: {bucket_name}")
        else:
            print(f"⚠ Error creating bucket {bucket_name}: {e}")

print("\n" + "=" * 80)
print("CREATING FOLDER STRUCTURE IN BUCKET")
print("=" * 80)

# Create placeholder files to establish folder structure
folders = ['saved_models', 'saved_scalers', 'model_results']

print("\nNote: Supabase Storage doesn't have explicit folders.")
print("Folders are created automatically when you upload files with paths.")
print("\nFolder structure will be created when you run:")
print("  python scripts/train_multiple_model_types_supabase.py")

print("\n" + "=" * 80)
print("✅ STORAGE SETUP COMPLETE")
print("=" * 80)

print("\nNext steps:")
print("1. Run: python scripts/train_multiple_model_types_supabase.py")
print("2. Models and scalers will be uploaded to Supabase Storage")
print("3. Use download_models_from_supabase.py to download them when needed")
