"""
Download Models from Supabase Storage
Downloads all trained models, scalers, and results from Supabase to local folders
"""

import os
import json
from supabase_config import SupabaseStorage, supabase

print("=" * 80)
print("DOWNLOAD MODELS FROM SUPABASE")
print("=" * 80)

# Configuration
STORAGE_BUCKET = 'models'

# Create local directories
os.makedirs('saved_models', exist_ok=True)
os.makedirs('saved_scalers', exist_ok=True)
os.makedirs('model_results', exist_ok=True)

# ============================================
# DOWNLOAD MODELS
# ============================================

print("\n1. Downloading models...")

try:
    # List all files in saved_models folder
    model_files = supabase.storage.from_(STORAGE_BUCKET).list('saved_models')

    if not model_files:
        print("   ⚠ No models found in Supabase Storage")
    else:
        for file_info in model_files:
            if isinstance(file_info, dict) and 'name' in file_info:
                filename = file_info['name']
                if filename.endswith('.keras'):
                    remote_path = f"saved_models/{filename}"
                    local_path = f"saved_models/{filename}"

                    print(f"   Downloading {filename}...", end='')
                    success = SupabaseStorage.download_file_from_storage(
                        STORAGE_BUCKET,
                        remote_path,
                        local_path
                    )

                    if success:
                        print(" ✓")
                    else:
                        print(" ⚠ Failed")

        print(f"   ✓ Downloaded {len([f for f in model_files if isinstance(f, dict) and f.get('name', '').endswith('.keras')])} models")
except Exception as e:
    print(f"   ⚠ Error listing models: {e}")

# ============================================
# DOWNLOAD SCALERS
# ============================================

print("\n2. Downloading scalers...")

try:
    # List all files in saved_scalers folder
    scaler_files = supabase.storage.from_(STORAGE_BUCKET).list('saved_scalers')

    if not scaler_files:
        print("   ⚠ No scalers found in Supabase Storage")
    else:
        for file_info in scaler_files:
            if isinstance(file_info, dict) and 'name' in file_info:
                filename = file_info['name']
                if filename.endswith('.pkl'):
                    remote_path = f"saved_scalers/{filename}"
                    local_path = f"saved_scalers/{filename}"

                    print(f"   Downloading {filename}...", end='')
                    success = SupabaseStorage.download_file_from_storage(
                        STORAGE_BUCKET,
                        remote_path,
                        local_path
                    )

                    if success:
                        print(" ✓")
                    else:
                        print(" ⚠ Failed")

        print(f"   ✓ Downloaded {len([f for f in scaler_files if isinstance(f, dict) and f.get('name', '').endswith('.pkl')])} scalers")
except Exception as e:
    print(f"   ⚠ Error listing scalers: {e}")

# ============================================
# DOWNLOAD RESULTS
# ============================================

print("\n3. Downloading results...")

try:
    # List all files in model_results folder
    result_files = supabase.storage.from_(STORAGE_BUCKET).list('model_results')

    if not result_files:
        print("   ⚠ No results found in Supabase Storage")
    else:
        for file_info in result_files:
            if isinstance(file_info, dict) and 'name' in file_info:
                filename = file_info['name']
                if filename.endswith('.json'):
                    remote_path = f"model_results/{filename}"
                    local_path = f"model_results/{filename}"

                    print(f"   Downloading {filename}...", end='')
                    success = SupabaseStorage.download_file_from_storage(
                        STORAGE_BUCKET,
                        remote_path,
                        local_path
                    )

                    if success:
                        print(" ✓")
                    else:
                        print(" ⚠ Failed")

        print(f"   ✓ Downloaded {len([f for f in result_files if isinstance(f, dict) and f.get('name', '').endswith('.json')])} result files")
except Exception as e:
    print(f"   ⚠ Error listing results: {e}")

# ============================================
# SUMMARY
# ============================================

print("\n" + "=" * 80)
print("DOWNLOAD SUMMARY")
print("=" * 80)

# Count downloaded files
model_count = len([f for f in os.listdir('saved_models') if f.endswith('.keras')])
scaler_count = len([f for f in os.listdir('saved_scalers') if f.endswith('.pkl')])
result_count = len([f for f in os.listdir('model_results') if f.endswith('.json')])

print(f"✓ Models: {model_count} files in saved_models/")
print(f"✓ Scalers: {scaler_count} files in saved_scalers/")
print(f"✓ Results: {result_count} files in model_results/")

print("\n" + "=" * 80)
print("✅ DOWNLOAD COMPLETE")
print("=" * 80)

print("\nYou can now run:")
print("  python api_backend.py")
print("or")
print("  python api_backend_supabase.py")
