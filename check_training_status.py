"""
Quick check of training status
"""
import os
import glob
import sys
import io

# Fix encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("=" * 60)
print("TRAINING STATUS CHECK")
print("=" * 60)

# Check for model files
models = glob.glob('saved_models/*_model.keras')
scalers = glob.glob('saved_scalers/*_scaler.pkl')
plots = glob.glob('training_results/plots/*.png')

expected_models = ['all', 'miguel_pro', 'american_corner', 'gisbert_2nd', 'gisbert_3rd', 'gisbert_4th', 'gisbert_5th']

print(f"\n✓ Models found: {len(models)}/7")
for model in sorted(models):
    name = os.path.basename(model).replace('_model.keras', '')
    size_mb = os.path.getsize(model) / 1024 / 1024
    print(f"  - {name}: {size_mb:.2f} MB")

print(f"\n✓ Scalers found: {len(scalers)}/7")
print(f"✓ Plots found: {len(plots)}/7")

# Check for results JSON
if os.path.exists('training_results/all_models_results.json'):
    print("\n✓ Training results JSON exists")
    import json
    with open('training_results/all_models_results.json', 'r') as f:
        results = json.load(f)
    print(f"✓ Results for {len(results)} models saved")
else:
    print("\n✗ Training results JSON not found")

# Missing models
model_names = [os.path.basename(m).replace('_model.keras', '') for m in models]
missing = [m for m in expected_models if m not in model_names]

if missing:
    print(f"\n⚠ Missing models: {', '.join(missing)}")
    print("   Training may still be in progress...")
else:
    print("\n✅ ALL MODELS TRAINED SUCCESSFULLY!")

print("=" * 60)
