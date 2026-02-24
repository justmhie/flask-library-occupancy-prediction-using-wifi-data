# Tests

Run from **project root**:

```bash
# Backend data processing (requires all_data_cleaned.csv in project root)
python tests/test_backend.py

# SHAP integration (requires saved_models/ and optionally saved_scalers/, all_data_cleaned.csv)
python tests/test_shap_integration.py
```

With pytest (install first: `pip install pytest`):

```bash
pytest tests/ -v
```
