# Scripts

Run all scripts from the **project root**:

```bash
# Training (local)
python scripts/train_multiple_model_types.py
python scripts/train_model.py
python scripts/train_all_libraries.py

# Training (Supabase)
python scripts/setup_supabase_storage.py          # First time only
python scripts/train_multiple_model_types_supabase.py
python scripts/download_models_from_supabase.py
python scripts/migrate_to_supabase.py

# Analysis & utilities
python scripts/check_training_status.py
python scripts/verify_ap_mapping.py
python scripts/train_ablation_study.py
python scripts/train_granular_ablation_study.py
python scripts/train_with_exam_awareness.py
python scripts/student_survey_validation.py
python scripts/wifi_rfid_correlation.py
python scripts/visualize_shap_comparison.py
```

To use a script as a module (e.g. in your own code):

```python
from scripts.student_survey_validation import StudentSurveyValidator
from scripts.wifi_rfid_correlation import WiFiRFIDCorrelation
```
