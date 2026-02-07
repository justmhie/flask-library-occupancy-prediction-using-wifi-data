# Quick Start: Implementing Recommendations

This is a quick-start guide for implementing the research recommendations. For detailed documentation, see [RECOMMENDATIONS_IMPLEMENTATION_GUIDE.md](RECOMMENDATIONS_IMPLEMENTATION_GUIDE.md).

## Overview

Four new systems have been implemented to address research recommendations:

1. **Granular Ablation Study** - Identify which features cause performance collapse
2. **Exam Period Awareness** - Improve predictions during exam periods
3. **Student Survey Validation** - Validate WiFi detection accuracy
4. **WiFi-RFID Correlation** - Validate against ground truth data

## Quick Commands

### 1. Run Granular Ablation Study (20-30 min)

Tests 11 feature configurations to identify problematic features:

```bash
python train_granular_ablation_study.py
```

**Output**: `thesis_figures/Granular_Ablation_Analysis.png`

**What it does**: Systematically removes auxiliary features (hour, day, weekend, etc.) to determine which cause the R² drop from 96.62% to 3.38%.

---

### 2. Set Up Exam Period Awareness (5 min)

Configure exam periods for your institution:

```bash
# Create default configuration
python exam_period_tagger.py

# Edit exam_periods_config.json to add your exam dates
# Then train exam-aware model
python train_with_exam_awareness.py
```

**Output**: `thesis_figures/Exam_Aware_Model_Comparison.png`

**What it does**: Trains models that use historical exam period patterns to improve predictions during future exams.

---

### 3. Create Student Survey (10 min setup + 2-4 weeks collection)

Generate survey template and Google Form guide:

```bash
python student_survey_validation.py
```

**Output**:
- `survey_validation_results/survey_template.csv`
- `survey_validation_results/GOOGLE_FORM_SETUP_GUIDE.txt`

**What to do**:
1. Read the guide to create a Google Form
2. Distribute via QR codes at library entrances
3. Collect 200-500 responses over 2-4 weeks
4. Export and analyze

**After collection**:
```bash
python -c "
from student_survey_validation import StudentSurveyValidator
validator = StudentSurveyValidator()
survey_df = validator.load_survey_data('your_responses.csv')
validator.analyze_device_patterns(survey_df)
correction = validator.generate_correction_factor(survey_df)
print(f'Apply correction factor: {correction[\"correction_factor\"]:.4f}')
"
```

---

### 4. Validate with RFID Data (if available)

Requires RFID entry/exit logs from your library system:

```bash
# Generate template
python wifi_rfid_correlation.py

# After obtaining RFID logs, run analysis
python -c "
from wifi_rfid_correlation import WiFiRFIDCorrelation
analyzer = WiFiRFIDCorrelation()
wifi_df = analyzer.load_wifi_data('all_data_cleaned.csv')
rfid_df = analyzer.load_rfid_data('rfid_logs.csv')
merged, results = analyzer.correlate_data(wifi_df, rfid_df)
analyzer.visualize_correlation(merged, results)
print(f'Correlation: {results[\"correlation\"]:.4f}')
print(f'MAE: {results[\"mae\"]:.2f} users')
"
```

**Output**: `wifi_rfid_correlation_results/wifi_rfid_correlation_analysis.png`

---

## For Your Thesis/Paper

### Section 1: Feature Engineering Analysis

**Figure**: `thesis_figures/Granular_Ablation_Analysis.png`

**Key findings to report**:
```python
# After running granular ablation study
import json
with open('granular_ablation_results/granular_ablation_results.json') as f:
    results = json.load(f)

baseline_r2 = results['baseline']['metrics']['r2']
all_features_r2 = results['all_features']['metrics']['r2']
degradation = ((baseline_r2 - all_features_r2) / baseline_r2) * 100

print(f"Baseline R²: {baseline_r2:.4f}")
print(f"All features R²: {all_features_r2:.4f}")
print(f"Performance degradation: {degradation:.1f}%")
```

**Thesis statement template**:
> "Through granular ablation analysis, we identified that [specific feature group] caused the most significant performance degradation ([X]% reduction in R²). The baseline model (sequence only) achieved R² = [baseline], while progressive addition of auxiliary features resulted in degradation up to [max degradation]%, confirming that the 24-hour occupancy sequence contains sufficient temporal information without requiring explicit feature engineering."

---

### Section 2: Context-Aware Predictions

**Figure**: `thesis_figures/Exam_Aware_Model_Comparison.png`

**Key metrics to report**:
```python
# After training exam-aware model
with open('exam_aware_results/exam_aware_comparison.json') as f:
    results = json.load(f)

baseline = results['baseline_model']['overall_metrics']
exam_aware = results['exam_aware_model']['overall_metrics']

print(f"Baseline R²: {baseline['r2']:.4f}")
print(f"Exam-aware R²: {exam_aware['r2']:.4f}")

if results['exam_aware_model']['exam_metrics']:
    exam_specific = results['exam_aware_model']['exam_metrics']
    print(f"Exam-specific R²: {exam_specific['r2']:.4f}")
```

---

### Section 3: Validation Studies

**Figures**:
- `survey_validation_results/device_usage_analysis.png`
- `wifi_rfid_correlation_results/wifi_rfid_correlation_analysis.png`

**Key findings**:
- Average devices per student
- Correction factor
- WiFi-RFID correlation coefficient
- Systematic bias (over/under counting)

---

## Priority Ordering

If time is limited, implement in this order:

1. **Granular Ablation Study** (HIGHEST PRIORITY)
   - Directly addresses the catastrophic performance collapse finding
   - Quick to run (~30 min)
   - Provides immediate thesis content

2. **Exam Period Awareness** (HIGH PRIORITY)
   - Novel contribution
   - Easy to implement
   - Practical value for production deployment

3. **Student Survey** (MEDIUM PRIORITY - but requires time)
   - Important validation
   - Requires 2-4 weeks for data collection
   - Start early if planning to include

4. **RFID Correlation** (LOWER PRIORITY - if data available)
   - Best validation method
   - Only if institution has RFID system
   - Requires coordination with IT department

---

## Common Issues

### Granular Ablation Study

**Issue**: Training takes too long
```bash
# Reduce epochs for faster testing
# Edit train_granular_ablation_study.py, change EPOCHS to 100
```

**Issue**: Out of memory
```bash
# Reduce batch size
# Edit train_granular_ablation_study.py, change BATCH_SIZE to 32
```

---

### Exam Period Tagger

**Issue**: No exam periods found in my data
```python
# Check date range
from exam_period_tagger import ExamPeriodTagger
tagger = ExamPeriodTagger()
print(tagger.exam_periods)

# Add your institution's dates
tagger.add_exam_period(
    name="Your Exam Period",
    start_date="2024-XX-XX",
    end_date="2024-XX-XX",
    exam_type="final",
    semester="spring",
    year=2024
)
```

---

### Student Survey

**Issue**: Low response rate
- Increase incentives (raffle prizes, extra credit)
- Extend collection period
- Add QR codes in high-traffic areas
- Send reminder emails

**Issue**: Many skip MAC address field
- Make it optional (it is by default)
- Analysis works without MACs, just less validation

---

### RFID Correlation

**Issue**: No RFID data available
- Skip this validation
- Use survey validation instead
- Document as future work

**Issue**: Location names don't match
```python
# Create mapping in your code
location_mapping = {
    'Library Building A': 'miguel_pro',
    'Library Building B': 'gisbert_3',
    # etc.
}
rfid_df['Location'] = rfid_df['LocationName'].map(location_mapping)
```

---

## Getting Help

1. **Check the detailed guide**: [RECOMMENDATIONS_IMPLEMENTATION_GUIDE.md](RECOMMENDATIONS_IMPLEMENTATION_GUIDE.md)
2. **Check output logs**: Most scripts print detailed progress
3. **Check result files**: All systems generate JSON files with detailed metrics
4. **Read code comments**: Each file has extensive documentation

---

## File Checklist

After running all systems, you should have:

- [ ] `granular_ablation_results/granular_ablation_results.json`
- [ ] `thesis_figures/Granular_Ablation_Analysis.png`
- [ ] `exam_periods_config.json`
- [ ] `exam_aware_results/exam_aware_comparison.json`
- [ ] `thesis_figures/Exam_Aware_Model_Comparison.png`
- [ ] `survey_validation_results/survey_template.csv`
- [ ] `survey_validation_results/GOOGLE_FORM_SETUP_GUIDE.txt`
- [ ] `survey_validation_results/device_usage_analysis.png` (after survey)
- [ ] `wifi_rfid_correlation_results/wifi_rfid_correlation_analysis.png` (if RFID available)

---

## Next Steps for Production

After completing validation studies:

1. **Apply correction factor** from survey to live predictions
2. **Deploy exam-aware model** instead of baseline
3. **Configure exam calendar** for automatic context switching
4. **Document findings** in thesis/paper
5. **Share improvements** with library administration

---

**Need more details?** See [RECOMMENDATIONS_IMPLEMENTATION_GUIDE.md](RECOMMENDATIONS_IMPLEMENTATION_GUIDE.md)

**Last Updated**: 2026-02-08
