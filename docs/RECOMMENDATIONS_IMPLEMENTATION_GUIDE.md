# Recommendations Implementation Guide

This guide documents the implementation of all recommendations for improving the library occupancy prediction system, including granular ablation studies, exam period awareness, and validation mechanisms.

## Table of Contents

1. [Granular Ablation Study](#1-granular-ablation-study)
2. [Exam Period Tagging System](#2-exam-period-tagging-system)
3. [Student Survey Validation](#3-student-survey-validation)
4. [WiFi vs RFID Correlation](#4-wifi-vs-rfid-correlation)
5. [Usage Examples](#5-usage-examples)
6. [Results Interpretation](#6-results-interpretation)

---

## 1. Granular Ablation Study

### Purpose
The original ablation study found catastrophic performance collapse when ALL auxiliary features were included. The granular study systematically removes individual feature groups to identify which specific features cause the degradation.

### Implementation

**File**: `scripts/train_granular_ablation_study.py`

**Feature Groups Tested**:
- `baseline`: Sequence only (no auxiliary features)
- `hour_only`: Cyclical hour encoding (sin/cos)
- `day_only`: Cyclical day-of-week encoding (sin/cos)
- `weekend_only`: Weekend indicator flag
- `part_of_day`: Morning/afternoon/evening/night indicators
- `week_patterns`: Weekday flag + week of year
- `activity_periods`: Peak hours + open hours flags
- `hour_day`: Hour + day of week (common combination)
- `hour_weekend`: Hour + weekend flag
- `day_weekend`: Day of week + weekend flag
- `all_features`: All 13 auxiliary features

### Running the Study

```bash
python scripts/train_granular_ablation_study.py
```

### Output Files

- **`granular_ablation_results/granular_ablation_results.json`**: Complete metrics for all configurations
- **`thesis_figures/Granular_Ablation_Analysis.png`**: Comprehensive visualization showing:
  - R² scores by configuration
  - RMSE and MAE comparisons
  - Feature count vs performance scatter plot
  - Performance degradation table

### Key Findings

The study reveals:
1. Which individual features hurt performance the most
2. Whether feature combinations amplify or mitigate problems
3. Quantified percentage point degradation for each feature group
4. Optimal feature subset (if any provide improvement)

### Expected Results

Based on the original study showing the baseline (sequence only) performing best at 96.62% R², we expect:
- **Baseline configuration** to achieve highest R²
- **Incremental degradation** as features are added
- **Identification** of the most problematic feature groups
- **Confirmation** that simpler is better for this use case

---

## 2. Exam Period Tagging System

### Purpose
Library occupancy patterns differ significantly during exam periods. This system:
- Tags historical data with exam period indicators
- Enables future predictions to use exam-specific patterns
- Improves accuracy during high-stress academic periods

### Implementation

**Files**:
- `exam_period_tagger.py`: Core tagging system
- `scripts/train_with_exam_awareness.py`: Exam-aware model training
- `exam_periods_config.json`: Exam period configuration (auto-generated)

### Features

1. **Historical Tagging**: Marks past dates as exam/pre-exam periods
2. **Future Awareness**: Identifies upcoming exams for predictions
3. **Configurable Periods**: Easy JSON-based configuration
4. **Pre-Exam Buffer**: Tags one week before exams as preparation period

### Setup

```python
from exam_period_tagger import ExamPeriodTagger

# Initialize (creates default config on first run)
tagger = ExamPeriodTagger()

# Load your data
df = pd.read_csv('all_data_cleaned.csv')

# Tag exam periods
tagged_df = tagger.tag_dataframe(df, date_column='Start_dt')

# Check future exams
future_exams = tagger.get_future_exam_dates(days_ahead=90)
```

### Adding Custom Exam Periods

Edit `exam_periods_config.json`:

```json
{
  "exam_periods": [
    {
      "name": "Finals Spring 2026",
      "start_date": "2026-05-04",
      "end_date": "2026-05-15",
      "type": "final",
      "semester": "spring",
      "year": 2026
    }
  ],
  "pre_exam_buffer_days": 7
}
```

Or programmatically:

```python
tagger.add_exam_period(
    name="Finals Fall 2026",
    start_date="2026-12-01",
    end_date="2026-12-12",
    exam_type="final",
    semester="fall",
    year=2026
)
```

### Training Exam-Aware Models

```bash
python scripts/train_with_exam_awareness.py
```

**Outputs**:
- **Baseline model**: Standard CNN-LSTM (no exam awareness)
- **Exam-aware model**: CNN-LSTM + exam context features
- **Comparative analysis**: Performance during exam vs regular periods
- **Visualization**: `thesis_figures/Exam_Aware_Model_Comparison.png`

### Model Architecture

The exam-aware model uses:
- **Sequence input**: 24-hour occupancy sequence (same as baseline)
- **Exam context**: `[is_exam_period, is_pre_exam_period]` binary flags
- **Architecture**: CNN layers → BiLSTM → Attention → Concatenate with exam features → Dense layers

### Expected Improvements

- **Overall**: Marginal improvement (exam periods are minority of data)
- **During exams**: Significant improvement in prediction accuracy
- **Pre-exam periods**: Better captures ramp-up in library usage
- **Pattern switching**: Model learns different weights for exam contexts

### Prediction Usage

When predicting for a date:

1. Check if date is exam period: `tagger.is_exam_period(prediction_date)`
2. Set exam flags accordingly: `[1, 0]` for exam, `[0, 1]` for pre-exam, `[0, 0]` for regular
3. Pass to model: `model.predict([sequence, exam_flags])`

---

## 3. Student Survey Validation

### Purpose
Validates WiFi-based occupancy detection against student self-reports to:
- Confirm WiFi connection rates
- Understand multi-device usage patterns
- Calculate correction factors for device count inflation

### Implementation

**File**: `scripts/student_survey_validation.py`

### Components

1. **Survey Template Generator**: Creates CSV template for manual surveys
2. **Google Form Setup Guide**: Step-by-step instructions for online distribution
3. **Device Pattern Analyzer**: Analyzes multi-device usage
4. **Correction Factor Calculator**: Computes occupancy adjustment factor
5. **WiFi Log Validator**: Matches survey responses with actual WiFi logs

### Setup

```bash
python scripts/student_survey_validation.py
```

This generates:
- **`survey_validation_results/survey_template.csv`**: Template for collecting responses
- **`survey_validation_results/GOOGLE_FORM_SETUP_GUIDE.txt`**: Complete guide for creating Google Form

### Survey Fields

| Field | Description |
|-------|-------------|
| ResponseID | Unique identifier (auto-generated) |
| Date | Visit date (YYYY-MM-DD) |
| StudentID | Anonymous student code |
| VisitedLibrary | Yes/No |
| LibraryLocation | Which library |
| ArrivalTime | Entry time (HH:MM) |
| DepartureTime | Exit time (HH:MM) |
| ConnectedToWiFi | Yes/No |
| NumberOfDevices | Count of connected devices |
| DeviceTypes | Types (comma-separated) |
| PrimaryDeviceMAC | Optional MAC for validation |
| ReasonForNotConnecting | If didn't connect, why? |
| UsagePurpose | Study/Research/Social/Other |
| Notes | Additional comments |

### Distribution Strategies

1. **QR Codes**: Place at library entrances/exits
2. **Email**: Send to student mailing lists
3. **Social Media**: Post on university platforms
4. **In-Person**: Table at library entrance
5. **Incentives**: Prize raffle, extra credit, library vouchers

### Target Collection

- **Duration**: 2-4 weeks during regular semester
- **Responses**: 200-500 students
- **Coverage**: Both exam and non-exam periods
- **Locations**: All library locations

### Analysis

```python
from scripts.student_survey_validation import StudentSurveyValidator

validator = StudentSurveyValidator()

# Load survey responses (from Google Form export)
survey_df = validator.load_survey_data('student_survey_responses.csv')

# Analyze device patterns
device_analysis = validator.analyze_device_patterns(survey_df)

# Calculate correction factor
correction = validator.generate_correction_factor(survey_df)

# Validate against WiFi logs (optional, if MACs provided)
wifi_df = pd.read_csv('all_data_cleaned.csv')
validation_results = validator.validate_with_wifi_logs(survey_df, wifi_df)
```

### Key Metrics

1. **Average Devices per Student**: How many devices does a typical student connect?
2. **Device Type Distribution**: Laptop vs phone vs tablet usage
3. **WiFi Connection Rate**: What percentage of visitors connect?
4. **Correction Factor**: `1 / avg_devices_per_student`

### Example Results

If survey finds:
- Average devices per student: 1.8
- Correction factor: 1/1.8 = 0.556

Then:
- WiFi detects 150 unique MACs
- Estimated actual students: 150 × 0.556 = 83 students

### Output Visualizations

- **`survey_validation_results/device_usage_analysis.png`**:
  - Device count distribution bar chart
  - Device types usage horizontal bar chart

---

## 4. WiFi vs RFID Correlation

### Purpose
Validates WiFi-based occupancy against RFID entry/exit logs (ground truth) to:
- Assess detection accuracy
- Identify systematic biases (over/under counting)
- Calculate correlation strength
- Determine if calibration is needed

### Implementation

**File**: `scripts/wifi_rfid_correlation.py`

### Requirements

You need:
1. **WiFi logs**: Your existing `all_data_cleaned.csv`
2. **RFID logs**: Entry/exit records from library access control system

### RFID Data Format

**Required columns**:
- `Timestamp`: Date/time of entry or exit (YYYY-MM-DD HH:MM:SS)
- `StudentID`: Anonymous student/card identifier
- `Location`: Library location code (must match WiFi location names)
- `Action`: "Entry" or "Exit"

**Template generation**:

```bash
python scripts/wifi_rfid_correlation.py
```

This creates: `wifi_rfid_correlation_results/rfid_data_template.csv`

### Running Analysis

```python
from scripts.wifi_rfid_correlation import WiFiRFIDCorrelation

analyzer = WiFiRFIDCorrelation()

# Load WiFi data
wifi_df = analyzer.load_wifi_data(
    'all_data_cleaned.csv',
    location_column='Location',
    datetime_column='Start_dt'
)

# Load RFID data
rfid_df = analyzer.load_rfid_data(
    'rfid_logs.csv',
    location_column='Location',
    datetime_column='Timestamp'
)

# Correlate
merged_df, results = analyzer.correlate_data(wifi_df, rfid_df)

# Visualize
analyzer.visualize_correlation(merged_df, results)

# Generate report
report = analyzer.generate_report(merged_df, results)
```

### Output Files

1. **`wifi_rfid_correlation_results/wifi_rfid_correlation_analysis.png`**:
   - Scatter plot with regression line
   - Correlation metrics summary
   - Difference distribution histogram
   - Absolute error distribution
   - Box plot comparison
   - Time series comparison

2. **`wifi_rfid_correlation_results/correlation_report.txt`**:
   - Detailed statistical analysis
   - Detection bias analysis
   - Recommendations based on findings

3. **`wifi_rfid_correlation_results/merged_correlation_data.csv`**:
   - Merged WiFi and RFID data for further analysis

### Key Metrics

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| **Pearson r** | Linear correlation (-1 to 1) | > 0.9: Excellent, > 0.7: Good, > 0.5: Moderate |
| **R² Score** | Explained variance (0 to 1) | > 0.8: Strong predictive power |
| **MAE** | Mean Absolute Error | Average user count difference |
| **RMSE** | Root Mean Square Error | Penalizes large errors |
| **Slope** | Regression slope | > 1: Over-count, < 1: Under-count |
| **p-value** | Statistical significance | < 0.05: Statistically significant |

### Detection Bias

**Over-detection** (WiFi > RFID):
- Causes: Multiple devices per student, MAC randomization, signal bleed
- Solution: Apply device correction factor from survey

**Under-detection** (WiFi < RFID):
- Causes: Students not connecting, airplane mode, weak signal
- Solution: Improve WiFi coverage, encourage connection

### Calibration

If systematic bias detected:

```python
# Example: WiFi consistently over-counts by 25%
calibration_factor = 1 / 1.25  # 0.8

# Apply to predictions
calibrated_occupancy = wifi_occupancy * calibration_factor
```

---

## 5. Usage Examples

### Complete Workflow

```python
# 1. Run granular ablation study
import subprocess
subprocess.run(['python', 'scripts/train_granular_ablation_study.py'])

# 2. Set up exam period tagging
from exam_period_tagger import ExamPeriodTagger
tagger = ExamPeriodTagger()

# Add your institution's exam periods
tagger.add_exam_period(
    name="Finals Spring 2026",
    start_date="2026-05-04",
    end_date="2026-05-15",
    exam_type="final",
    semester="spring",
    year=2026
)

# 3. Train exam-aware model
subprocess.run(['python', 'scripts/train_with_exam_awareness.py'])

# 4. Deploy student survey
from scripts.student_survey_validation import StudentSurveyValidator
validator = StudentSurveyValidator()
validator.create_survey_template()
validator.create_google_form_guide()
# ... distribute survey and collect responses ...

# 5. Analyze survey results
survey_df = validator.load_survey_data('survey_responses.csv')
device_stats = validator.analyze_device_patterns(survey_df)
correction = validator.generate_correction_factor(survey_df)

# 6. Validate with RFID (if available)
from scripts.wifi_rfid_correlation import WiFiRFIDCorrelation
analyzer = WiFiRFIDCorrelation()
wifi_df = analyzer.load_wifi_data('all_data_cleaned.csv')
rfid_df = analyzer.load_rfid_data('rfid_logs.csv')
merged, results = analyzer.correlate_data(wifi_df, rfid_df)
analyzer.visualize_correlation(merged, results)

# 7. Apply corrections to production predictions
correction_factor = correction['correction_factor']  # From survey
calibration_factor = results['slope']  # From RFID correlation

final_occupancy = raw_wifi_count * correction_factor * calibration_factor
```

### Prediction with Exam Awareness

```python
import numpy as np
from tensorflow.keras.models import load_model
from exam_period_tagger import ExamPeriodTagger

# Load exam-aware model
model = load_model('exam_aware_results/exam_aware_model.keras')
tagger = ExamPeriodTagger()

# Prepare sequence (last 24 hours)
sequence = get_last_24_hours_occupancy()  # Your function
sequence_scaled = scaler.transform(sequence.reshape(-1, 1))
sequence_input = sequence_scaled.reshape(1, 24, 1)

# Determine exam context for prediction date
prediction_date = '2026-05-10'
is_exam, exam_info = tagger.is_exam_period(prediction_date)
is_pre_exam, _ = tagger.is_pre_exam_period(prediction_date)

exam_features = np.array([[int(is_exam), int(is_pre_exam)]])

# Predict
prediction_scaled = model.predict([sequence_input, exam_features])
prediction = scaler.inverse_transform(prediction_scaled)

print(f"Predicted occupancy for {prediction_date}: {prediction[0][0]:.0f} users")
if is_exam:
    print(f"⚠ Exam period: {exam_info['name']}")
```

---

## 6. Results Interpretation

### Granular Ablation Study

**What to look for**:
1. Which feature configuration achieves highest R²?
2. Is there a "sweet spot" with minimal features?
3. Which individual features cause the most degradation?
4. Do feature combinations amplify problems?

**Thesis implications**:
- If baseline is best: Argue for model simplicity and implicit learning
- If some features help: Document optimal feature subset
- Report percentage point degradation per feature group

**Expected outcome**:
Based on original study, baseline (sequence only) should dominate. Document this as validation of the "less is more" principle for this dataset.

### Exam-Aware Model

**What to look for**:
1. Overall metrics (may show minimal improvement due to data imbalance)
2. **Exam-specific metrics** (key improvement area)
3. Pre-exam period detection
4. Prediction stability during exam transitions

**Thesis implications**:
- Context-aware predictions improve during high-stakes periods
- Model demonstrates adaptability to different academic contexts
- Practical value for library resource planning during exams

**Deployment strategy**:
- Use exam-aware model in production
- Query exam calendar for each prediction
- Provide "exam mode" indicators in dashboard

### Student Survey

**What to look for**:
1. Average devices per student (typically 1.5-2.5)
2. Device type distribution (laptops vs phones)
3. WiFi connection rate (ideally > 80%)
4. Validation match rate with actual WiFi logs

**Thesis implications**:
- Multi-device usage justifies correction factors
- Survey validates WiFi as proxy for occupancy
- Device heterogeneity impacts counting accuracy

**Action items**:
- If avg devices > 2.0: Apply strong correction (divide by 2)
- If connection rate < 70%: WiFi may undercount significantly
- If validation matches < 80%: Investigate discrepancies

### WiFi vs RFID Correlation

**What to look for**:
1. Pearson r > 0.8: Excellent correlation (WiFi is reliable)
2. Slope near 1.0: Well-calibrated
3. Low MAE/RMSE: Accurate absolute counts
4. Systematic bias direction (over vs under)

**Thesis implications**:
- High correlation validates WiFi-based approach
- Quantified accuracy vs ground truth
- Documented limitations and systematic biases

**Calibration decisions**:
- r > 0.9, slope ≈ 1.0: No calibration needed
- r > 0.8, slope ≠ 1.0: Apply slope correction
- r < 0.7: Consider hybrid approach or WiFi improvements

---

## Thesis/Paper Writing Guide

### Recommended Sections

#### 1. Granular Ablation Study

**Section**: "Feature Engineering Analysis"

> "To identify which auxiliary temporal features contributed to the performance collapse observed in our initial ablation study, we conducted a granular analysis systematically testing 11 feature configurations. The baseline model (sequence only) achieved R² = 96.62%, while progressive addition of features resulted in degradation up to [X]%. Individual feature groups showed degradation ranging from [min]% to [max]%, with [specific feature group] causing the most significant performance loss of [Y]%. This analysis confirmed that the 24-hour occupancy sequence contains sufficient temporal information, and auxiliary features introduce redundancy leading to overfitting."

**Figures**:
- Figure X: Granular Ablation Analysis (comprehensive 6-panel visualization)
- Table Y: Feature Configuration Performance Summary

#### 2. Exam Period Awareness

**Section**: "Context-Aware Predictions"

> "Library occupancy patterns exhibit significant variation during examination periods due to increased study time and altered daily routines. We developed an exam-aware model incorporating binary indicators for exam and pre-exam periods. While overall performance showed modest improvement ([baseline R²] to [exam-aware R²]), exam-specific predictions improved substantially (R² = [exam-specific]). The model successfully adapted to context-dependent patterns, with pre-exam period detection capturing the gradual ramp-up in library usage beginning approximately one week before examinations."

**Figures**:
- Figure Z: Exam-Aware Model Performance Comparison
- Table W: Exam Period vs Regular Period Prediction Accuracy

#### 3. Validation Studies

**Section**: "System Validation and Correction Factors"

> "To validate our WiFi-based occupancy detection, we conducted two parallel validation studies. A student survey (N=[responses]) revealed an average of [X] devices per student, yielding a correction factor of [Y] for device count inflation. WiFi-RFID correlation analysis (N=[hours]) demonstrated strong correlation (r = [Z], p < 0.001), validating WiFi as a reliable proxy for occupancy. However, systematic bias analysis revealed [over/under]-counting by an average of [W] users, attributed to [reasons]. Combined correction and calibration factors improved absolute occupancy estimates by [improvement]%."

**Figures**:
- Figure A: Multi-Device Usage Patterns
- Figure B: WiFi-RFID Correlation Analysis
- Table C: Validation Metrics Summary

---

## Troubleshooting

### Granular Ablation Study

**Issue**: Training too slow
- **Solution**: Reduce `EPOCHS` to 100 or use `model='haiku'` for faster training

**Issue**: All configurations show poor performance
- **Solution**: Check data preprocessing, ensure occupancy values are properly scaled

### Exam Period Tagger

**Issue**: No exam periods detected in data
- **Solution**: Check date ranges in `exam_periods_config.json` match your data

**Issue**: Future predictions not using exam context
- **Solution**: Ensure exam flag is passed to model: `model.predict([sequence, exam_flags])`

### Student Survey

**Issue**: Low response rate
- **Solution**: Increase incentives, extend collection period, improve visibility

**Issue**: Many incomplete responses
- **Solution**: Simplify form, add progress bar, enable save-and-resume

### WiFi-RFID Correlation

**Issue**: No overlapping data found
- **Solution**: Check datetime formats, location name matching, timezone consistency

**Issue**: Very low correlation
- **Solution**: Verify RFID logs are complete, check for data quality issues

---

## Dependencies

All new features use existing project dependencies:
- pandas
- numpy
- tensorflow/keras
- scikit-learn
- matplotlib
- seaborn
- scipy (for WiFi-RFID correlation)

No additional installations required if you already have the project running.

---

## File Summary

| File | Purpose | Output |
|------|---------|--------|
| `scripts/train_granular_ablation_study.py` | Test individual feature impact | JSON results, visualization |
| `exam_period_tagger.py` | Tag exam periods | Tagged dataframe, JSON config |
| `scripts/train_with_exam_awareness.py` | Train exam-aware models | Models, comparison plots |
| `scripts/student_survey_validation.py` | Survey analysis | Templates, guides, visualizations |
| `scripts/wifi_rfid_correlation.py` | Validate vs RFID | Correlation plots, report |
| `RECOMMENDATIONS_IMPLEMENTATION_GUIDE.md` | This guide | - |

---

## Next Steps

1. **Run granular ablation study** to identify problematic features
2. **Configure exam periods** for your institution's calendar
3. **Deploy student survey** using provided Google Form guide
4. **Obtain RFID logs** (if available) for correlation analysis
5. **Analyze results** and document in thesis/paper
6. **Apply corrections** to production predictions

---

## Contact and Support

For questions or issues:
1. Check this guide first
2. Review code comments in individual files
3. Examine output files and error messages
4. Consult project README.md for general setup

---

**Last Updated**: 2026-02-08

**Version**: 1.0
