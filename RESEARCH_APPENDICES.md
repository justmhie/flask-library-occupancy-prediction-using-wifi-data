# Research Appendix: Model Validation and Ablation Study

This document contains the consolidated results of the Granular Ablation Study and the Multi-Device Calibration Study for the Library Occupancy Prediction System.

---

## Appendix A: Granular Ablation Study Results

The purpose of this study was to identify the performance impact of auxiliary temporal features compared to a baseline model using only 24-hour occupancy sequences.

### Table A.1: Model Performance by Feature Configuration

| Configuration | Auxiliary Features Included | $R^2$ Score | RMSE | MAE |
| :--- | :--- | :---: | :---: | :---: |
| **Baseline** | **None (Sequence Only)** | **0.9612** | **9.18** | **4.11** |
| Hour Only | hour\_sin, hour\_cos | 0.9506 | 10.36 | 5.96 |
| Part of Day | is\_morning, afternoon, evening, night | 0.9444 | 10.98 | 5.41 |
| Day Only | day\_sin, day\_cos | 0.9372 | 11.67 | 6.00 |
| Activity Periods | is\_peak\_hours, is\_open\_hours | 0.9357 | 11.81 | 6.89 |
| Hour + Weekend | hour\_sin/cos, is\_weekend | 0.9304 | 12.30 | 6.57 |
| Hour + Day | hour\_sin/cos, day\_sin/cos | 0.9284 | 12.46 | 7.00 |
| Day + Weekend | day\_sin/cos, is\_weekend | 0.9213 | 13.07 | 7.36 |
| Week Patterns | is\_weekday, week\_of\_year | 0.9169 | 13.43 | 6.62 |
| Weekend Only | is\_weekend | 0.9133 | 13.72 | 6.36 |
| **Full Model** | **All 13 Features** | **0.9605** | **9.26** | **4.63** |

### Key Findings:
- **Redundancy Analysis**: The baseline sequence-only model outperforms the full model, suggesting that 24-hour historical occupancy already captures the necessary temporal and cyclical patterns.
- **Worst Case**: The `is_weekend` indicator caused the highest single-feature degradation (~4.8%), likely introducing bias that conflicted with the sequence data.

---

## Appendix B: Multi-Device Validation and Calibration

To account for students using multiple WiFi devices (e.g., laptop and smartphone simultaneously), a calibration study was conducted by correlating WiFi MAC counts with RFID entry/exit ground truth data and Student Survey responses.

### Table B.1: Device-to-Student Ratio Comparison

| Source | Metric | Value |
| :--- | :--- | :---: |
| **Student Survey** | Average Devices per Student | 1.45 |
| **WiFi / RFID Correlation** | Implied Devices per Student | 1.50 |
| **Systematic Bias** | Detection Status | Over-counting |

### Calibration Formula

Based on the average of observed and reported multi-device usage, the following correction factor was derived to arrive at the predicted student count (Ground Truth approximation):

$$Adjusted\ Occupancy = WiFi\ MAC\ Count \times 0.6769$$

### Data Summary:
- **WiFi Records Processed**: 579,333
- **RFID Records Processed**: 29,604
- **Survey Sample Size**: 200 responses
- **Correlation ($r$ )**: High temporal alignment observed across peak hours.

---

## Appendix C: Exam Period Awareness Logic

The system incorporates specific logic for academic exam periods, utilizing a 7-day pre-exam buffer.

- **Pre-Exam Period**: 7 days prior to scheduled exam dates.
- **Pattern Matching**: On days tagged as `Exam Day`, the model bypasses standard day-of-week averages and switches to a global "Exam Period" pattern matched across all historical data tagged with the same status.
