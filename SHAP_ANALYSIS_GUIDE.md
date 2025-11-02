# SHAP Analysis Integration Guide

## Overview

This project now includes comprehensive **SHAP (SHapley Additive exPlanations)** analysis for all trained models. SHAP provides model interpretability by explaining which time steps (features) contribute most to predictions.

## What is SHAP?

SHAP values reveal:
- **Which hours in the 24-hour sequence** are most important for predictions
- **How each time step influences** the prediction (positive or negative)
- **Feature interactions** and patterns across different libraries
- **Model consistency** and interpretability differences

## Files Created

### Core SHAP Analysis Module
- **[shap_analysis.py](shap_analysis.py)** - Main SHAP analysis functions
  - `create_shap_explainer()` - Creates SHAP explainer (GradientExplainer or KernelExplainer)
  - `calculate_shap_values()` - Computes SHAP values for test data
  - `plot_shap_summary()` - Feature importance summary plots
  - `plot_shap_waterfall()` - Individual prediction breakdown
  - `plot_shap_force()` - Force plots showing contributions
  - `plot_shap_dependence()` - Feature interaction plots
  - `get_feature_importance_ranking()` - Extract top features
  - `create_comprehensive_shap_analysis()` - One-stop analysis function

### Visualization Scripts
- **[visualize_shap_comparison.py](visualize_shap_comparison.py)** - Compare SHAP across all models
  - Feature importance heatmaps
  - Average feature importance trends
  - Library-specific comparisons
  - Statistical summaries

### Testing
- **[test_shap_integration.py](test_shap_integration.py)** - Test SHAP with existing models

## How to Use

### 1. Train Models with SHAP Analysis

The training script has been updated to automatically include SHAP analysis:

```bash
python train_multiple_model_types.py
```

This will:
- Train 4 model types (LSTM, CNN, Hybrid, Advanced) for each library
- Run SHAP analysis on each model after training
- Generate individual SHAP plots for each model
- Save results to `model_results/all_model_types_results.json`

### 2. View Individual Model SHAP Results

After training, SHAP plots are saved in `model_results/shap/`:

```
model_results/shap/
├── lstm_only_miguel_pro_summary.png       # Feature importance overview
├── lstm_only_miguel_pro_waterfall.png     # Single prediction breakdown
├── lstm_only_miguel_pro_force.png         # Force plot
├── lstm_only_miguel_pro_dependence.png    # Feature interactions
├── cnn_only_american_corner_summary.png
├── ... (one set per model-library combination)
```

### 3. Generate Comparison Visualizations

Compare SHAP analysis across all models:

```bash
python visualize_shap_comparison.py
```

This creates:
- **feature_importance_heatmap.png** - Heatmaps showing top features per model
- **average_feature_importance.png** - Trends across all libraries
- **library_comparison.png** - Library-specific feature importance
- **feature_importance_summary.png** - Statistical analysis
- **SHAP_ANALYSIS_REPORT.txt** - Text summary with insights

### 4. Test SHAP on Single Model

To test SHAP integration with an existing trained model:

```bash
python test_shap_integration.py
```

## Understanding SHAP Visualizations

### 1. Summary Plot (Bar Chart)
**File**: `*_summary.png` (top panel)

Shows the **mean absolute SHAP value** for each time step:
- **Higher bars** = More important for predictions
- **X-axis**: Mean |SHAP value|
- **Y-axis**: Time steps (t-0 = most recent, t-23 = 23 hours ago)

**Interpretation**:
- Recent hours (t-0 to t-5) typically have highest importance
- Shows which past time windows matter most

### 2. Summary Plot (Dot Plot)
**File**: `*_summary.png` (bottom panel)

Shows SHAP values with feature values:
- **Color**: Red = High feature value, Blue = Low feature value
- **X-axis**: SHAP value (positive = increases prediction, negative = decreases)
- **Each dot**: One sample

**Interpretation**:
- Red dots on right = High occupancy in past → predicts high future occupancy
- Blue dots on left = Low occupancy in past → predicts low future occupancy

### 3. Waterfall Plot
**File**: `*_waterfall.png`

Shows how a **single prediction** is built:
- **Base value**: Average model prediction
- **Each bar**: Contribution of one time step
- **Red bars**: Push prediction higher
- **Blue bars**: Push prediction lower
- **Final value**: Actual prediction

**Interpretation**:
- See exactly why model made a specific prediction
- Identify which hours drove the prediction

### 4. Force Plot
**File**: `*_force.png`

Alternative view of single prediction:
- **Base value** (center) → **Prediction** (end)
- **Red features**: Increase prediction
- **Blue features**: Decrease prediction
- **Width**: Magnitude of contribution

### 5. Dependence Plot
**File**: `*_dependence.png`

Shows relationship between feature value and SHAP value:
- **X-axis**: Feature value (occupancy at specific time)
- **Y-axis**: SHAP value (impact on prediction)
- **Color**: Another feature for interactions

**Interpretation**:
- Positive slope = Higher past occupancy → Higher prediction
- Nonlinear patterns reveal complex relationships

## Key Insights from SHAP Analysis

### Feature Importance Patterns

Based on typical results:

1. **Most Important Time Steps**: t-0 to t-5 (last 5 hours)
   - Immediate past is most predictive
   - Recent trends drive predictions

2. **Secondary Importance**: t-6 to t-12
   - Mid-range history provides context
   - Helps identify daily patterns

3. **Long-term Context**: t-13 to t-23
   - Less important but still used
   - Captures day-of-week patterns

### Model Differences

**LSTM Models**:
- More evenly distributed importance across sequence
- Captures long-term dependencies
- Good at recognizing temporal patterns

**CNN Models**:
- Concentrated importance on recent hours
- Strong local pattern recognition
- Best overall performance

**Hybrid CNN-LSTM**:
- Balanced approach
- Combines local patterns with temporal memory

**Advanced CNN-LSTM**:
- Most complex feature interactions
- Higher model capacity but diminishing returns

## SHAP Results in JSON

Training results include SHAP data:

```json
{
  "model_type": {
    "libraries": {
      "library_id": {
        "shap_analysis": {
          "success": true,
          "feature_importance": {
            "mean_abs_shap": [0.0287, 0.0152, ...],
            "top_features_idx": [23, 22, 18, ...],
            "top_features_importance": [0.0287, 0.0152, ...]
          },
          "plots": {
            "summary": "path/to/summary.png",
            "waterfall": "path/to/waterfall.png",
            "force": "path/to/force.png",
            "dependence": "path/to/dependence.png"
          }
        }
      }
    }
  }
}
```

## Technical Details

### SHAP Explainer Types

The code automatically tries explainers in order:

1. **GradientExplainer** (default)
   - Fast and accurate for neural networks
   - Uses gradients to compute SHAP values
   - Works with TensorFlow 2.x

2. **KernelExplainer** (fallback)
   - Model-agnostic
   - Slower but more reliable
   - Used if GradientExplainer fails

### Performance Considerations

- **Background samples**: 100 (for explainer initialization)
- **Test samples**: 200 (for SHAP value computation)
- **KernelExplainer samples**: 50 (slower, so fewer samples)

These limits balance accuracy with computation time.

### Compatibility

- **TensorFlow**: 2.x (tested with 2.13+)
- **SHAP**: 0.49.1+
- **Python**: 3.8+
- **Models**: All Keras Sequential models

## Troubleshooting

### Issue: SHAP analysis fails with "gradient registry" error

**Solution**: The code automatically falls back to KernelExplainer. If both fail:
1. Check TensorFlow version compatibility
2. Ensure model is properly compiled
3. Try reducing background/test sample sizes

### Issue: SHAP analysis is slow

**Solution**:
1. KernelExplainer is slower - this is expected
2. Reduce `max_samples` in `calculate_shap_values()`
3. Use fewer background samples in `create_shap_explainer()`

### Issue: SHAP plots look strange

**Solution**:
1. Check if model is properly trained (R² > 0.8)
2. Ensure data is normalized/scaled correctly
3. Verify sequence length matches model input

## Example Workflow

```bash
# 1. Install dependencies
pip install shap

# 2. Train models with SHAP analysis
python train_multiple_model_types.py

# 3. View results
# - Individual plots in model_results/shap/
# - JSON results in model_results/all_model_types_results.json

# 4. Generate comparison visualizations
python visualize_shap_comparison.py

# 5. View comparison plots
# - model_results/shap/feature_importance_heatmap.png
# - model_results/shap/average_feature_importance.png
# - model_results/shap/library_comparison.png
# - model_results/shap/feature_importance_summary.png
# - model_results/shap/SHAP_ANALYSIS_REPORT.txt
```

## Benefits of SHAP Analysis

1. **Model Transparency**: Understand why models make predictions
2. **Trust**: Verify models use reasonable patterns
3. **Debugging**: Identify if model learns wrong patterns
4. **Optimization**: Focus on most important time windows
5. **Comparison**: See how different models use features differently
6. **Validation**: Confirm model behavior matches domain knowledge

## Next Steps

- **Feature Engineering**: Use SHAP insights to create better features
- **Model Selection**: Choose models with interpretable patterns
- **Data Collection**: Focus on collecting data for important time windows
- **Real-time Explanations**: Add SHAP to API for live predictions
- **Custom Analysis**: Modify scripts for specific research questions

## References

- [SHAP GitHub](https://github.com/slundberg/shap)
- [SHAP Documentation](https://shap.readthedocs.io/)
- [SHAP Paper](https://arxiv.org/abs/1705.07874)
- [Lundberg & Lee (2017) - A Unified Approach to Interpreting Model Predictions](https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html)

---

**Last Updated**: 2025-10-30
**Version**: 1.0
**Author**: Claude Code + SHAP Integration
