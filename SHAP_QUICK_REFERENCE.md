# SHAP Analysis Quick Reference

## Quick Start

```bash
# Train with SHAP analysis
python train_multiple_model_types.py

# Generate comparison visualizations
python visualize_shap_comparison.py

# Test SHAP integration
python test_shap_integration.py
```

## File Locations

```
model_results/shap/
├── Individual Model Plots:
│   ├── {model_type}_{library}_summary.png      # Feature importance
│   ├── {model_type}_{library}_waterfall.png    # Prediction breakdown
│   ├── {model_type}_{library}_force.png        # Force plot
│   └── {model_type}_{library}_dependence.png   # Feature interactions
│
├── Comparison Plots:
│   ├── feature_importance_heatmap.png          # All models heatmap
│   ├── average_feature_importance.png          # Trends across libraries
│   ├── library_comparison.png                  # Per-library comparison
│   └── feature_importance_summary.png          # Statistical summary
│
└── SHAP_ANALYSIS_REPORT.txt                    # Text summary
```

## Reading SHAP Plots

### Summary Plot (Bar)
```
Higher bars = More important features
X-axis: Mean |SHAP value|
Y-axis: Time steps (t-0 = now, t-23 = 23h ago)
```

### Summary Plot (Dots)
```
Red dots on right = High value → Higher prediction
Blue dots on left = Low value → Lower prediction
Spread = Consistency of feature effect
```

### Waterfall Plot
```
Base value → + Red bars - Blue bars → Final prediction
Shows: How each hour contributes to one prediction
```

### Force Plot
```
Base value ──[Red push up]──[Blue push down]──> Prediction
Width = Contribution strength
```

### Dependence Plot
```
X-axis: Feature value (past occupancy)
Y-axis: SHAP value (impact)
Positive slope = Higher past → Higher prediction
```

## Top Features Interpretation

| Time Step | Hours Ago | Typical Importance | Use Case |
|-----------|-----------|-------------------|----------|
| t-0 to t-5 | 0-5 hours | HIGH | Recent trends, immediate patterns |
| t-6 to t-12 | 6-12 hours | MEDIUM | Daily patterns, business hours |
| t-13 to t-23 | 13-23 hours | LOW | Day-of-week patterns, context |

## Model Patterns

| Model Type | SHAP Pattern | Interpretation |
|------------|--------------|----------------|
| **LSTM Only** | Even distribution | Uses full sequence memory |
| **CNN Only** | Recent focus | Strong local pattern recognition |
| **Hybrid CNN-LSTM** | Balanced | Combines local + temporal |
| **Advanced CNN-LSTM** | Complex interactions | Highest capacity, diminishing returns |

## Key Metrics in JSON

```json
{
  "shap_analysis": {
    "success": true,
    "feature_importance": {
      "top_features_idx": [23, 22, 18, ...],        // Most important time steps
      "top_features_importance": [0.0287, 0.0152, ...], // SHAP values
      "mean_abs_shap": [...]                        // All features
    },
    "plots": { ... }
  }
}
```

## Common Questions

**Q: What does high SHAP value mean?**
A: That feature strongly influences the prediction (positive or negative direction).

**Q: Which time steps are most important?**
A: Usually t-0 to t-5 (last 5 hours). Check summary plots for your specific models.

**Q: Why are some features negative?**
A: Negative SHAP = Feature pushes prediction DOWN (lower occupancy).

**Q: How to compare models?**
A: Run `visualize_shap_comparison.py` to see side-by-side comparisons.

**Q: Can I use SHAP for real-time predictions?**
A: Yes! Calculate SHAP values for live predictions to explain them.

## Code Snippets

### Load SHAP Results
```python
import json

with open('model_results/all_model_types_results.json', 'r') as f:
    results = json.load(f)

# Get SHAP results for specific model
shap_data = results['cnn_only']['libraries']['miguel_pro']['shap_analysis']
top_features = shap_data['feature_importance']['top_features_idx'][:5]
print(f"Top 5 features: {top_features}")
```

### Custom SHAP Analysis
```python
from shap_analysis import create_comprehensive_shap_analysis
from tensorflow.keras.models import load_model

# Load model
model = load_model('saved_models/cnn_only_miguel_pro_model.keras')

# Run SHAP analysis
results = create_comprehensive_shap_analysis(
    model=model,
    X_train=X_train,
    X_test=X_test,
    model_name="CNN Only",
    location_name="Miguel Pro",
    output_dir='custom_shap_output'
)
```

### Extract Top Features
```python
from shap_analysis import get_feature_importance_ranking

# After computing shap_values
feature_ranking = get_feature_importance_ranking(shap_values)

print("Top 10 features:")
for idx, importance in zip(
    feature_ranking['top_features_idx'][:10],
    feature_ranking['top_features_importance'][:10]
):
    print(f"  t-{idx}: {importance:.4f}")
```

## Performance Tuning

| Parameter | Location | Effect | Recommended |
|-----------|----------|--------|-------------|
| `background_size` | `create_shap_explainer()` | Explainer accuracy vs speed | 50-100 |
| `max_samples` | `calculate_shap_values()` | Analysis samples | 100-200 |
| `nsamples` | KernelExplainer | KernelShap iterations | 50-100 |

Reduce these values if SHAP analysis is too slow.

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "gradient registry" error | Code auto-falls back to KernelExplainer |
| Slow computation | Reduce sample sizes (see Performance Tuning) |
| Strange plots | Check model R² > 0.8, verify data scaling |
| Import errors | `pip install shap` |

## Next Actions

After reviewing SHAP results:

1. **Validate patterns**: Do top features make domain sense?
2. **Model selection**: Choose models with interpretable patterns
3. **Feature engineering**: Create features based on important time windows
4. **Data quality**: Ensure high-quality data for important time steps
5. **Real-time explanations**: Add SHAP to API endpoints

---

**For detailed explanation, see**: [SHAP_ANALYSIS_GUIDE.md](SHAP_ANALYSIS_GUIDE.md)
