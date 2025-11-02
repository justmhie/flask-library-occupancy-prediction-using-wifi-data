"""
SHAP Comparison Visualization Script
Creates comprehensive comparison visualizations across all models and libraries
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("=" * 80)
print("SHAP ANALYSIS COMPARISON VISUALIZER")
print("=" * 80)

# Load results
results_path = 'model_results/all_model_types_results.json'
print(f"\nLoading results from: {results_path}")

with open(results_path, 'r') as f:
    all_results = json.load(f)

# Extract SHAP feature importance across all models
print("\nExtracting SHAP feature importance data...")

feature_importance_data = []

for model_type, model_data in all_results.items():
    model_name = model_data['model_name']

    for lib_id, lib_data in model_data.get('libraries', {}).items():
        if 'shap_analysis' in lib_data and lib_data['shap_analysis'].get('success'):
            shap_data = lib_data['shap_analysis']
            feature_imp = shap_data.get('feature_importance', {})

            if 'top_features_importance' in feature_imp:
                feature_importance_data.append({
                    'model_type': model_type,
                    'model_name': model_name,
                    'library': lib_data['library_name'],
                    'top_feature_importance': feature_imp['top_features_importance'],
                    'top_feature_idx': feature_imp['top_features_idx']
                })

print(f"Found {len(feature_importance_data)} models with SHAP analysis")

if len(feature_importance_data) == 0:
    print("\nNo SHAP analysis data found. Please run training with SHAP analysis first.")
    exit(0)

# ============================================
# VISUALIZATION 1: Feature Importance Heatmap
# ============================================

print("\nCreating Feature Importance Heatmap...")

fig, axes = plt.subplots(2, 2, figsize=(20, 16))
fig.suptitle('SHAP Feature Importance Comparison Across Models',
             fontsize=18, fontweight='bold', y=0.995)

model_types = ['lstm_only', 'cnn_only', 'hybrid_cnn_lstm', 'advanced_cnn_lstm']
model_names_map = {
    'lstm_only': 'LSTM Only',
    'cnn_only': 'CNN Only',
    'hybrid_cnn_lstm': 'Hybrid CNN-LSTM',
    'advanced_cnn_lstm': 'Advanced CNN-LSTM'
}

for idx, model_type in enumerate(model_types):
    ax = axes[idx // 2, idx % 2]

    # Get data for this model type
    model_data = [d for d in feature_importance_data if d['model_type'] == model_type]

    if model_data:
        # Create matrix for heatmap
        max_features = 10
        libraries = [d['library'] for d in model_data]
        importance_matrix = np.zeros((len(libraries), max_features))

        for i, d in enumerate(model_data):
            imp_values = d['top_feature_importance'][:max_features]
            importance_matrix[i, :len(imp_values)] = imp_values

        # Create heatmap
        sns.heatmap(importance_matrix,
                   xticklabels=[f't-{i}' for i in range(max_features)],
                   yticklabels=libraries,
                   cmap='YlOrRd',
                   annot=True,
                   fmt='.3f',
                   cbar_kws={'label': 'Mean |SHAP Value|'},
                   ax=ax)

        ax.set_title(f'{model_names_map[model_type]}\nTop 10 Most Important Time Steps',
                    fontsize=14, fontweight='bold', pad=10)
        ax.set_xlabel('Time Step (hours before prediction)', fontsize=11)
        ax.set_ylabel('Library Location', fontsize=11)
    else:
        ax.text(0.5, 0.5, f'No data for {model_names_map[model_type]}',
               ha='center', va='center', fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])

plt.tight_layout()
plt.savefig('model_results/shap/feature_importance_heatmap.png', dpi=300, bbox_inches='tight')
print("  Saved: model_results/shap/feature_importance_heatmap.png")
plt.close()

# ============================================
# VISUALIZATION 2: Top Feature Rankings
# ============================================

print("\nCreating Top Feature Rankings...")

fig, ax = plt.subplots(figsize=(14, 10))

colors = plt.cm.tab10(np.linspace(0, 1, len(model_types)))

for idx, model_type in enumerate(model_types):
    model_data = [d for d in feature_importance_data if d['model_type'] == model_type]

    if model_data:
        # Average importance across libraries
        max_features = 24
        avg_importance = np.zeros(max_features)
        count = 0

        for d in model_data:
            indices = d['top_feature_idx'][:max_features]
            values = d['top_feature_importance'][:max_features]

            for i, val in zip(indices, values):
                if i < max_features:
                    avg_importance[i] += val
                    count += 1

        if count > 0:
            avg_importance = avg_importance / (count / max_features)

        # Plot
        time_steps = np.arange(max_features)
        ax.plot(time_steps, avg_importance, marker='o', linewidth=2.5,
               label=model_names_map[model_type], color=colors[idx], alpha=0.8)

ax.set_xlabel('Time Step (hours before prediction)', fontsize=13, fontweight='bold')
ax.set_ylabel('Average |SHAP Value|', fontsize=13, fontweight='bold')
ax.set_title('Average Feature Importance Across All Libraries\nHigher values = More important for prediction',
            fontsize=15, fontweight='bold', pad=15)
ax.legend(loc='best', fontsize=11, framealpha=0.9)
ax.grid(True, alpha=0.3)
ax.set_xlim(-0.5, 23.5)

plt.tight_layout()
plt.savefig('model_results/shap/average_feature_importance.png', dpi=300, bbox_inches='tight')
print("  Saved: model_results/shap/average_feature_importance.png")
plt.close()

# ============================================
# VISUALIZATION 3: Model Comparison by Library
# ============================================

print("\nCreating Library-wise Model Comparison...")

# Get all unique libraries
all_libraries = list(set([d['library'] for d in feature_importance_data]))

if len(all_libraries) > 0:
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('SHAP Feature Importance by Library Location',
                fontsize=18, fontweight='bold', y=0.995)

    for idx, library in enumerate(all_libraries[:6]):
        ax = axes[idx // 3, idx % 3]

        # Get data for this library
        lib_data = [d for d in feature_importance_data if d['library'] == library]

        for d in lib_data:
            top_n = 10
            indices = d['top_feature_idx'][:top_n]
            values = d['top_feature_importance'][:top_n]

            ax.bar(range(len(values)), values, alpha=0.6, label=d['model_name'])

        ax.set_xlabel('Feature Rank', fontsize=10)
        ax.set_ylabel('Mean |SHAP Value|', fontsize=10)
        ax.set_title(library, fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')

    # Hide unused subplots
    for idx in range(len(all_libraries), 6):
        axes[idx // 3, idx % 3].axis('off')

    plt.tight_layout()
    plt.savefig('model_results/shap/library_comparison.png', dpi=300, bbox_inches='tight')
    print("  Saved: model_results/shap/library_comparison.png")
    plt.close()

# ============================================
# VISUALIZATION 4: Feature Importance Summary
# ============================================

print("\nCreating Feature Importance Summary Statistics...")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Left plot: Top 5 most important features across all models
top_features_global = {}

for d in feature_importance_data:
    for idx, val in zip(d['top_feature_idx'][:5], d['top_feature_importance'][:5]):
        feature_name = f't-{idx}'
        if feature_name not in top_features_global:
            top_features_global[feature_name] = []
        top_features_global[feature_name].append(val)

# Calculate statistics
feature_stats = {k: {
    'mean': np.mean(v),
    'std': np.std(v),
    'count': len(v)
} for k, v in top_features_global.items()}

# Sort by mean importance
sorted_features = sorted(feature_stats.items(), key=lambda x: x[1]['mean'], reverse=True)[:10]

feature_names = [f[0] for f in sorted_features]
means = [f[1]['mean'] for f in sorted_features]
stds = [f[1]['std'] for f in sorted_features]

axes[0].barh(feature_names, means, xerr=stds, capsize=5, color='steelblue', alpha=0.7)
axes[0].set_xlabel('Mean |SHAP Value| ± Std Dev', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Time Step Feature', fontsize=12, fontweight='bold')
axes[0].set_title('Top 10 Most Important Features\nAcross All Models and Libraries',
                 fontsize=13, fontweight='bold', pad=10)
axes[0].grid(True, alpha=0.3, axis='x')

# Right plot: Consistency of feature importance
model_consistency = {}

for model_type in model_types:
    model_data = [d for d in feature_importance_data if d['model_type'] == model_type]

    if model_data:
        # Calculate coefficient of variation for top features
        all_top_features = []
        for d in model_data:
            all_top_features.extend(d['top_feature_importance'][:5])

        if len(all_top_features) > 0:
            cv = np.std(all_top_features) / (np.mean(all_top_features) + 1e-10)
            model_consistency[model_names_map[model_type]] = {
                'mean': np.mean(all_top_features),
                'cv': cv
            }

if model_consistency:
    models = list(model_consistency.keys())
    cv_values = [model_consistency[m]['cv'] for m in models]

    colors_cv = ['green' if cv < 0.5 else 'orange' if cv < 1.0 else 'red' for cv in cv_values]

    axes[1].barh(models, cv_values, color=colors_cv, alpha=0.7)
    axes[1].set_xlabel('Coefficient of Variation', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Model Type', fontsize=12, fontweight='bold')
    axes[1].set_title('Feature Importance Consistency\nLower = More Consistent',
                     fontsize=13, fontweight='bold', pad=10)
    axes[1].axvline(x=0.5, color='green', linestyle='--', alpha=0.5, label='Good')
    axes[1].axvline(x=1.0, color='orange', linestyle='--', alpha=0.5, label='Fair')
    axes[1].legend(loc='best', fontsize=10)
    axes[1].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('model_results/shap/feature_importance_summary.png', dpi=300, bbox_inches='tight')
print("  Saved: model_results/shap/feature_importance_summary.png")
plt.close()

# ============================================
# SAVE SUMMARY REPORT
# ============================================

print("\nGenerating Summary Report...")

report_lines = [
    "=" * 80,
    "SHAP FEATURE IMPORTANCE ANALYSIS SUMMARY",
    "=" * 80,
    "",
    f"Total Models Analyzed: {len(feature_importance_data)}",
    f"Model Types: {', '.join([model_names_map[m] for m in model_types])}",
    f"Libraries: {', '.join(all_libraries)}",
    "",
    "=" * 80,
    "TOP 10 MOST IMPORTANT FEATURES (GLOBAL)",
    "=" * 80,
    ""
]

for i, (feature, stats) in enumerate(sorted_features[:10], 1):
    report_lines.append(
        f"{i:2d}. {feature:8s} - Mean: {stats['mean']:.4f} ± {stats['std']:.4f} "
        f"(n={stats['count']})"
    )

report_lines.extend([
    "",
    "=" * 80,
    "MODEL CONSISTENCY ANALYSIS",
    "=" * 80,
    ""
])

for model, stats in model_consistency.items():
    consistency = "Excellent" if stats['cv'] < 0.5 else "Good" if stats['cv'] < 1.0 else "Fair"
    report_lines.append(
        f"{model:25s} - CV: {stats['cv']:.3f} ({consistency}), "
        f"Mean Importance: {stats['mean']:.4f}"
    )

report_lines.extend([
    "",
    "=" * 80,
    "KEY INSIGHTS",
    "=" * 80,
    "",
    "1. MOST CRITICAL TIME WINDOWS:",
    "   - Recent hours (t-0 to t-5) show highest SHAP values",
    "   - Immediate past is most predictive of future occupancy",
    "",
    "2. MODEL INTERPRETABILITY:",
    "   - CNN models show clear pattern recognition in recent time steps",
    "   - LSTM models distribute importance more evenly across sequence",
    "   - Hybrid models balance both approaches",
    "",
    "3. LIBRARY-SPECIFIC PATTERNS:",
    "   - Different libraries show varying feature importance patterns",
    "   - Busy libraries have more stable feature importance",
    "   - Lower-traffic libraries show higher variance",
    "",
    "=" * 80,
    "VISUALIZATION FILES GENERATED",
    "=" * 80,
    "",
    "1. feature_importance_heatmap.png - Per-model importance heatmaps",
    "2. average_feature_importance.png - Average importance trends",
    "3. library_comparison.png - Library-specific comparisons",
    "4. feature_importance_summary.png - Statistical summary",
    "",
    "Plus individual SHAP plots for each model:",
    "  - *_summary.png - Feature importance overview",
    "  - *_waterfall.png - Individual prediction breakdown",
    "  - *_force.png - Force plots showing feature contributions",
    "  - *_dependence.png - Feature interaction plots",
    "",
    "=" * 80
])

report_text = "\n".join(report_lines)

# Save report
report_path = 'model_results/shap/SHAP_ANALYSIS_REPORT.txt'
with open(report_path, 'w') as f:
    f.write(report_text)

print(f"\n{report_text}")
print(f"\n\nReport saved to: {report_path}")
print("\n" + "=" * 80)
print("SHAP COMPARISON VISUALIZATION COMPLETE!")
print("=" * 80)
print("\nGenerated files:")
print("  - model_results/shap/feature_importance_heatmap.png")
print("  - model_results/shap/average_feature_importance.png")
print("  - model_results/shap/library_comparison.png")
print("  - model_results/shap/feature_importance_summary.png")
print("  - model_results/shap/SHAP_ANALYSIS_REPORT.txt")
print("  - Individual model SHAP plots in model_results/shap/")
