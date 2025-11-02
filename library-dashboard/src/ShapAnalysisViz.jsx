// SHAP Analysis Visualization Component
// Displays SHAP interpretability analysis for all models

import React, { useState, useEffect } from 'react';
import './ShapAnalysisViz.css';

const ShapAnalysisViz = () => {
  const [selectedModel, setSelectedModel] = useState('cnn_only');
  const [selectedLibrary, setSelectedLibrary] = useState('miguel_pro');
  const [selectedView, setSelectedView] = useState('summary');
  const [shapResults, setShapResults] = useState(null);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState('individual');

  // Model types available
  const modelTypes = [
    { id: 'lstm_only', name: 'LSTM Only', color: '#3B82F6' },
    { id: 'cnn_only', name: 'CNN Only', color: '#10B981' },
    { id: 'hybrid_cnn_lstm', name: 'Hybrid CNN-LSTM', color: '#F59E0B' },
    { id: 'advanced_cnn_lstm', name: 'Advanced CNN-LSTM', color: '#8B5CF6' }
  ];

  // Library locations
  const libraries = [
    { id: 'miguel_pro', name: 'Miguel Pro Library' },
    { id: 'american_corner', name: 'American Corner' },
    { id: 'gisbert_2nd', name: 'Gisbert 2nd Floor' },
    { id: 'gisbert_3rd', name: 'Gisbert 3rd Floor' },
    { id: 'gisbert_4th', name: 'Gisbert 4th Floor' },
    { id: 'gisbert_5th', name: 'Gisbert 5th Floor' }
  ];

  // View types
  const viewTypes = [
    { id: 'summary', name: 'Feature Importance', icon: '📊' },
    { id: 'waterfall', name: 'Prediction Breakdown', icon: '💧' },
    { id: 'force', name: 'Force Plot', icon: '⚡' },
    { id: 'dependence', name: 'Feature Interactions', icon: '🔗' }
  ];

  // Fetch SHAP results from API
  const fetchShapResults = async () => {
    try {
      setLoading(true);
      const response = await fetch('http://localhost:5000/api/shap/results');
      const data = await response.json();
      setShapResults(data);
    } catch (error) {
      console.error('Error fetching SHAP results:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchShapResults();
  }, []);

  // Map library IDs to actual filenames
  const getLibraryFilename = (libraryId) => {
    const mapping = {
      'miguel_pro': 'miguel_pro_library',
      'american_corner': 'american_corner',
      'gisbert_2nd': 'gisbert_2nd_floor',
      'gisbert_3rd': 'gisbert_3rd_floor',
      'gisbert_4th': 'gisbert_4th_floor',
      'gisbert_5th': 'gisbert_5th_floor'
    };
    return mapping[libraryId] || libraryId;
  };

  // Get image path for current selection
  const getImagePath = (viewType) => {
    const libraryFilename = getLibraryFilename(selectedLibrary);
    return `http://localhost:5000/api/shap/images/${selectedModel}_${libraryFilename}_${viewType}.png`;
  };

  // Get comparison image paths
  const getComparisonImagePath = (type) => {
    return `http://localhost:5000/api/shap/images/${type}.png`;
  };

  // Get feature importance data for selected model
  const getFeatureImportance = () => {
    if (!shapResults) return null;

    try {
      const modelData = shapResults[selectedModel];
      if (!modelData || !modelData.libraries) return null;

      const libraryData = modelData.libraries[selectedLibrary];
      if (!libraryData || !libraryData.shap_analysis) return null;

      return libraryData.shap_analysis.feature_importance;
    } catch (error) {
      return null;
    }
  };

  const featureImportance = getFeatureImportance();

  if (loading) {
    return (
      <div className="shap-loading">
        <div className="spinner"></div>
        <p>Loading SHAP analysis...</p>
      </div>
    );
  }

  return (
    <div className="shap-container">
      {/* Header */}
      <div className="shap-header">
        <h1>🔍 SHAP Model Interpretability Analysis</h1>
        <p>Understand which time steps drive predictions and why models make specific decisions</p>
      </div>

      {/* Controls */}
      <div className="shap-controls">
        <div className="control-group">
          <label>Model Type</label>
          <select value={selectedModel} onChange={(e) => setSelectedModel(e.target.value)}>
            {modelTypes.map(model => (
              <option key={model.id} value={model.id}>
                {model.name}
              </option>
            ))}
          </select>
        </div>

        <div className="control-group">
          <label>Library Location</label>
          <select value={selectedLibrary} onChange={(e) => setSelectedLibrary(e.target.value)}>
            {libraries.map(lib => (
              <option key={lib.id} value={lib.id}>
                {lib.name}
              </option>
            ))}
          </select>
        </div>

        <div className="control-group">
          <label>Visualization Type</label>
          <select value={selectedView} onChange={(e) => setSelectedView(e.target.value)}>
            {viewTypes.map(view => (
              <option key={view.id} value={view.id}>
                {view.icon} {view.name}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Tabs */}
      <div className="shap-tabs">
        <button
          className={activeTab === 'individual' ? 'tab active' : 'tab'}
          onClick={() => setActiveTab('individual')}
        >
          Individual Model Analysis
        </button>
        <button
          className={activeTab === 'comparison' ? 'tab active' : 'tab'}
          onClick={() => setActiveTab('comparison')}
        >
          Cross-Model Comparison
        </button>
      </div>

      {/* Individual Model Tab */}
      {activeTab === 'individual' && (
        <div className="tab-content">
          {/* View Description */}
          <div className="view-description">
            {selectedView === 'summary' && (
              <p>📊 Which time steps (hours) are most important for predictions? Higher bars indicate stronger influence.</p>
            )}
            {selectedView === 'waterfall' && (
              <p>💧 How individual features contribute to a single prediction. Red = increases prediction, Blue = decreases.</p>
            )}
            {selectedView === 'force' && (
              <p>⚡ Alternative view showing how features push prediction away from base value.</p>
            )}
            {selectedView === 'dependence' && (
              <p>🔗 How the value of a feature affects its impact on predictions. Shows non-linear relationships.</p>
            )}
          </div>

          {/* Main Visualization */}
          <div className="shap-viz-card">
            <img
              src={getImagePath(selectedView)}
              alt={`SHAP ${selectedView} Plot`}
              className="shap-image"
              onError={(e) => {
                e.target.style.display = 'none';
                e.target.nextSibling.style.display = 'block';
              }}
            />
            <div className="image-placeholder" style={{display: 'none'}}>
              <p>SHAP plot not yet generated. Run training with SHAP analysis first.</p>
              <code>python train_multiple_model_types.py</code>
            </div>
          </div>

          {/* Feature Importance Table */}
          {selectedView === 'summary' && featureImportance && (
            <div className="feature-table-card">
              <h3>Top 10 Most Important Features</h3>
              <table className="feature-table">
                <thead>
                  <tr>
                    <th>Rank</th>
                    <th>Time Step</th>
                    <th>Hours Ago</th>
                    <th>Importance</th>
                    <th>Impact</th>
                  </tr>
                </thead>
                <tbody>
                  {featureImportance.top_features_idx.slice(0, 10).map((idx, rank) => {
                    const importance = featureImportance.top_features_importance[rank];
                    const barWidth = (importance / featureImportance.top_features_importance[0]) * 100;

                    return (
                      <tr key={rank}>
                        <td>#{rank + 1}</td>
                        <td>t-{idx}</td>
                        <td>{idx} hours ago</td>
                        <td>{importance.toFixed(4)}</td>
                        <td>
                          <div className="importance-bar-container">
                            <div
                              className="importance-bar"
                              style={{ width: `${barWidth}%` }}
                            />
                          </div>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}

      {/* Comparison Tab */}
      {activeTab === 'comparison' && (
        <div className="tab-content">
          <h2>Cross-Model Comparison</h2>
          <p className="section-desc">Compare feature importance across all models and libraries</p>

          <div className="comparison-grid">
            <div className="comparison-item">
              <h3>Feature Importance Heatmap</h3>
              <img
                src={getComparisonImagePath('feature_importance_heatmap')}
                alt="Feature Importance Heatmap"
                className="shap-image"
                onError={(e) => e.target.style.display = 'none'}
              />
            </div>

            <div className="comparison-item">
              <h3>Average Feature Importance</h3>
              <img
                src={getComparisonImagePath('average_feature_importance')}
                alt="Average Feature Importance"
                className="shap-image"
                onError={(e) => e.target.style.display = 'none'}
              />
            </div>

            <div className="comparison-item full-width">
              <h3>Library-wise Comparison</h3>
              <img
                src={getComparisonImagePath('library_comparison')}
                alt="Library Comparison"
                className="shap-image"
                onError={(e) => e.target.style.display = 'none'}
              />
            </div>

            <div className="comparison-item full-width">
              <h3>Feature Importance Summary Statistics</h3>
              <img
                src={getComparisonImagePath('feature_importance_summary')}
                alt="Feature Importance Summary"
                className="shap-image"
                onError={(e) => e.target.style.display = 'none'}
              />
            </div>
          </div>
        </div>
      )}

      {/* Info Cards */}
      <div className="info-cards">
        <div className="info-card blue">
          <div className="info-icon">📊</div>
          <h3>Interpretability</h3>
          <p>Understand exactly which hours drive predictions</p>
        </div>

        <div className="info-card green">
          <div className="info-icon">🔍</div>
          <h3>Transparency</h3>
          <p>See why the model makes specific decisions</p>
        </div>

        <div className="info-card purple">
          <div className="info-icon">✅</div>
          <h3>Validation</h3>
          <p>Verify models use reasonable patterns</p>
        </div>
      </div>
    </div>
  );
};

export default ShapAnalysisViz;
