import React, { useState, useEffect } from 'react';
import {
  LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer, ComposedChart, ScatterChart, Scatter
} from 'recharts';

const AdminDashboard = () => {
  const [selectedModelType, setSelectedModelType] = useState('cnn_only');
  const [modelTypes, setModelTypes] = useState([]);
  const [predictions, setPredictions] = useState(null);
  const [modelMetrics, setModelMetrics] = useState(null);
  const [loading, setLoading] = useState(true);
  const [lastUpdate, setLastUpdate] = useState(new Date());
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [error, setError] = useState(null);
  const [viewMode, setViewMode] = useState('overview'); // overview, metrics, comparison

  // Fetch model types
  const fetchModelTypes = async () => {
    try {
      const response = await fetch('http://localhost:5000/api/model-types');
      if (response.ok) {
        const data = await response.json();
        setModelTypes(data.model_types || []);
      }
    } catch (error) {
      console.error('Error fetching model types:', error);
    }
  };

  // Fetch predictions for selected model type
  const fetchPredictions = async () => {
    try {
      setLoading(true);
      setError(null);

      const response = await fetch(`http://localhost:5000/api/predictions?model_type=${selectedModelType}`);
      if (!response.ok) throw new Error(`API returned ${response.status}`);

      const jsonData = await response.json();
      setPredictions(jsonData);
      setLastUpdate(new Date());
    } catch (error) {
      console.error('Error fetching predictions:', error);
      setError(error.message);
    } finally {
      setLoading(false);
    }
  };

  // Fetch model metrics (all model types)
  const fetchModelMetrics = async () => {
    try {
      const response = await fetch('http://localhost:5000/api/models/metrics');
      if (response.ok) {
        const metrics = await response.json();
        setModelMetrics(metrics);
      }
    } catch (error) {
      console.error('Error fetching model metrics:', error);
    }
  };

  useEffect(() => {
    fetchModelTypes();
    fetchModelMetrics();
  }, []);

  useEffect(() => {
    if (selectedModelType) {
      fetchPredictions();

      if (autoRefresh) {
        const interval = setInterval(fetchPredictions, 60000);
        return () => clearInterval(interval);
      }
    }
  }, [selectedModelType, autoRefresh]);

  // Get comparison data for all model types
  const getModelComparisonData = () => {
    if (!modelMetrics || !modelMetrics.model_types) return [];

    return Object.entries(modelMetrics.model_types).map(([modelTypeId, modelData]) => ({
      name: modelData.model_name || modelTypeId,
      id: modelTypeId,
      avgR2: calculateAvgMetric(modelData.libraries, 'r2'),
      avgRMSE: calculateAvgMetric(modelData.libraries, 'rmse'),
      avgMAE: calculateAvgMetric(modelData.libraries, 'mae'),
      avgMAPE: calculateAvgMetric(modelData.libraries, 'mape'),
      numLibraries: Object.keys(modelData.libraries || {}).length
    }));
  };

  // Calculate average metric across all libraries for a model
  const calculateAvgMetric = (libraries, metricName) => {
    if (!libraries) return 0;
    const values = Object.values(libraries).map(lib => lib.metrics?.[metricName] || 0);
    return values.reduce((sum, val) => sum + val, 0) / values.length;
  };

  // Get library performance for selected model type
  const getLibraryPerformanceData = () => {
    if (!modelMetrics || !modelMetrics.model_types || !selectedModelType) return [];

    const selectedModel = modelMetrics.model_types[selectedModelType];
    if (!selectedModel || !selectedModel.libraries) return [];

    return Object.entries(selectedModel.libraries).map(([libId, libData]) => ({
      name: libData.library_name || libId,
      id: libId,
      r2: (libData.metrics?.r2 || 0).toFixed(4),
      rmse: (libData.metrics?.rmse || 0).toFixed(2),
      mae: (libData.metrics?.mae || 0).toFixed(2),
      mape: (libData.metrics?.mape || 0).toFixed(2),
      avgOccupancy: (libData.data_stats?.avg_occupancy || 0).toFixed(1),
      maxOccupancy: libData.data_stats?.max_occupancy || 0
    }));
  };

  // Get current predictions for all libraries
  const getCurrentPredictions = () => {
    if (!predictions || !predictions.libraries) return [];

    return Object.values(predictions.libraries).map(lib => ({
      name: lib.library_name || lib.library_id,
      current: lib.current || 0,
      predicted: lib.predicted || 0,
      change: lib.change || 0,
      avg_24h: lib.avg_24h || 0
    }));
  };

  // Loading state
  if (loading && !predictions) {
    return (
      <div style={{ padding: '40px', textAlign: 'center', color: '#666' }}>
        <h2>Loading Model Comparison Dashboard...</h2>
      </div>
    );
  }

  // Error state
  if (error) {
    return (
      <div style={{ padding: '40px', textAlign: 'center' }}>
        <h2 style={{ color: '#e74c3c' }}>Error Loading Data</h2>
        <p>{error}</p>
        <button onClick={fetchPredictions} style={{
          padding: '10px 20px',
          background: '#3498db',
          color: 'white',
          border: 'none',
          borderRadius: '5px',
          cursor: 'pointer'
        }}>
          Retry
        </button>
      </div>
    );
  }

  const libraryData = getCurrentPredictions();
  const modelComparison = getModelComparisonData();
  const libraryPerformance = getLibraryPerformanceData();

  return (
    <div style={{ background: '#f5f7fa', minHeight: '100vh', padding: '20px' }}>
      {/* Header */}
      <div style={{
        background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
        padding: '30px',
        borderRadius: '15px',
        marginBottom: '20px',
        color: 'white'
      }}>
        <h1 style={{ margin: 0, marginBottom: '10px', fontSize: '32px' }}>
          Model Type Comparison Dashboard
        </h1>
        <p style={{ margin: 0, opacity: 0.9 }}>
          Compare LSTM, CNN, Hybrid, and Advanced CNN-LSTM Models Across All Libraries
        </p>
      </div>

      {/* Controls */}
      <div style={{
        background: 'white',
        padding: '20px',
        borderRadius: '10px',
        marginBottom: '20px',
        boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
        display: 'flex',
        gap: '20px',
        flexWrap: 'wrap',
        alignItems: 'center'
      }}>
        <div style={{ flex: '1', minWidth: '200px' }}>
          <label style={{ display: 'block', marginBottom: '5px', fontWeight: 'bold', color: '#555' }}>
            Select Model Type:
          </label>
          <select
            value={selectedModelType}
            onChange={(e) => setSelectedModelType(e.target.value)}
            style={{
              width: '100%',
              padding: '10px',
              borderRadius: '5px',
              border: '2px solid #ddd',
              fontSize: '16px',
              cursor: 'pointer'
            }}
          >
            {modelTypes.map(model => (
              <option key={model.id} value={model.id}>{model.name}</option>
            ))}
          </select>
        </div>

        <div style={{ flex: '1', minWidth: '200px' }}>
          <label style={{ display: 'block', marginBottom: '5px', fontWeight: 'bold', color: '#555' }}>
            View Mode:
          </label>
          <select
            value={viewMode}
            onChange={(e) => setViewMode(e.target.value)}
            style={{
              width: '100%',
              padding: '10px',
              borderRadius: '5px',
              border: '2px solid #ddd',
              fontSize: '16px',
              cursor: 'pointer'
            }}
          >
            <option value="overview">Overview - Current Predictions</option>
            <option value="metrics">Performance Metrics by Library</option>
            <option value="comparison">Model Type Comparison</option>
          </select>
        </div>

        <div style={{ display: 'flex', gap: '10px', alignItems: 'flex-end' }}>
          <button
            onClick={() => setAutoRefresh(!autoRefresh)}
            style={{
              padding: '10px 20px',
              background: autoRefresh ? '#2ecc71' : '#95a5a6',
              color: 'white',
              border: 'none',
              borderRadius: '5px',
              cursor: 'pointer'
            }}
          >
            Auto-refresh: {autoRefresh ? 'ON' : 'OFF'}
          </button>
          <button
            onClick={fetchPredictions}
            style={{
              padding: '10px 20px',
              background: '#3498db',
              color: 'white',
              border: 'none',
              borderRadius: '5px',
              cursor: 'pointer'
            }}
          >
            Refresh Now
          </button>
        </div>

        <div style={{ marginLeft: 'auto', fontSize: '14px', color: '#666' }}>
          Last updated: {lastUpdate.toLocaleTimeString()}
        </div>
      </div>

      {/* Overview Mode */}
      {viewMode === 'overview' && (
        <>
          <div style={{
            background: 'white',
            padding: '20px',
            borderRadius: '10px',
            marginBottom: '20px',
            boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
          }}>
            <h2 style={{ marginTop: 0 }}>
              Current Predictions - {modelTypes.find(m => m.id === selectedModelType)?.name}
            </h2>
            <p style={{ color: '#666' }}>
              Real-time occupancy predictions for all libraries using the selected model
            </p>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: '15px', marginTop: '20px' }}>
              {libraryData.map((lib, idx) => (
                <div key={idx} style={{
                  background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                  padding: '20px',
                  borderRadius: '10px',
                  color: 'white'
                }}>
                  <h3 style={{ margin: '0 0 15px 0', fontSize: '16px' }}>{lib.name}</h3>
                  <div style={{ fontSize: '32px', fontWeight: 'bold', marginBottom: '10px' }}>
                    {lib.current} users
                  </div>
                  <div style={{ fontSize: '14px', opacity: 0.9 }}>
                    Predicted next: <strong>{lib.predicted}</strong> users
                    <br />
                    Change: <strong style={{ color: lib.change >= 0 ? '#2ecc71' : '#e74c3c' }}>
                      {lib.change >= 0 ? '+' : ''}{lib.change}
                    </strong>
                    <br />
                    24h avg: {lib.avg_24h} users
                  </div>
                </div>
              ))}
            </div>
          </div>

          {predictions && predictions.libraries && Object.values(predictions.libraries)[0]?.hourly_data && (
            <div style={{
              background: 'white',
              padding: '20px',
              borderRadius: '10px',
              marginBottom: '20px',
              boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
            }}>
              <h2>Hourly Trends (Last 24h + Predictions)</h2>
              <ResponsiveContainer width="100%" height={400}>
                <LineChart data={Object.values(predictions.libraries)[0].hourly_data}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="time" angle={-45} textAnchor="end" height={100} />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Line type="monotone" dataKey="actual" stroke="#3498db" strokeWidth={2} name="Actual" />
                  <Line type="monotone" dataKey="predicted" stroke="#e74c3c" strokeWidth={2} strokeDasharray="5 5" name="Predicted" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}
        </>
      )}

      {/* Metrics Mode */}
      {viewMode === 'metrics' && (
        <div style={{
          background: 'white',
          padding: '20px',
          borderRadius: '10px',
          marginBottom: '20px',
          boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
        }}>
          <h2 style={{ marginTop: 0 }}>
            Performance Metrics by Library - {modelTypes.find(m => m.id === selectedModelType)?.name}
          </h2>
          <p style={{ color: '#666' }}>
            Training accuracy metrics for each library using this model type
          </p>

          {/* R² Score Chart */}
          <div style={{ marginTop: '30px' }}>
            <h3>R² Score (Coefficient of Determination)</h3>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={libraryPerformance}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" angle={-15} textAnchor="end" height={100} />
                <YAxis domain={[0, 1]} />
                <Tooltip />
                <Bar dataKey="r2" fill="#2ecc71" name="R² Score" />
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* RMSE Chart */}
          <div style={{ marginTop: '30px' }}>
            <h3>RMSE (Root Mean Squared Error)</h3>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={libraryPerformance}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" angle={-15} textAnchor="end" height={100} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="rmse" fill="#e74c3c" name="RMSE" />
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Detailed Metrics Table */}
          <div style={{ marginTop: '30px', overflowX: 'auto' }}>
            <h3>Detailed Metrics</h3>
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead>
                <tr style={{ background: '#f8f9fa' }}>
                  <th style={{ padding: '12px', textAlign: 'left', borderBottom: '2px solid #dee2e6' }}>Library</th>
                  <th style={{ padding: '12px', textAlign: 'center', borderBottom: '2px solid #dee2e6' }}>R²</th>
                  <th style={{ padding: '12px', textAlign: 'center', borderBottom: '2px solid #dee2e6' }}>RMSE</th>
                  <th style={{ padding: '12px', textAlign: 'center', borderBottom: '2px solid #dee2e6' }}>MAE</th>
                  <th style={{ padding: '12px', textAlign: 'center', borderBottom: '2px solid #dee2e6' }}>MAPE</th>
                  <th style={{ padding: '12px', textAlign: 'center', borderBottom: '2px solid #dee2e6' }}>Avg Occupancy</th>
                  <th style={{ padding: '12px', textAlign: 'center', borderBottom: '2px solid #dee2e6' }}>Max Occupancy</th>
                </tr>
              </thead>
              <tbody>
                {libraryPerformance.map((lib, idx) => (
                  <tr key={idx} style={{ borderBottom: '1px solid #dee2e6' }}>
                    <td style={{ padding: '12px' }}>{lib.name}</td>
                    <td style={{ padding: '12px', textAlign: 'center' }}>{lib.r2}</td>
                    <td style={{ padding: '12px', textAlign: 'center' }}>{lib.rmse}</td>
                    <td style={{ padding: '12px', textAlign: 'center' }}>{lib.mae}</td>
                    <td style={{ padding: '12px', textAlign: 'center' }}>{lib.mape}%</td>
                    <td style={{ padding: '12px', textAlign: 'center' }}>{lib.avgOccupancy}</td>
                    <td style={{ padding: '12px', textAlign: 'center' }}>{lib.maxOccupancy}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Comparison Mode */}
      {viewMode === 'comparison' && (
        <div style={{
          background: 'white',
          padding: '20px',
          borderRadius: '10px',
          marginBottom: '20px',
          boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
        }}>
          <h2 style={{ marginTop: 0 }}>Model Type Comparison</h2>
          <p style={{ color: '#666' }}>
            Compare average performance across all model architectures
          </p>

          {/* Average R² Comparison */}
          <div style={{ marginTop: '30px' }}>
            <h3>Average R² Score by Model Type</h3>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={modelComparison}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis domain={[0, 1]} />
                <Tooltip />
                <Bar dataKey="avgR2" fill="#2ecc71" name="Avg R²" />
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Average RMSE Comparison */}
          <div style={{ marginTop: '30px' }}>
            <h3>Average RMSE by Model Type</h3>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={modelComparison}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis />
                <Tooltip />
                <Bar dataKey="avgRMSE" fill="#e74c3c" name="Avg RMSE" />
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Comparison Table */}
          <div style={{ marginTop: '30px', overflowX: 'auto' }}>
            <h3>Model Performance Summary</h3>
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead>
                <tr style={{ background: '#f8f9fa' }}>
                  <th style={{ padding: '12px', textAlign: 'left', borderBottom: '2px solid #dee2e6' }}>Model Type</th>
                  <th style={{ padding: '12px', textAlign: 'center', borderBottom: '2px solid #dee2e6' }}>Avg R²</th>
                  <th style={{ padding: '12px', textAlign: 'center', borderBottom: '2px solid #dee2e6' }}>Avg RMSE</th>
                  <th style={{ padding: '12px', textAlign: 'center', borderBottom: '2px solid #dee2e6' }}>Avg MAE</th>
                  <th style={{ padding: '12px', textAlign: 'center', borderBottom: '2px solid #dee2e6' }}>Avg MAPE</th>
                  <th style={{ padding: '12px', textAlign: 'center', borderBottom: '2px solid #dee2e6' }}>Libraries</th>
                </tr>
              </thead>
              <tbody>
                {modelComparison.map((model, idx) => (
                  <tr key={idx} style={{
                    borderBottom: '1px solid #dee2e6',
                    background: model.id === selectedModelType ? '#e3f2fd' : 'transparent'
                  }}>
                    <td style={{ padding: '12px', fontWeight: model.id === selectedModelType ? 'bold' : 'normal' }}>
                      {model.name}
                    </td>
                    <td style={{ padding: '12px', textAlign: 'center' }}>{model.avgR2.toFixed(4)}</td>
                    <td style={{ padding: '12px', textAlign: 'center' }}>{model.avgRMSE.toFixed(2)}</td>
                    <td style={{ padding: '12px', textAlign: 'center' }}>{model.avgMAE.toFixed(2)}</td>
                    <td style={{ padding: '12px', textAlign: 'center' }}>{model.avgMAPE.toFixed(2)}%</td>
                    <td style={{ padding: '12px', textAlign: 'center' }}>{model.numLibraries}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Footer */}
      <div style={{
        background: 'white',
        padding: '15px',
        borderRadius: '10px',
        textAlign: 'center',
        color: '#666',
        fontSize: '14px',
        boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
      }}>
        <p style={{ margin: 0 }}>
          Model Comparison Dashboard | Last updated: {lastUpdate.toLocaleString()}
        </p>
      </div>
    </div>
  );
};

export default AdminDashboard;
