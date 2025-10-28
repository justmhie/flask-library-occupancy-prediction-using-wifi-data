import React, { useState, useEffect } from 'react';
import {
  LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer, AreaChart, Area, RadarChart,
  Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis, ScatterChart, Scatter
} from 'recharts';

const AdminDashboard = () => {
  const [selectedLibrary, setSelectedLibrary] = useState('all');
  const [data, setData] = useState(null);
  const [modelMetrics, setModelMetrics] = useState(null);
  const [loading, setLoading] = useState(true);
  const [lastUpdate, setLastUpdate] = useState(new Date());
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [error, setError] = useState(null);
  const [viewMode, setViewMode] = useState('overview'); // overview, predictions, comparison, metrics

  // Library locations
  const libraries = [
    { id: 'all', name: 'All Libraries (Overall)' },
    { id: 'miguel_pro', name: 'Miguel Pro Library' },
    { id: 'gisbert_2nd', name: 'Gisbert 2nd Floor' },
    { id: 'american_corner', name: 'American Corner' },
    { id: 'gisbert_3rd', name: 'Gisbert 3rd Floor' },
    { id: 'gisbert_4th', name: 'Gisbert 4th Floor' },
    { id: 'gisbert_5th', name: 'Gisbert 5th Floor' }
  ];

  // Fetch predictions from API
  const fetchData = async () => {
    try {
      setLoading(true);
      setError(null);

      const response = await fetch('http://localhost:5000/api/predictions');
      if (!response.ok) throw new Error(`API returned ${response.status}`);

      const jsonData = await response.json();
      setData(jsonData);
      setLastUpdate(new Date());
    } catch (error) {
      console.error('Error fetching predictions:', error);
      setError(error.message);
    } finally {
      setLoading(false);
    }
  };

  // Fetch model metrics
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
    fetchData();
    fetchModelMetrics();

    if (autoRefresh) {
      const interval = setInterval(() => {
        fetchData();
        fetchModelMetrics();
      }, 60000);
      return () => clearInterval(interval);
    }
  }, [autoRefresh]);

  // Get data for selected library
  const getLibraryData = () => {
    if (!data || !data.libraries) return null;
    return data.libraries[selectedLibrary] || data.libraries.all;
  };

  const libraryData = getLibraryData();

  // Get metrics for comparison
  const getComparisonData = () => {
    if (!modelMetrics || !modelMetrics.models) return [];

    return Object.entries(modelMetrics.models).map(([id, model]) => ({
      name: libraries.find(lib => lib.id === id)?.name || id,
      id: id,
      r2: model.metrics?.r2 || 0,
      rmse: model.metrics?.rmse || 0,
      mae: model.metrics?.mae || 0,
      mape: model.metrics?.mape || 0,
      avgOccupancy: model.data_stats?.avg_occupancy || 0,
      maxOccupancy: model.data_stats?.max_occupancy || 0
    }));
  };

  // Loading state
  if (loading && !data) {
    return (
      <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh', background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)' }}>
        <div style={{ textAlign: 'center', color: 'white' }}>
          <div style={{ fontSize: '3rem', marginBottom: '1rem' }}>⏳</div>
          <p style={{ fontSize: '1.5rem' }}>Loading Admin Dashboard...</p>
        </div>
      </div>
    );
  }

  return (
    <div style={{ minHeight: '100vh', background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)', padding: '2rem' }}>
      <div style={{ maxWidth: '1600px', margin: '0 auto' }}>

        {/* Header */}
        <div style={{
          background: 'white',
          borderRadius: '12px',
          boxShadow: '0 4px 6px rgba(0,0,0,0.1)',
          padding: '2rem',
          marginBottom: '2rem'
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1rem' }}>
            <div>
              <h1 style={{ fontSize: '2.5rem', fontWeight: 'bold', color: '#1a202c', margin: 0 }}>
                🎛️ Library Occupancy Admin Dashboard
              </h1>
              <p style={{ color: '#718096', marginTop: '0.5rem', fontSize: '1.1rem' }}>
                Real-time predictions with CNN-LSTM models for all library locations
              </p>
            </div>
            <div style={{ textAlign: 'right' }}>
              <p style={{ fontSize: '0.875rem', color: '#a0aec0', margin: 0 }}>Last Updated</p>
              <p style={{ fontSize: '1.125rem', fontWeight: '600', color: '#2d3748', margin: '0.25rem 0' }}>
                {lastUpdate.toLocaleTimeString()}
              </p>
              <button
                onClick={() => setAutoRefresh(!autoRefresh)}
                style={{
                  marginTop: '0.5rem',
                  padding: '0.5rem 1rem',
                  borderRadius: '8px',
                  fontSize: '0.875rem',
                  fontWeight: '500',
                  border: 'none',
                  cursor: 'pointer',
                  background: autoRefresh ? '#48bb78' : '#cbd5e0',
                  color: 'white'
                }}
              >
                {autoRefresh ? '🔄 Auto-refresh ON' : '⏸️ Auto-refresh OFF'}
              </button>
            </div>
          </div>

          {/* View Mode Selector */}
          <div style={{ marginTop: '1.5rem', display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
            {[
              { id: 'overview', name: '📊 Overview', icon: '📊' },
              { id: 'predictions', name: '🔮 Predictions', icon: '🔮' },
              { id: 'comparison', name: '📈 Model Comparison', icon: '📈' },
              { id: 'metrics', name: '📉 Detailed Metrics', icon: '📉' }
            ].map(mode => (
              <button
                key={mode.id}
                onClick={() => setViewMode(mode.id)}
                style={{
                  padding: '0.75rem 1.5rem',
                  borderRadius: '8px',
                  border: viewMode === mode.id ? '2px solid #667eea' : '2px solid #e2e8f0',
                  background: viewMode === mode.id ? '#667eea' : 'white',
                  color: viewMode === mode.id ? 'white' : '#4a5568',
                  cursor: 'pointer',
                  fontWeight: '600',
                  fontSize: '0.95rem',
                  transition: 'all 0.2s'
                }}
              >
                {mode.name}
              </button>
            ))}
          </div>

          {/* Library Selector */}
          {viewMode !== 'comparison' && (
            <div style={{ marginTop: '1.5rem' }}>
              <label style={{ fontSize: '0.875rem', fontWeight: '600', color: '#4a5568', display: 'block', marginBottom: '0.5rem' }}>
                Select Library:
              </label>
              <select
                value={selectedLibrary}
                onChange={(e) => setSelectedLibrary(e.target.value)}
                style={{
                  width: '100%',
                  maxWidth: '500px',
                  padding: '0.75rem',
                  fontSize: '1rem',
                  border: '2px solid #e2e8f0',
                  borderRadius: '8px',
                  background: 'white',
                  cursor: 'pointer'
                }}
              >
                {libraries.map(lib => (
                  <option key={lib.id} value={lib.id}>{lib.name}</option>
                ))}
              </select>
            </div>
          )}
        </div>

        {error && (
          <div style={{
            background: '#fff5f5',
            border: '1px solid #feb2b2',
            borderRadius: '8px',
            padding: '1rem',
            marginBottom: '1rem',
            color: '#c53030'
          }}>
            ⚠️ API Error: {error}
          </div>
        )}

        {/* OVERVIEW MODE */}
        {viewMode === 'overview' && libraryData && (
          <OverviewView libraryData={libraryData} selectedLibrary={selectedLibrary} libraries={libraries} />
        )}

        {/* PREDICTIONS MODE */}
        {viewMode === 'predictions' && libraryData && (
          <PredictionsView libraryData={libraryData} />
        )}

        {/* COMPARISON MODE */}
        {viewMode === 'comparison' && modelMetrics && (
          <ComparisonView comparisonData={getComparisonData()} />
        )}

        {/* METRICS MODE */}
        {viewMode === 'metrics' && modelMetrics && (
          <MetricsView modelMetrics={modelMetrics} selectedLibrary={selectedLibrary} libraries={libraries} />
        )}

      </div>
    </div>
  );
};

// ============================================
// OVERVIEW VIEW COMPONENT
// ============================================

const OverviewView = ({ libraryData, selectedLibrary, libraries }) => (
  <div>
    {/* Stats Cards */}
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1rem', marginBottom: '2rem' }}>
      <StatCard
        title="Current Users"
        value={libraryData.current || 0}
        subtitle="Right now"
        icon="👥"
        color="#667eea"
      />
      <StatCard
        title="Predicted Next Hour"
        value={libraryData.predicted || 0}
        subtitle={`${libraryData.change > 0 ? '📈' : '📉'} ${Math.abs(libraryData.change || 0)} users`}
        icon="🔮"
        color="#764ba2"
      />
      <StatCard
        title="24h Average"
        value={libraryData.avg_24h || 0}
        subtitle="Average users"
        icon="📊"
        color="#f093fb"
      />
      <StatCard
        title="Peak Today"
        value={libraryData.peak_today || 0}
        subtitle={`at ${libraryData.peak_time || 'N/A'}`}
        icon="⭐"
        color="#4facfe"
      />
      <StatCard
        title="7-Day Average"
        value={libraryData.avg_7d || 0}
        subtitle="Weekly average"
        icon="📅"
        color="#43e97b"
      />
      <StatCard
        title="Max Capacity"
        value={libraryData.max_capacity || 'N/A'}
        subtitle={libraryData.max_capacity ? `${Math.round((libraryData.current / libraryData.max_capacity) * 100)}% used` : 'N/A'}
        icon="📈"
        color="#fa709a"
      />
    </div>

    {/* Hourly Trend Chart */}
    {libraryData.hourly_data && (
      <ChartCard title="📈 Hourly Occupancy Trend (Last 24 Hours + Predictions)">
        <ResponsiveContainer width="100%" height={350}>
          <AreaChart data={libraryData.hourly_data}>
            <defs>
              <linearGradient id="colorActual" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#667eea" stopOpacity={0.8}/>
                <stop offset="95%" stopColor="#667eea" stopOpacity={0}/>
              </linearGradient>
              <linearGradient id="colorPredicted" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#f093fb" stopOpacity={0.8}/>
                <stop offset="95%" stopColor="#f093fb" stopOpacity={0}/>
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="time" angle={-45} textAnchor="end" height={80} />
            <YAxis />
            <Tooltip />
            <Legend />
            <Area type="monotone" dataKey="actual" stroke="#667eea" fillOpacity={1} fill="url(#colorActual)" name="Actual Users" />
            <Area type="monotone" dataKey="predicted" stroke="#f093fb" strokeDasharray="5 5" fillOpacity={1} fill="url(#colorPredicted)" name="Predicted" />
          </AreaChart>
        </ResponsiveContainer>
      </ChartCard>
    )}

    {/* Daily Pattern */}
    {libraryData.daily_pattern && (
      <ChartCard title="📊 Daily Usage Pattern (Average by Hour of Day)">
        <ResponsiveContainer width="100%" height={350}>
          <BarChart data={libraryData.daily_pattern}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="hour" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Bar dataKey="average" fill="#764ba2" name="Average Users" />
            <Bar dataKey="peak" fill="#f093fb" name="Peak Users" />
          </BarChart>
        </ResponsiveContainer>
      </ChartCard>
    )}
  </div>
);

// ============================================
// PREDICTIONS VIEW COMPONENT
// ============================================

const PredictionsView = ({ libraryData }) => (
  <div>
    {/* Next Hours Forecast */}
    {libraryData.next_hours && (
      <ChartCard title="🔮 Next 6 Hours Forecast with Confidence Intervals">
        <ResponsiveContainer width="100%" height={400}>
          <LineChart data={libraryData.next_hours}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="time" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Line type="monotone" dataKey="predicted" stroke="#667eea" strokeWidth={3} name="Predicted" dot={{ r: 6 }} />
            <Line type="monotone" dataKey="confidence_upper" stroke="#cbd5e0" strokeDasharray="5 5" name="Upper Bound" />
            <Line type="monotone" dataKey="confidence_lower" stroke="#cbd5e0" strokeDasharray="5 5" name="Lower Bound" />
          </LineChart>
        </ResponsiveContainer>
      </ChartCard>
    )}

    {/* Predictions Table */}
    {libraryData.next_hours && (
      <div style={{
        background: 'white',
        borderRadius: '12px',
        boxShadow: '0 4px 6px rgba(0,0,0,0.1)',
        padding: '2rem',
        marginTop: '2rem'
      }}>
        <h2 style={{ fontSize: '1.5rem', fontWeight: 'bold', color: '#1a202c', marginBottom: '1.5rem' }}>
          📋 Detailed Predictions Table
        </h2>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ background: '#f7fafc', borderBottom: '2px solid #e2e8f0' }}>
                <th style={{ padding: '1rem', textAlign: 'left', fontWeight: '600' }}>Time</th>
                <th style={{ padding: '1rem', textAlign: 'right', fontWeight: '600' }}>Predicted Users</th>
                <th style={{ padding: '1rem', textAlign: 'right', fontWeight: '600' }}>Lower Bound</th>
                <th style={{ padding: '1rem', textAlign: 'right', fontWeight: '600' }}>Upper Bound</th>
                <th style={{ padding: '1rem', textAlign: 'right', fontWeight: '600' }}>Confidence Range</th>
              </tr>
            </thead>
            <tbody>
              {libraryData.next_hours.map((hour, idx) => (
                <tr key={idx} style={{ borderBottom: '1px solid #e2e8f0' }}>
                  <td style={{ padding: '1rem' }}>{hour.time}</td>
                  <td style={{ padding: '1rem', textAlign: 'right', fontWeight: '600', color: '#667eea' }}>{hour.predicted}</td>
                  <td style={{ padding: '1rem', textAlign: 'right', color: '#718096' }}>{hour.confidence_lower}</td>
                  <td style={{ padding: '1rem', textAlign: 'right', color: '#718096' }}>{hour.confidence_upper}</td>
                  <td style={{ padding: '1rem', textAlign: 'right', color: '#4a5568' }}>
                    ±{Math.round((hour.confidence_upper - hour.confidence_lower) / 2)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    )}
  </div>
);

// ============================================
// COMPARISON VIEW COMPONENT
// ============================================

const ComparisonView = ({ comparisonData }) => (
  <div>
    {/* R² Comparison */}
    <ChartCard title="📊 Model Performance: R² Scores (Higher is Better)">
      <ResponsiveContainer width="100%" height={400}>
        <BarChart data={comparisonData} layout="vertical">
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis type="number" domain={[0, 1]} />
          <YAxis dataKey="name" type="category" width={200} />
          <Tooltip />
          <Legend />
          <Bar dataKey="r2" fill="#667eea" name="R² Score" />
        </BarChart>
      </ResponsiveContainer>
    </ChartCard>

    {/* RMSE Comparison */}
    <ChartCard title="📉 Model Performance: RMSE (Lower is Better)">
      <ResponsiveContainer width="100%" height={400}>
        <BarChart data={comparisonData} layout="vertical">
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis type="number" />
          <YAxis dataKey="name" type="category" width={200} />
          <Tooltip />
          <Legend />
          <Bar dataKey="rmse" fill="#f093fb" name="RMSE (users)" />
        </BarChart>
      </ResponsiveContainer>
    </ChartCard>

    {/* Average Occupancy Comparison */}
    <ChartCard title="👥 Average Occupancy by Library">
      <ResponsiveContainer width="100%" height={400}>
        <BarChart data={comparisonData}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="name" angle={-45} textAnchor="end" height={120} />
          <YAxis />
          <Tooltip />
          <Legend />
          <Bar dataKey="avgOccupancy" fill="#43e97b" name="Average Users" />
          <Bar dataKey="maxOccupancy" fill="#fa709a" name="Peak Users" />
        </BarChart>
      </ResponsiveContainer>
    </ChartCard>

    {/* Comparison Table */}
    <div style={{
      background: 'white',
      borderRadius: '12px',
      boxShadow: '0 4px 6px rgba(0,0,0,0.1)',
      padding: '2rem',
      marginTop: '2rem'
    }}>
      <h2 style={{ fontSize: '1.5rem', fontWeight: 'bold', color: '#1a202c', marginBottom: '1.5rem' }}>
        📋 Model Metrics Comparison Table
      </h2>
      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead>
            <tr style={{ background: '#f7fafc', borderBottom: '2px solid #e2e8f0' }}>
              <th style={{ padding: '1rem', textAlign: 'left', fontWeight: '600' }}>Library</th>
              <th style={{ padding: '1rem', textAlign: 'right', fontWeight: '600' }}>R²</th>
              <th style={{ padding: '1rem', textAlign: 'right', fontWeight: '600' }}>RMSE</th>
              <th style={{ padding: '1rem', textAlign: 'right', fontWeight: '600' }}>MAE</th>
              <th style={{ padding: '1rem', textAlign: 'right', fontWeight: '600' }}>MAPE</th>
              <th style={{ padding: '1rem', textAlign: 'right', fontWeight: '600' }}>Avg Occupancy</th>
            </tr>
          </thead>
          <tbody>
            {comparisonData.map((lib, idx) => (
              <tr key={idx} style={{ borderBottom: '1px solid #e2e8f0' }}>
                <td style={{ padding: '1rem', fontWeight: '600' }}>{lib.name}</td>
                <td style={{ padding: '1rem', textAlign: 'right' }}>{lib.r2.toFixed(4)}</td>
                <td style={{ padding: '1rem', textAlign: 'right' }}>{lib.rmse.toFixed(2)}</td>
                <td style={{ padding: '1rem', textAlign: 'right' }}>{lib.mae.toFixed(2)}</td>
                <td style={{ padding: '1rem', textAlign: 'right' }}>{lib.mape.toFixed(2)}%</td>
                <td style={{ padding: '1rem', textAlign: 'right' }}>{lib.avgOccupancy.toFixed(1)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  </div>
);

// ============================================
// METRICS VIEW COMPONENT
// ============================================

const MetricsView = ({ modelMetrics, selectedLibrary, libraries }) => {
  const libraryName = libraries.find(lib => lib.id === selectedLibrary)?.name || selectedLibrary;
  const metrics = modelMetrics.models?.[selectedLibrary];

  if (!metrics) {
    return (
      <div style={{
        background: 'white',
        borderRadius: '12px',
        padding: '3rem',
        textAlign: 'center',
        color: '#718096'
      }}>
        <p style={{ fontSize: '1.2rem' }}>No metrics available for {libraryName}</p>
        <p>Please train the model first.</p>
      </div>
    );
  }

  return (
    <div>
      {/* Metrics Cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: '1rem', marginBottom: '2rem' }}>
        <MetricCard title="R² Score" value={metrics.metrics?.r2?.toFixed(4) || 'N/A'} description="Coefficient of Determination" color="#667eea" />
        <MetricCard title="RMSE" value={`${metrics.metrics?.rmse?.toFixed(2) || 'N/A'} users`} description="Root Mean Squared Error" color="#f093fb" />
        <MetricCard title="MAE" value={`${metrics.metrics?.mae?.toFixed(2) || 'N/A'} users`} description="Mean Absolute Error" color="#43e97b" />
        <MetricCard title="MAPE" value={`${metrics.metrics?.mape?.toFixed(2) || 'N/A'}%`} description="Mean Absolute % Error" color="#fa709a" />
      </div>

      {/* Data Statistics */}
      <div style={{
        background: 'white',
        borderRadius: '12px',
        boxShadow: '0 4px 6px rgba(0,0,0,0.1)',
        padding: '2rem',
        marginBottom: '2rem'
      }}>
        <h2 style={{ fontSize: '1.5rem', fontWeight: 'bold', color: '#1a202c', marginBottom: '1.5rem' }}>
          📊 Dataset Statistics
        </h2>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1.5rem' }}>
          <DataStat label="Total Records" value={(metrics.data_stats?.total_records || 0).toLocaleString()} />
          <DataStat label="Total Hours" value={(metrics.data_stats?.total_hours || 0).toLocaleString()} />
          <DataStat label="Training Samples" value={(metrics.data_stats?.training_samples || 0).toLocaleString()} />
          <DataStat label="Testing Samples" value={(metrics.data_stats?.testing_samples || 0).toLocaleString()} />
          <DataStat label="Min Occupancy" value={`${metrics.data_stats?.min_occupancy || 0} users`} />
          <DataStat label="Max Occupancy" value={`${metrics.data_stats?.max_occupancy || 0} users`} />
          <DataStat label="Avg Occupancy" value={`${metrics.data_stats?.avg_occupancy?.toFixed(1) || 0} users`} />
          <DataStat label="Epochs Trained" value={metrics.data_stats?.epochs_trained || 'N/A'} />
        </div>
      </div>

      {/* Model Info */}
      <div style={{
        background: 'white',
        borderRadius: '12px',
        boxShadow: '0 4px 6px rgba(0,0,0,0.1)',
        padding: '2rem'
      }}>
        <h2 style={{ fontSize: '1.5rem', fontWeight: 'bold', color: '#1a202c', marginBottom: '1.5rem' }}>
          🤖 Model Information
        </h2>
        <div style={{ display: 'grid', gap: '1rem' }}>
          <InfoRow label="Location" value={libraryName} />
          <InfoRow label="Model File" value={metrics.model_path || 'N/A'} />
          <InfoRow label="Scaler File" value={metrics.scaler_path || 'N/A'} />
          <InfoRow label="Training Date" value={metrics.trained_date ? new Date(metrics.trained_date).toLocaleString() : 'N/A'} />
        </div>
      </div>
    </div>
  );
};

// ============================================
// UTILITY COMPONENTS
// ============================================

const StatCard = ({ title, value, subtitle, icon, color }) => (
  <div style={{
    background: 'white',
    borderRadius: '12px',
    boxShadow: '0 4px 6px rgba(0,0,0,0.1)',
    padding: '1.5rem',
    borderTop: `4px solid ${color}`
  }}>
    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'start' }}>
      <div style={{ flex: 1 }}>
        <p style={{ fontSize: '0.875rem', color: '#718096', marginBottom: '0.5rem' }}>{title}</p>
        <p style={{ fontSize: '2rem', fontWeight: 'bold', color: '#1a202c', margin: '0.5rem 0' }}>{value}</p>
        <p style={{ fontSize: '0.75rem', color: '#a0aec0' }}>{subtitle}</p>
      </div>
      <div style={{ fontSize: '2.5rem' }}>{icon}</div>
    </div>
  </div>
);

const MetricCard = ({ title, value, description, color }) => (
  <div style={{
    background: 'white',
    borderRadius: '12px',
    boxShadow: '0 4px 6px rgba(0,0,0,0.1)',
    padding: '1.5rem',
    borderLeft: `6px solid ${color}`
  }}>
    <h3 style={{ fontSize: '0.875rem', color: '#718096', margin: '0 0 0.5rem 0' }}>{title}</h3>
    <p style={{ fontSize: '2rem', fontWeight: 'bold', color: '#1a202c', margin: '0.5rem 0' }}>{value}</p>
    <p style={{ fontSize: '0.75rem', color: '#a0aec0', margin: '0.5rem 0 0 0' }}>{description}</p>
  </div>
);

const ChartCard = ({ title, children }) => (
  <div style={{
    background: 'white',
    borderRadius: '12px',
    boxShadow: '0 4px 6px rgba(0,0,0,0.1)',
    padding: '2rem',
    marginBottom: '2rem'
  }}>
    <h2 style={{ fontSize: '1.5rem', fontWeight: 'bold', color: '#1a202c', marginBottom: '1.5rem' }}>
      {title}
    </h2>
    {children}
  </div>
);

const DataStat = ({ label, value }) => (
  <div>
    <p style={{ fontSize: '0.875rem', color: '#718096', marginBottom: '0.25rem' }}>{label}</p>
    <p style={{ fontSize: '1.25rem', fontWeight: '600', color: '#2d3748' }}>{value}</p>
  </div>
);

const InfoRow = ({ label, value }) => (
  <div style={{ display: 'flex', padding: '0.75rem', background: '#f7fafc', borderRadius: '6px' }}>
    <span style={{ fontWeight: '600', color: '#4a5568', minWidth: '150px' }}>{label}:</span>
    <span style={{ color: '#718096', wordBreak: 'break-all' }}>{value}</span>
  </div>
);

export default AdminDashboard;
