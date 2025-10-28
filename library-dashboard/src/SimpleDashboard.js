import React, { useState, useEffect } from 'react';
import {
  LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer, AreaChart, Area
} from 'recharts';

const SimpleDashboard = () => {
  const [selectedLibrary, setSelectedLibrary] = useState('all');
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [lastUpdate, setLastUpdate] = useState(new Date());
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [error, setError] = useState(null);

  // Library locations
  const libraries = [
    { id: 'all', name: 'All Libraries' },
    { id: 'miguel_pro', name: 'Miguel Pro' },
    { id: 'gisbert_2nd', name: 'Gisbert 2nd Floor' },
    { id: 'american_corner', name: 'American Corner' },
    { id: 'gisbert_3rd', name: 'Gisbert 3rd Floor' },
    { id: 'gisbert_4th', name: 'Gisbert 4th Floor' },
    { id: 'gisbert_5th', name: 'Gisbert 5th Floor' }
  ];

  // Fetch predictions from API
  const fetchPredictions = async () => {
    try {
      setLoading(true);
      setError(null);
      // Replace with your actual API endpoint
      const response = await fetch('http://localhost:5000/api/predictions');

      if (!response.ok) {
        throw new Error(`API returned ${response.status}`);
      }

      const jsonData = await response.json();
      setData(jsonData);
      setLastUpdate(new Date());
    } catch (error) {
      console.error('Error fetching predictions:', error);
      setError(error.message);
      // Use mock data for development
      setData(getMockData());
    } finally {
      setLoading(false);
    }
  };

  // Mock data for development/testing
  const getMockData = () => ({
    overall: {
      current: 245,
      predicted: 268,
      avg24h: 232,
      capacity: 0.62,
      trend: 'up',
      hourlyData: Array.from({ length: 24 }, (_, i) => ({
        time: `${i}:00`,
        users: Math.floor(Math.random() * 100) + 150
      })),
      forecast: Array.from({ length: 6 }, (_, i) => ({
        hour: `+${i+1}h`,
        predicted: Math.floor(Math.random() * 50) + 240,
        lower: Math.floor(Math.random() * 30) + 220,
        upper: Math.floor(Math.random() * 30) + 270
      })),
      dailyPattern: Array.from({ length: 24 }, (_, i) => ({
        hour: `${i}:00`,
        avgUsers: Math.floor(Math.random() * 80) + 100
      }))
    },
    libraries: {
      miguel_pro: {
        current: 45,
        predicted: 48,
        avg24h: 42,
        capacity: 0.45
      }
      // Add more libraries as needed
    }
  });

  // Auto-refresh every 60 seconds
  useEffect(() => {
    fetchPredictions();

    if (autoRefresh) {
      const interval = setInterval(fetchPredictions, 60000);
      return () => clearInterval(interval);
    }
  }, [autoRefresh]);

  // Get data for selected library
  const getLibraryData = () => {
    if (!data) return null;
    if (!data.libraries) return null;

    // Backend returns data in libraries.all for overall
    if (selectedLibrary === 'all') {
      return data.libraries.all || data.overall;
    }
    return data.libraries[selectedLibrary] || data.libraries.all;
  };

  const libraryData = getLibraryData();

  if (loading && !data) {
    return (
      <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh' }}>
        <div style={{ textAlign: 'center' }}>
          <div style={{ fontSize: '2rem' }}>⏳</div>
          <p style={{ marginTop: '1rem', color: '#666' }}>Loading predictions...</p>
        </div>
      </div>
    );
  }

  return (
    <div style={{ minHeight: '100vh', background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)', padding: '2rem' }}>
      <div style={{ maxWidth: '1400px', margin: '0 auto' }}>

        {/* Header */}
        <div style={{
          background: 'white',
          borderRadius: '12px',
          boxShadow: '0 4px 6px rgba(0,0,0,0.1)',
          padding: '2rem',
          marginBottom: '2rem'
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap' }}>
            <div>
              <h1 style={{ fontSize: '2rem', fontWeight: 'bold', color: '#1a202c', margin: 0 }}>
                📊 Library Occupancy Dashboard
              </h1>
              <p style={{ color: '#718096', marginTop: '0.5rem' }}>
                Real-time predictions and analytics
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

          {/* Library Selector */}
          <div style={{ marginTop: '1.5rem' }}>
            <label style={{ fontSize: '0.875rem', fontWeight: '600', color: '#4a5568', display: 'block', marginBottom: '0.5rem' }}>
              Select Library:
            </label>
            <select
              value={selectedLibrary}
              onChange={(e) => setSelectedLibrary(e.target.value)}
              style={{
                width: '100%',
                maxWidth: '400px',
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
        </div>

        {/* Error Alert */}
        {error && (
          <div style={{
            background: '#fff5f5',
            border: '1px solid #feb2b2',
            borderRadius: '8px',
            padding: '1rem',
            marginBottom: '1rem',
            color: '#c53030'
          }}>
            ⚠️ Using demo data. Backend API not available: {error}
          </div>
        )}

        {/* Stats Cards */}
        {libraryData && (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: '1rem', marginBottom: '2rem' }}>
            <StatCard
              title="Current Users"
              value={libraryData.current || 0}
              subtitle="Right now"
              icon="👥"
              color="#667eea"
            />
            <StatCard
              title="Predicted"
              value={libraryData.predicted || 0}
              subtitle="Next hour"
              icon="🔮"
              color="#764ba2"
            />
            <StatCard
              title="24h Average"
              value={libraryData.avg_24h || libraryData.avg24h || 0}
              subtitle="Average users"
              icon="📊"
              color="#f093fb"
            />
            <StatCard
              title="Capacity"
              value={libraryData.max_capacity ? `${Math.round((libraryData.current / libraryData.max_capacity) * 100)}%` : 'N/A'}
              subtitle="Of maximum"
              icon="📈"
              color="#4facfe"
            />
          </div>
        )}

        {/* Charts */}
        {libraryData && (libraryData.hourly_data || libraryData.hourlyData) && (
          <div style={{
            background: 'white',
            borderRadius: '12px',
            boxShadow: '0 4px 6px rgba(0,0,0,0.1)',
            padding: '2rem',
            marginBottom: '2rem'
          }}>
            <h2 style={{ fontSize: '1.5rem', fontWeight: 'bold', color: '#1a202c', marginBottom: '1.5rem' }}>
              📈 Hourly Occupancy Trend (Last 24 Hours)
            </h2>
            <ResponsiveContainer width="100%" height={300}>
              <AreaChart data={libraryData.hourly_data || libraryData.hourlyData}>
                <defs>
                  <linearGradient id="colorUsers" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#667eea" stopOpacity={0.8}/>
                    <stop offset="95%" stopColor="#667eea" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="time" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Area
                  type="monotone"
                  dataKey="actual"
                  stroke="#667eea"
                  fillOpacity={1}
                  fill="url(#colorUsers)"
                  name="Actual Users"
                />
                <Area
                  type="monotone"
                  dataKey="predicted"
                  stroke="#f093fb"
                  strokeDasharray="5 5"
                  fillOpacity={0}
                  name="Predicted"
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        )}

        {/* Daily Pattern */}
        {libraryData && (libraryData.daily_pattern || libraryData.dailyPattern) && (
          <div style={{
            background: 'white',
            borderRadius: '12px',
            boxShadow: '0 4px 6px rgba(0,0,0,0.1)',
            padding: '2rem',
            marginBottom: '2rem'
          }}>
            <h2 style={{ fontSize: '1.5rem', fontWeight: 'bold', color: '#1a202c', marginBottom: '1.5rem' }}>
              📊 Daily Usage Pattern (Average by Hour)
            </h2>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={libraryData.daily_pattern || libraryData.dailyPattern}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="hour" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Bar dataKey="average" fill="#764ba2" name="Average Users" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}

        {/* Forecast */}
        {libraryData && (libraryData.next_hours || libraryData.forecast) && (
          <div style={{
            background: 'white',
            borderRadius: '12px',
            boxShadow: '0 4px 6px rgba(0,0,0,0.1)',
            padding: '2rem'
          }}>
            <h2 style={{ fontSize: '1.5rem', fontWeight: 'bold', color: '#1a202c', marginBottom: '1.5rem' }}>
              🔮 Next 6 Hours Forecast
            </h2>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={libraryData.next_hours || libraryData.forecast}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="time" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey="predicted" stroke="#667eea" strokeWidth={3} name="Predicted" />
                <Line type="monotone" dataKey="confidence_lower" stroke="#cbd5e0" strokeDasharray="5 5" name="Lower Bound" />
                <Line type="monotone" dataKey="confidence_upper" stroke="#cbd5e0" strokeDasharray="5 5" name="Upper Bound" />
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}

      </div>
    </div>
  );
};

// Stat Card Component
const StatCard = ({ title, value, subtitle, icon, color }) => (
  <div style={{
    background: 'white',
    borderRadius: '12px',
    boxShadow: '0 4px 6px rgba(0,0,0,0.1)',
    padding: '1.5rem',
    borderTop: `4px solid ${color}`
  }}>
    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'start' }}>
      <div>
        <p style={{ fontSize: '0.875rem', color: '#718096', marginBottom: '0.5rem' }}>{title}</p>
        <p style={{ fontSize: '2rem', fontWeight: 'bold', color: '#1a202c', margin: '0.5rem 0' }}>{value}</p>
        <p style={{ fontSize: '0.75rem', color: '#a0aec0' }}>{subtitle}</p>
      </div>
      <div style={{ fontSize: '2rem' }}>{icon}</div>
    </div>
  </div>
);

export default SimpleDashboard;
