// Modern React Dashboard for Library Occupancy Predictions
// Shows graphs and users for each library with dropdown selection

import React, { useState, useEffect } from 'react';
import { 
  LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, 
  Tooltip, Legend, ResponsiveContainer, Area, AreaChart 
} from 'recharts';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Alert, AlertDescription } from '@/components/ui/alert';

const LibraryOccupancyDashboard = () => {
  const [selectedLibrary, setSelectedLibrary] = useState('all');
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [lastUpdate, setLastUpdate] = useState(new Date());
  const [autoRefresh, setAutoRefresh] = useState(true);

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
      // Replace with your actual API endpoint
      const response = await fetch('/api/predictions');
      const jsonData = await response.json();
      setData(jsonData);
      setLastUpdate(new Date());
    } catch (error) {
      console.error('Error fetching predictions:', error);
    } finally {
      setLoading(false);
    }
  };

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
    if (selectedLibrary === 'all') return data.overall;
    return data.libraries[selectedLibrary];
  };

  const libraryData = getLibraryData();

  if (loading && !data) {
    return (
      <div className="flex items-center justify-center h-screen">
        <div className="text-center">
          <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-blue-500 mx-auto"></div>
          <p className="mt-4 text-gray-600">Loading predictions...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
          <div className="flex justify-between items-center">
            <div>
              <h1 className="text-3xl font-bold text-gray-800">
                📊 Library Occupancy Dashboard
              </h1>
              <p className="text-gray-600 mt-2">
                Real-time predictions and analytics
              </p>
            </div>
            <div className="text-right">
              <p className="text-sm text-gray-500">Last Updated</p>
              <p className="text-lg font-semibold text-gray-700">
                {lastUpdate.toLocaleTimeString()}
              </p>
              <button
                onClick={() => setAutoRefresh(!autoRefresh)}
                className={`mt-2 px-4 py-2 rounded-lg text-sm font-medium ${
                  autoRefresh 
                    ? 'bg-green-500 text-white' 
                    : 'bg-gray-300 text-gray-700'
                }`}
              >
                {autoRefresh ? '🔄 Auto-refresh ON' : '⏸️ Auto-refresh OFF'}
              </button>
            </div>
          </div>

          {/* Library Selector */}
          <div className="mt-6">
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Select Library
            </label>
            <Select value={selectedLibrary} onValueChange={setSelectedLibrary}>
              <SelectTrigger className="w-full md:w-96">
                <SelectValue placeholder="Choose a library" />
              </SelectTrigger>
              <SelectContent>
                {libraries.map(lib => (
                  <SelectItem key={lib.id} value={lib.id}>
                    {lib.name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </div>

        {libraryData && (
          <>
            {/* Current Stats Cards */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-6">
              <Card className="bg-gradient-to-br from-blue-500 to-blue-600 text-white">
                <CardHeader>
                  <CardTitle className="text-lg">Current Users</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-4xl font-bold">{libraryData.current}</p>
                  <p className="text-sm opacity-90 mt-2">Right now</p>
                </CardContent>
              </Card>

              <Card className="bg-gradient-to-br from-purple-500 to-purple-600 text-white">
                <CardHeader>
                  <CardTitle className="text-lg">Predicted (Next Hour)</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-4xl font-bold">{libraryData.predicted}</p>
                  <p className="text-sm opacity-90 mt-2">
                    {libraryData.change >= 0 ? '📈' : '📉'} 
                    {libraryData.change >= 0 ? '+' : ''}{libraryData.change} users
                  </p>
                </CardContent>
              </Card>

              <Card className="bg-gradient-to-br from-green-500 to-green-600 text-white">
                <CardHeader>
                  <CardTitle className="text-lg">24h Average</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-4xl font-bold">{libraryData.avg_24h}</p>
                  <p className="text-sm opacity-90 mt-2">Last 24 hours</p>
                </CardContent>
              </Card>

              <Card className="bg-gradient-to-br from-orange-500 to-orange-600 text-white">
                <CardHeader>
                  <CardTitle className="text-lg">Capacity</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-4xl font-bold">
                    {Math.round((libraryData.current / libraryData.max_capacity) * 100)}%
                  </p>
                  <p className="text-sm opacity-90 mt-2">
                    {libraryData.max_capacity} max
                  </p>
                </CardContent>
              </Card>
            </div>

            {/* Hourly Trend Chart */}
            <Card className="mb-6">
              <CardHeader>
                <CardTitle>📈 Hourly Occupancy Trend</CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={400}>
                  <AreaChart data={libraryData.hourly_data}>
                    <defs>
                      <linearGradient id="colorUsers" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.8}/>
                        <stop offset="95%" stopColor="#3b82f6" stopOpacity={0}/>
                      </linearGradient>
                      <linearGradient id="colorPredicted" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#8b5cf6" stopOpacity={0.8}/>
                        <stop offset="95%" stopColor="#8b5cf6" stopOpacity={0}/>
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis 
                      dataKey="time" 
                      label={{ value: 'Time', position: 'insideBottom', offset: -5 }}
                    />
                    <YAxis 
                      label={{ value: 'Number of Users', angle: -90, position: 'insideLeft' }}
                    />
                    <Tooltip />
                    <Legend />
                    <Area 
                      type="monotone" 
                      dataKey="actual" 
                      stroke="#3b82f6" 
                      fillOpacity={1} 
                      fill="url(#colorUsers)"
                      name="Actual Users"
                    />
                    <Area 
                      type="monotone" 
                      dataKey="predicted" 
                      stroke="#8b5cf6" 
                      fillOpacity={1} 
                      fill="url(#colorPredicted)"
                      name="Predicted"
                      strokeDasharray="5 5"
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            {/* Daily Pattern Chart */}
            <Card className="mb-6">
              <CardHeader>
                <CardTitle>📊 Daily Usage Pattern (Last 7 Days)</CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={libraryData.daily_pattern}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="hour" label={{ value: 'Hour of Day', position: 'insideBottom', offset: -5 }} />
                    <YAxis label={{ value: 'Average Users', angle: -90, position: 'insideLeft' }} />
                    <Tooltip />
                    <Legend />
                    <Bar dataKey="average" fill="#3b82f6" name="Average Users" />
                    <Bar dataKey="peak" fill="#f59e0b" name="Peak" />
                  </BarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            {/* Multi-Hour Predictions */}
            <Card className="mb-6">
              <CardHeader>
                <CardTitle>🔮 Next 6 Hours Forecast</CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={libraryData.next_hours}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis 
                      dataKey="time" 
                      label={{ value: 'Time', position: 'insideBottom', offset: -5 }}
                    />
                    <YAxis 
                      label={{ value: 'Predicted Users', angle: -90, position: 'insideLeft' }}
                    />
                    <Tooltip />
                    <Legend />
                    <Line 
                      type="monotone" 
                      dataKey="predicted" 
                      stroke="#8b5cf6" 
                      strokeWidth={3}
                      dot={{ r: 6 }}
                      name="Forecast"
                    />
                    <Line 
                      type="monotone" 
                      dataKey="confidence_upper" 
                      stroke="#d8b4fe" 
                      strokeDasharray="3 3"
                      name="Upper Bound"
                    />
                    <Line 
                      type="monotone" 
                      dataKey="confidence_lower" 
                      stroke="#d8b4fe" 
                      strokeDasharray="3 3"
                      name="Lower Bound"
                    />
                  </LineChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            {/* Status Alert */}
            {libraryData.current > libraryData.max_capacity * 0.8 && (
              <Alert className="mb-6 bg-red-50 border-red-200">
                <AlertDescription className="text-red-800">
                  ⚠️ High occupancy alert! Library is at {
                    Math.round((libraryData.current / libraryData.max_capacity) * 100)
                  }% capacity. Consider directing users to other locations.
                </AlertDescription>
              </Alert>
            )}

            {/* Additional Stats */}
            <Card>
              <CardHeader>
                <CardTitle>📊 Additional Statistics</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div className="text-center p-4 bg-gray-50 rounded-lg">
                    <p className="text-gray-600 text-sm">Peak Today</p>
                    <p className="text-2xl font-bold text-gray-800">
                      {libraryData.peak_today}
                    </p>
                    <p className="text-xs text-gray-500">
                      at {libraryData.peak_time}
                    </p>
                  </div>
                  <div className="text-center p-4 bg-gray-50 rounded-lg">
                    <p className="text-gray-600 text-sm">7-Day Average</p>
                    <p className="text-2xl font-bold text-gray-800">
                      {libraryData.avg_7d}
                    </p>
                  </div>
                  <div className="text-center p-4 bg-gray-50 rounded-lg">
                    <p className="text-gray-600 text-sm">Trend</p>
                    <p className="text-2xl font-bold text-gray-800">
                      {libraryData.trend > 0 ? '📈' : libraryData.trend < 0 ? '📉' : '➡️'}
                    </p>
                    <p className="text-xs text-gray-500">
                      {libraryData.trend > 0 ? 'Increasing' : 
                       libraryData.trend < 0 ? 'Decreasing' : 'Stable'}
                    </p>
                  </div>
                  <div className="text-center p-4 bg-gray-50 rounded-lg">
                    <p className="text-gray-600 text-sm">Accuracy</p>
                    <p className="text-2xl font-bold text-gray-800">
                      {libraryData.model_accuracy}%
                    </p>
                    <p className="text-xs text-gray-500">Model R²</p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </>
        )}

        {/* Footer */}
        <div className="mt-8 text-center text-gray-600 text-sm">
          <p>Predictions updated every 60 seconds • Powered by ML Models</p>
          <div className="mt-1 space-x-4">
            <button 
              onClick={fetchPredictions}
              className="text-blue-600 hover:text-blue-800 font-medium"
            >
              🔄 Refresh Now
            </button>
            <a 
              href="/admin"
              className="text-gray-600 hover:text-gray-800 font-medium"
            >
              ⚙️ Admin Panel
            </a>
          </div>
        </div>
      </div>
    </div>
  );
};

export default LibraryOccupancyDashboard;
