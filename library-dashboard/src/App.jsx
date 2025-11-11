// Main App Component with Admin Panel and SHAP Analysis Integration
import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import LibraryOccupancyDashboard from './react_dashboard';
import AdminLayout from './components/admin/AdminLayout';
import AdminDashboard from './components/admin/AdminDashboard';
import DataUpload from './components/admin/DataUpload';
import ShapAnalysisViz from './ShapAnalysisViz';
import './App.css';

function App() {
  return (
    <Router>
      <Routes>
        {/* Main Dashboard */}
        <Route path="/" element={<LibraryOccupancyDashboard />} />
        
        {/* SHAP Analysis */}
        <Route path="/shap" element={<ShapAnalysisViz />} />

        {/* Admin Routes */}
        <Route path="/admin" element={<AdminLayout><AdminDashboard /></AdminLayout>} />
        <Route path="/admin/data" element={<AdminLayout><DataUpload /></AdminLayout>} />
      </Routes>
    </Router>
  );
}

export default App;
