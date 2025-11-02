// Main App Component with SHAP Analysis Integration
import React, { useState } from 'react';
import ShapAnalysisViz from './ShapAnalysisViz';
import './App.css';

function App() {
  const [activeView, setActiveView] = useState('shap');

  return (
    <div className="App">
      {/* Navigation */}
      <nav className="app-nav">
        <div className="nav-container">
          <div className="nav-brand">
            📚 Library Occupancy System
          </div>
          <div className="nav-links">
            <button
              className={activeView === 'predictions' ? 'nav-link active' : 'nav-link'}
              onClick={() => setActiveView('predictions')}
            >
              📊 Predictions
            </button>
            <button
              className={activeView === 'shap' ? 'nav-link active' : 'nav-link'}
              onClick={() => setActiveView('shap')}
            >
              🔍 SHAP Analysis
            </button>
          </div>
        </div>
      </nav>

      {/* Content */}
      <div className="app-content">
        {activeView === 'predictions' && (
          <div className="view-container">
            <h1>Predictions Dashboard</h1>
            <p className="text-muted">
              Import your existing dashboard component here (react_dashboard.jsx)
            </p>
          </div>
        )}

        {activeView === 'shap' && (
          <ShapAnalysisViz />
        )}
      </div>
    </div>
  );
}

export default App;
