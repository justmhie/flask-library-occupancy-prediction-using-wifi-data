import React, { useState } from 'react';
import SimpleDashboard from './SimpleDashboard';
import AdminDashboard from './AdminDashboard';
import ShapAnalysisViz from './ShapAnalysisViz';

function App() {
  const [currentView, setCurrentView] = useState('admin'); // 'admin', 'user', or 'shap'

  return (
    <div>
      {/* Dashboard Switcher */}
      <div style={{
        position: 'fixed',
        top: '1rem',
        right: '1rem',
        zIndex: 1000,
        display: 'flex',
        gap: '0.5rem'
      }}>
        <button
          onClick={() => setCurrentView('admin')}
          style={{
            padding: '0.75rem 1.5rem',
            borderRadius: '8px',
            border: currentView === 'admin' ? '2px solid #667eea' : 'none',
            background: currentView === 'admin' ? '#eff6ff' : 'white',
            boxShadow: '0 4px 6px rgba(0,0,0,0.1)',
            cursor: 'pointer',
            fontWeight: '600',
            fontSize: '0.9rem',
            color: '#667eea'
          }}
        >
          🎛️ Admin
        </button>
        <button
          onClick={() => setCurrentView('user')}
          style={{
            padding: '0.75rem 1.5rem',
            borderRadius: '8px',
            border: currentView === 'user' ? '2px solid #667eea' : 'none',
            background: currentView === 'user' ? '#eff6ff' : 'white',
            boxShadow: '0 4px 6px rgba(0,0,0,0.1)',
            cursor: 'pointer',
            fontWeight: '600',
            fontSize: '0.9rem',
            color: '#667eea'
          }}
        >
          👤 User
        </button>
        <button
          onClick={() => setCurrentView('shap')}
          style={{
            padding: '0.75rem 1.5rem',
            borderRadius: '8px',
            border: currentView === 'shap' ? '2px solid #667eea' : 'none',
            background: currentView === 'shap' ? '#eff6ff' : 'white',
            boxShadow: '0 4px 6px rgba(0,0,0,0.1)',
            cursor: 'pointer',
            fontWeight: '600',
            fontSize: '0.9rem',
            color: '#667eea'
          }}
        >
          🔍 SHAP
        </button>
      </div>

      {/* Render appropriate dashboard */}
      {currentView === 'admin' && <AdminDashboard />}
      {currentView === 'user' && <SimpleDashboard />}
      {currentView === 'shap' && <ShapAnalysisViz />}
    </div>
  );
}

export default App;
