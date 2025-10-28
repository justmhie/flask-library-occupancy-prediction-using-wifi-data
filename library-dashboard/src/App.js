import React, { useState } from 'react';
import SimpleDashboard from './SimpleDashboard';
import AdminDashboard from './AdminDashboard';

function App() {
  const [isAdmin, setIsAdmin] = useState(true); // Set to true for admin mode

  return (
    <div>
      {/* Dashboard Switcher */}
      <div style={{
        position: 'fixed',
        top: '1rem',
        right: '1rem',
        zIndex: 1000
      }}>
        <button
          onClick={() => setIsAdmin(!isAdmin)}
          style={{
            padding: '0.75rem 1.5rem',
            borderRadius: '8px',
            border: 'none',
            background: 'white',
            boxShadow: '0 4px 6px rgba(0,0,0,0.1)',
            cursor: 'pointer',
            fontWeight: '600',
            fontSize: '0.9rem',
            color: '#667eea'
          }}
        >
          {isAdmin ? '👤 Switch to User View' : '🎛️ Switch to Admin View'}
        </button>
      </div>

      {/* Render appropriate dashboard */}
      {isAdmin ? <AdminDashboard /> : <SimpleDashboard />}
    </div>
  );
}

export default App;
