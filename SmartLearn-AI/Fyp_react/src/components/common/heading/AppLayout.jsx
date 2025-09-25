import React, { useState } from 'react';
import { Outlet, useNavigate } from 'react-router-dom';

function Sidebar({ expanded, onToggle }) {
  const navigate = useNavigate();
  return (
    <div
      className="sidebar"
      style={{
        width: expanded ? 250 : 60,
        minHeight: '100vh',
        background: '#fff',
        boxShadow: '2px 0 8px rgba(0,0,0,0.04)',
        padding: expanded ? '1.5rem 0' : '1.5rem 0.5rem',
        display: 'flex',
        flexDirection: 'column',
        gap: '1.5rem',
        alignItems: expanded ? 'flex-start' : 'center',
        transition: 'width 0.2s',
        zIndex: 30
      }}
    >
      <div style={{ padding: expanded ? '0 1.5rem' : 0, width: '100%' }}>
        <div style={{ margin: '1rem 0', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 10 }} onClick={() => navigate('/input-selection')}>
          <span role="img" aria-label="chat">🗨️</span>
          {expanded && 'New Chat'}
        </div>
        <div style={{ margin: '1rem 0', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 10 }} onClick={() => navigate('/history')}>
          <span role="img" aria-label="history">⏳</span>
          {expanded && 'History'}
        </div>
        <div style={{ margin: '1rem 0', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 10 }} onClick={() => navigate('/step1')}>
          <span role="img" aria-label="step1">1️⃣</span>
          {expanded && 'Step 1'}
        </div>
        <div style={{ margin: '1rem 0', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 10 }} onClick={() => navigate('/step2')}>
          <span role="img" aria-label="step2">2️⃣</span>
          {expanded && 'Step 2'}
        </div>
        <div style={{ margin: '1rem 0', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 10 }} onClick={() => navigate('/processing')}>
          <span role="img" aria-label="processing">🖥️</span>
          {expanded && 'Processing'}
        </div>
      </div>
    </div>
  );
}

const AppLayout = () => {
  const [sidebarExpanded, setSidebarExpanded] = useState(true);
  return (
    <div style={{ display: 'flex', minHeight: '100vh' }}>
      <Sidebar expanded={sidebarExpanded} onToggle={() => setSidebarExpanded((prev) => !prev)} />
      <div style={{ flex: 1 }}>
        <Outlet />
      </div>
    </div>
  );
};

export default AppLayout; 