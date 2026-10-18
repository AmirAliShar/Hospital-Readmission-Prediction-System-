import React from 'react';
import { Activity, BarChart2, ShieldCheck, Users, Server, CheckCircle2, AlertCircle } from 'lucide-react';

export default function Header({ activeTab, setActiveTab, healthStatus, modelInfo }) {
  const isOnline = healthStatus?.status === 'healthy';

  const navItems = [
    { id: 'predict', label: 'Patient Predictor', icon: Activity },
    { id: 'eda', label: 'Exploratory Analysis', icon: BarChart2 },
    { id: 'model', label: 'Model Validation', icon: ShieldCheck },
    { id: 'batch', label: 'Cohort Screener', icon: Users },
  ];

  return (
    <header style={{
      borderBottom: '1px solid var(--border-subtle)',
      background: 'rgba(10, 14, 23, 0.85)',
      backdropFilter: 'blur(20px)',
      position: 'sticky',
      top: 0,
      zIndex: 50,
      padding: '12px 24px',
    }}>
      <div style={{
        maxWidth: 1400,
        margin: '0 auto',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        flexWrap: 'wrap',
        gap: 16
      }}>
        {/* Brand */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
          <div style={{
            width: 44,
            height: 44,
            borderRadius: 12,
            background: 'linear-gradient(135deg, #06b6d4, #3b82f6)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            boxShadow: '0 0 20px rgba(6, 182, 212, 0.4)'
          }}>
            <Activity size={24} color="#ffffff" />
          </div>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <h1 style={{ fontSize: '1.25rem', fontWeight: 800, letterSpacing: '-0.02em', color: '#ffffff' }}>
                MedPulse<span style={{ color: 'var(--cyan-primary)' }}>AI</span>
              </h1>
              <span style={{
                background: 'rgba(6, 182, 212, 0.15)',
                color: 'var(--cyan-primary)',
                fontSize: '0.68rem',
                fontWeight: 700,
                padding: '2px 8px',
                borderRadius: 6,
                border: '1px solid rgba(6, 182, 212, 0.3)'
              }}>
                v{modelInfo?.model_version || '2.0'}
              </span>
            </div>
            <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>
              Clinical 30-Day Hospital Readmission Risk Intelligence
            </p>
          </div>
        </div>

        {/* Navigation Tabs */}
        <nav style={{ display: 'flex', gap: 6, background: 'rgba(17, 24, 39, 0.7)', padding: 4, borderRadius: 12, border: '1px solid var(--border-subtle)' }}>
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = activeTab === item.id;
            return (
              <button
                key={item.id}
                onClick={() => setActiveTab(item.id)}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 8,
                  padding: '8px 16px',
                  borderRadius: 8,
                  fontSize: '0.86rem',
                  fontWeight: 600,
                  cursor: 'pointer',
                  border: 'none',
                  transition: 'all 0.2s ease',
                  background: isActive ? 'linear-gradient(135deg, #06b6d4, #2563eb)' : 'transparent',
                  color: isActive ? '#ffffff' : 'var(--text-muted)',
                  boxShadow: isActive ? '0 4px 12px rgba(6, 182, 212, 0.3)' : 'none'
                }}
              >
                <Icon size={16} />
                <span>{item.label}</span>
              </button>
            );
          })}
        </nav>

        {/* System Status Indicators */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: 6,
            padding: '5px 12px',
            borderRadius: 20,
            background: isOnline ? 'rgba(16, 185, 129, 0.1)' : 'rgba(244, 63, 94, 0.1)',
            border: `1px solid ${isOnline ? 'rgba(16, 185, 129, 0.3)' : 'rgba(244, 63, 94, 0.3)'}`
          }}>
            <span style={{
              width: 8,
              height: 8,
              borderRadius: '50%',
              backgroundColor: isOnline ? '#10b981' : '#f43f5e',
              boxShadow: isOnline ? '0 0 8px #10b981' : '0 0 8px #f43f5e'
            }} />
            <span style={{ fontSize: '0.75rem', fontWeight: 600, color: isOnline ? '#34d399' : '#fb7185' }}>
              {isOnline ? 'FastAPI Online' : 'Connecting API...'}
            </span>
          </div>

          <div style={{
            fontSize: '0.75rem',
            color: 'var(--text-dim)',
            background: 'rgba(255, 255, 255, 0.03)',
            padding: '5px 10px',
            borderRadius: 8,
            border: '1px solid var(--border-subtle)'
          }}>
            Model: <strong style={{ color: 'var(--text-main)' }}>{modelInfo?.model_name || 'Logistic Regression'}</strong>
          </div>
        </div>
      </div>
    </header>
  );
}
