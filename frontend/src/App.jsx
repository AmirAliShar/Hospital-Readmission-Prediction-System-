import React, { useState, useEffect } from 'react';
import Header from './components/Header';
import PredictorView from './components/PredictorView';
import EDAView from './components/EDAView';
import ModelView from './components/ModelView';
import BatchView from './components/BatchView';
import { api } from './services/api';
import { ShieldCheck, HeartPulse, FileText, ExternalLink, Activity } from 'lucide-react';

export default function App() {
  const [activeTab, setActiveTab] = useState('predict');
  const [healthStatus, setHealthStatus] = useState(null);
  const [modelInfo, setModelInfo] = useState(null);

  useEffect(() => {
    async function initSystem() {
      try {
        const health = await api.getHealth();
        setHealthStatus(health);
        const info = await api.getModelInfo();
        setModelInfo(info);
      } catch (err) {
        console.warn("API init error:", err);
      }
    }
    initSystem();
  }, []);

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      {/* Clinical Navbar */}
      <Header
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        healthStatus={healthStatus}
        modelInfo={modelInfo}
      />

      {/* Main Tab View */}
      <main style={{ flex: 1 }}>
        {activeTab === 'predict' && <PredictorView />}
        {activeTab === 'eda' && <EDAView />}
        {activeTab === 'model' && <ModelView />}
        {activeTab === 'batch' && <BatchView />}
      </main>

      {/* Footer */}
      <footer style={{
        marginTop: 40,
        borderTop: '1px solid var(--border-subtle)',
        background: 'rgba(10, 14, 23, 0.9)',
        padding: '20px 24px',
        fontSize: '0.82rem',
        color: 'var(--text-dim)'
      }}>
        <div style={{
          maxWidth: 1400,
          margin: '0 auto',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          flexWrap: 'wrap',
          gap: 16
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <HeartPulse size={18} color="var(--cyan-primary)" />
            <span>
              MedPulseAI • Clinical 30-Day Hospital Readmission Prediction System
            </span>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: 20 }}>
            <a
              href="http://localhost:8000/docs"
              target="_blank"
              rel="noreferrer"
              style={{ color: 'var(--cyan-primary)', textDecoration: 'none', display: 'flex', alignItems: 'center', gap: 4 }}
            >
              <FileText size={14} /> FastAPI OpenAPI Docs <ExternalLink size={12} />
            </a>
            <span>Dataset: UCI 130-US Hospitals (69,987 Unique Patients)</span>
            <span style={{ color: 'var(--text-muted)' }}>HIPAA / Clinical Research Safe</span>
          </div>
        </div>
      </footer>
    </div>
  );
}
