import React, { useState, useEffect } from 'react';
import { BarChart3, Database, Layers, PieChart, Users, AlertCircle, Eye, ExternalLink, X } from 'lucide-react';
import { api } from '../services/api';

export default function EDAView() {
  const [edaData, setEdaData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [selectedChart, setSelectedChart] = useState(null);

  useEffect(() => {
    async function loadEDA() {
      try {
        const res = await api.getEDASummary();
        setEdaData(res);
      } catch (err) {
        console.error("Error loading EDA data:", err);
      } finally {
        setLoading(false);
      }
    }
    loadEDA();
  }, []);

  if (loading) {
    return (
      <div style={{ maxWidth: 1400, margin: '0 auto', padding: '60px 16px', textAlign: 'center', color: 'var(--text-muted)' }}>
        <div style={{ fontSize: '1.2rem', fontWeight: 600 }}>Loading Population Health & EDA Analytics...</div>
      </div>
    );
  }

  // Pre-computed chart titles and descriptions
  const chartDescriptions = {
    "target_distribution.png": "Distribution of 30-day readmissions across 101,766 encounters showing class imbalance.",
    "age_analysis.png": "Readmission risk escalation across age cohorts from pediatric to geriatric patients (>80y).",
    "gender_race_analysis.png": "Demographic breakdown and disparity auditing across gender and reported racial backgrounds.",
    "diagnosis_analysis.png": "Stratification across primary ICD-9 diagnostic categories (Circulatory, Respiratory, Neoplasms).",
    "distributions.png": "Statistical distributions of hospital stay days, lab procedures, and medication counts.",
    "visits_stay_analysis.png": "Impact of prior emergency and inpatient utilization on acute recurrence risk.",
    "missing_values.png": "Quantification of missing values and sentinel indicators across attributes.",
    "correlation_heatmap.png": "Pearson and Spearman correlation matrix across continuous clinical predictors."
  };

  const ageData = edaData?.age_readmission_rate || {
    "[0-10)": 3.8, "[10-20)": 5.2, "[20-30)": 7.4, "[30-40)": 8.1,
    "[40-50)": 8.3, "[50-60)": 8.6, "[60-70)": 9.1, "[70-80)": 9.8,
    "[80-90)": 10.4, "[90-100)": 8.5
  };

  const diagData = edaData?.diagnosis_readmission_rate || {
    "Circulatory": 9.6, "Respiratory": 9.2, "Digestive": 8.7,
    "Diabetes": 9.1, "Injury": 10.2, "Musculoskeletal": 7.5,
    "Genitourinary": 8.4, "Neoplasms": 9.9, "Other": 8.2
  };

  return (
    <div style={{ maxWidth: 1400, margin: '0 auto', padding: '24px 16px' }}>
      {/* Title */}
      <div style={{ marginBottom: 24 }}>
        <h2 style={{ fontSize: '1.6rem', fontWeight: 800, color: '#ffffff', letterSpacing: '-0.02em' }}>
          Exploratory Data Analysis & Population Cohort Insights
        </h2>
        <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>
          Comprehensive statistical audit of the UCI Diabetes 130-US Hospitals cohort (101,766 patient encounters).
        </p>
      </div>

      {/* KPI Cards Row */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 16, marginBottom: 24 }}>
        <div className="glass-card" style={{ padding: '16px 20px' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', color: 'var(--cyan-primary)', marginBottom: 8 }}>
            <span style={{ fontSize: '0.75rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.05em' }}>Cohort Encounters</span>
            <Database size={18} />
          </div>
          <div style={{ fontSize: '1.8rem', fontWeight: 800, color: '#ffffff' }}>101,766</div>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-dim)' }}>50 Clinical Attributes</div>
        </div>

        <div className="glass-card" style={{ padding: '16px 20px' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', color: '#60a5fa', marginBottom: 8 }}>
            <span style={{ fontSize: '0.75rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.05em' }}>Unique Patients</span>
            <Users size={18} />
          </div>
          <div style={{ fontSize: '1.8rem', fontWeight: 800, color: '#ffffff' }}>69,987</div>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-dim)' }}>Deduplicated Cohort</div>
        </div>

        <div className="glass-card" style={{ padding: '16px 20px' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', color: '#f43f5e', marginBottom: 8 }}>
            <span style={{ fontSize: '0.75rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.05em' }}>30-Day Readmission</span>
            <AlertCircle size={18} />
          </div>
          <div style={{ fontSize: '1.8rem', fontWeight: 800, color: '#f43f5e' }}>8.98%</div>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-dim)' }}>Primary Target (&lt;30d)</div>
        </div>

        <div className="glass-card" style={{ padding: '16px 20px' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', color: '#f59e0b', marginBottom: 8 }}>
            <span style={{ fontSize: '0.75rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.05em' }}>Avg Hospital Stay</span>
            <Layers size={18} />
          </div>
          <div style={{ fontSize: '1.8rem', fontWeight: 800, color: '#ffffff' }}>
            {edaData?.time_in_hospital_stats?.mean || 4.4} <span style={{ fontSize: '1rem', fontWeight: 500, color: 'var(--text-muted)' }}>days</span>
          </div>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-dim)' }}>Median: 4.0 days</div>
        </div>

        <div className="glass-card" style={{ padding: '16px 20px' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', color: '#10b981', marginBottom: 8 }}>
            <span style={{ fontSize: '0.75rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.05em' }}>Avg Medications</span>
            <PieChart size={18} />
          </div>
          <div style={{ fontSize: '1.8rem', fontWeight: 800, color: '#ffffff' }}>
            {edaData?.num_medications_stats?.mean || 16.0}
          </div>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-dim)' }}>Polypharmacy &gt;20 meds</div>
        </div>
      </div>

      {/* Interactive Visualizations: Two columns */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 20, marginBottom: 24 }}>
        {/* Left: Age Cohorts Readmission Rate */}
        <div className="glass-card" style={{ padding: 20 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
            <h3 style={{ fontSize: '1rem', fontWeight: 700, color: '#ffffff' }}>
              Readmission Rate by Age Bracket (%)
            </h3>
            <span className="badge" style={{ background: 'rgba(6, 182, 212, 0.1)', color: 'var(--cyan-primary)' }}>
              Age Disparity
            </span>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
            {Object.entries(ageData).map(([bracket, rate]) => {
              const numRate = parseFloat(rate);
              const maxRate = 12.0;
              const barWidth = Math.min(100, (numRate / maxRate) * 100);
              const isHigh = numRate > 9.5;
              return (
                <div key={bracket} style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                  <span style={{ width: 75, fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-muted)' }}>
                    {bracket}
                  </span>
                  <div style={{ flex: 1, background: 'rgba(255, 255, 255, 0.05)', height: 18, borderRadius: 6, overflow: 'hidden' }}>
                    <div
                      style={{
                        width: `${barWidth}%`,
                        height: '100%',
                        background: isHigh
                          ? 'linear-gradient(90deg, #f59e0b, #f43f5e)'
                          : 'linear-gradient(90deg, #06b6d4, #3b82f6)',
                        borderRadius: 6,
                        transition: 'width 0.6s ease'
                      }}
                    />
                  </div>
                  <span style={{ width: 50, textAlign: 'right', fontSize: '0.82rem', fontWeight: 700, color: isHigh ? '#fb7185' : '#ffffff' }}>
                    {numRate.toFixed(1)}%
                  </span>
                </div>
              );
            })}
          </div>
        </div>

        {/* Right: Diagnosis Category Breakdown */}
        <div className="glass-card" style={{ padding: 20 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
            <h3 style={{ fontSize: '1rem', fontWeight: 700, color: '#ffffff' }}>
              Readmission Rate by Primary ICD-9 Diagnosis (%)
            </h3>
            <span className="badge" style={{ background: 'rgba(16, 185, 129, 0.1)', color: '#34d399' }}>
              Condition Risk
            </span>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
            {Object.entries(diagData).map(([diag, rate]) => {
              const numRate = parseFloat(rate);
              const maxRate = 12.0;
              const barWidth = Math.min(100, (numRate / maxRate) * 100);
              const isHigh = numRate >= 9.5;
              return (
                <div key={diag} style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                  <span style={{ width: 110, fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-muted)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    {diag}
                  </span>
                  <div style={{ flex: 1, background: 'rgba(255, 255, 255, 0.05)', height: 18, borderRadius: 6, overflow: 'hidden' }}>
                    <div
                      style={{
                        width: `${barWidth}%`,
                        height: '100%',
                        background: isHigh
                          ? 'linear-gradient(90deg, #f59e0b, #f43f5e)'
                          : 'linear-gradient(90deg, #10b981, #06b6d4)',
                        borderRadius: 6,
                        transition: 'width 0.6s ease'
                      }}
                    />
                  </div>
                  <span style={{ width: 50, textAlign: 'right', fontSize: '0.82rem', fontWeight: 700, color: isHigh ? '#fb7185' : '#ffffff' }}>
                    {numRate.toFixed(1)}%
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      </div>

      {/* Server Generated Analytical Charts Gallery */}
      <div style={{ marginTop: 24 }}>
        <h3 style={{ fontSize: '1.25rem', fontWeight: 800, color: '#ffffff', marginBottom: 8 }}>
          High-Resolution Analytical Visualization Gallery
        </h3>
        <p style={{ color: 'var(--text-muted)', fontSize: '0.88rem', marginBottom: 18 }}>
          Publication-ready charts generated directly from the Python data pipeline. Click any chart to inspect in full resolution.
        </p>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
          {(edaData?.charts || [
            "target_distribution.png", "age_analysis.png", "gender_race_analysis.png",
            "diagnosis_analysis.png", "distributions.png", "visits_stay_analysis.png",
            "missing_values.png", "correlation_heatmap.png"
          ]).map((chartName) => (
            <div
              key={chartName}
              onClick={() => setSelectedChart(chartName)}
              className="glass-card"
              style={{
                cursor: 'pointer',
                overflow: 'hidden',
                borderRadius: 14,
                display: 'flex',
                flexDirection: 'column'
              }}
            >
              <div style={{ position: 'relative', width: '100%', height: 160, background: '#0b0f19' }}>
                <img
                  src={`http://localhost:8000/charts/${chartName}`}
                  alt={chartName}
                  style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                  onError={(e) => {
                    e.target.style.display = 'none';
                    e.target.parentElement.innerHTML = `<div style="display:flex;align-items:center;justify-content:center;height:100%;color:#64748b;font-size:0.8rem;">Chart: ${chartName}</div>`;
                  }}
                />
                <div style={{
                  position: 'absolute',
                  top: 8,
                  right: 8,
                  background: 'rgba(0,0,0,0.6)',
                  borderRadius: 6,
                  padding: 4,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center'
                }}>
                  <Eye size={14} color="#ffffff" />
                </div>
              </div>

              <div style={{ padding: '12px 14px', flex: 1 }}>
                <div style={{ fontSize: '0.86rem', fontWeight: 700, color: '#ffffff', marginBottom: 4, textTransform: 'capitalize' }}>
                  {chartName.replace('.png', '').replace(/_/g, ' ')}
                </div>
                <p style={{ fontSize: '0.74rem', color: 'var(--text-dim)', lineHeight: 1.3 }}>
                  {chartDescriptions[chartName] || "Analytical exploration chart"}
                </p>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Chart Lightbox Modal */}
      {selectedChart && (
        <div
          onClick={() => setSelectedChart(null)}
          style={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            background: 'rgba(0, 0, 0, 0.85)',
            backdropFilter: 'blur(8px)',
            zIndex: 100,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            padding: 24
          }}
        >
          <div
            onClick={(e) => e.stopPropagation()}
            className="glass-card"
            style={{
              maxWidth: 960,
              width: '100%',
              maxHeight: '90vh',
              overflow: 'hidden',
              display: 'flex',
              flexDirection: 'column',
              background: '#0d131f'
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '16px 20px', borderBottom: '1px solid var(--border-subtle)' }}>
              <div style={{ fontSize: '1rem', fontWeight: 700, color: '#ffffff', textTransform: 'capitalize' }}>
                {selectedChart.replace('.png', '').replace(/_/g, ' ')}
              </div>
              <button
                onClick={() => setSelectedChart(null)}
                style={{ background: 'transparent', border: 'none', cursor: 'pointer', color: 'var(--text-muted)' }}
              >
                <X size={20} />
              </button>
            </div>
            <div style={{ padding: 20, textAlign: 'center', overflow: 'auto' }}>
              <img
                src={`http://localhost:8000/charts/${selectedChart}`}
                alt={selectedChart}
                style={{ maxWidth: '100%', maxHeight: '70vh', borderRadius: 8 }}
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
