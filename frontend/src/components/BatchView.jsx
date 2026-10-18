import React, { useState } from 'react';
import { Users, Upload, RefreshCw, AlertTriangle, CheckCircle, ShieldAlert, FileText } from 'lucide-react';
import { api } from '../services/api';

const SAMPLE_COHORT = [
  {
    name: "Patient A-101",
    gender: "Female", age: 79, race: "Caucasian", admission_type_id: 1,
    time_in_hospital: 8, num_lab_procedures: 72, num_procedures: 2, num_medications: 25,
    number_diagnoses: 9, diag_1_category: "Circulatory", number_outpatient: 2,
    number_emergency: 3, number_inpatient: 2, diabetesMed: true, medication_changed: true,
    n_med_changes: 2, insulin_changed: true, A1C_abnormal: true, max_glu_abnormal: false
  },
  {
    name: "Patient B-102",
    gender: "Male", age: 48, race: "Caucasian", admission_type_id: 3,
    time_in_hospital: 2, num_lab_procedures: 22, num_procedures: 1, num_medications: 7,
    number_diagnoses: 3, diag_1_category: "Respiratory", number_outpatient: 0,
    number_emergency: 0, number_inpatient: 0, diabetesMed: false, medication_changed: false,
    n_med_changes: 0, insulin_changed: false, A1C_abnormal: false, max_glu_abnormal: false
  },
  {
    name: "Patient C-103",
    gender: "Male", age: 66, race: "AfricanAmerican", admission_type_id: 1,
    time_in_hospital: 5, num_lab_procedures: 54, num_procedures: 1, num_medications: 19,
    number_diagnoses: 7, diag_1_category: "Diabetes", number_outpatient: 1,
    number_emergency: 1, number_inpatient: 1, diabetesMed: true, medication_changed: true,
    n_med_changes: 1, insulin_changed: true, A1C_abnormal: true, max_glu_abnormal: true
  },
  {
    name: "Patient D-104",
    gender: "Female", age: 83, race: "Caucasian", admission_type_id: 1,
    time_in_hospital: 11, num_lab_procedures: 85, num_procedures: 3, num_medications: 32,
    number_diagnoses: 9, diag_1_category: "Neoplasms", number_outpatient: 3,
    number_emergency: 2, number_inpatient: 4, diabetesMed: true, medication_changed: true,
    n_med_changes: 3, insulin_changed: true, A1C_abnormal: false, max_glu_abnormal: false
  },
  {
    name: "Patient E-105",
    gender: "Female", age: 52, race: "Hispanic", admission_type_id: 2,
    time_in_hospital: 3, num_lab_procedures: 35, num_procedures: 0, num_medications: 12,
    number_diagnoses: 5, diag_1_category: "Digestive", number_outpatient: 0,
    number_emergency: 0, number_inpatient: 0, diabetesMed: true, medication_changed: false,
    n_med_changes: 0, insulin_changed: false, A1C_abnormal: false, max_glu_abnormal: false
  }
];

export default function BatchView() {
  const [patients, setPatients] = useState(SAMPLE_COHORT);
  const [batchResults, setBatchResults] = useState(null);
  const [loading, setLoading] = useState(false);

  const runScreening = async () => {
    setLoading(true);
    try {
      const res = await api.batchPredict(patients);
      setBatchResults(res);
    } catch (err) {
      console.error(err);
      alert("Batch screening failed: " + err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ maxWidth: 1400, margin: '0 auto', padding: '24px 16px' }}>
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 16, marginBottom: 24 }}>
        <div>
          <h2 style={{ fontSize: '1.6rem', fontWeight: 800, color: '#ffffff', letterSpacing: '-0.02em' }}>
            Clinical Cohort Screening & Triage Tool
          </h2>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>
            Screen multiple admitted patients simultaneously to allocate transition coordinators and post-discharge interventions.
          </p>
        </div>

        <button
          onClick={runScreening}
          disabled={loading}
          className="btn btn-primary"
          style={{ padding: '10px 20px' }}
        >
          <RefreshCw size={16} className={loading ? "animate-spin" : ""} />
          {loading ? "Stratifying Cohort..." : "Run Cohort Risk Stratification"}
        </button>
      </div>

      {/* Triage Summary Cards */}
      {batchResults && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16, marginBottom: 24 }}>
          <div className="glass-card" style={{ padding: 18 }}>
            <div style={{ fontSize: '0.75rem', fontWeight: 700, color: 'var(--text-dim)', textTransform: 'uppercase' }}>
              Total Screened
            </div>
            <div style={{ fontSize: '1.8rem', fontWeight: 800, color: '#ffffff' }}>
              {batchResults.total_patients}
            </div>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-dim)' }}>Patients Evaluated</div>
          </div>

          <div className="glass-card" style={{ padding: 18, borderLeft: '4px solid #f43f5e' }}>
            <div style={{ fontSize: '0.75rem', fontWeight: 700, color: '#f43f5e', textTransform: 'uppercase' }}>
              High / Critical Risk
            </div>
            <div style={{ fontSize: '1.8rem', fontWeight: 800, color: '#fb7185' }}>
              {batchResults.high_risk_count}
            </div>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-dim)' }}>Requires Transition Care</div>
          </div>

          <div className="glass-card" style={{ padding: 18, borderLeft: '4px solid #f59e0b' }}>
            <div style={{ fontSize: '0.75rem', fontWeight: 700, color: '#f59e0b', textTransform: 'uppercase' }}>
              Moderate Risk
            </div>
            <div style={{ fontSize: '1.8rem', fontWeight: 800, color: '#fbbf24' }}>
              {batchResults.moderate_risk_count}
            </div>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-dim)' }}>7-Day Outpatient Follow-up</div>
          </div>

          <div className="glass-card" style={{ padding: 18, borderLeft: '4px solid #10b981' }}>
            <div style={{ fontSize: '0.75rem', fontWeight: 700, color: '#10b981', textTransform: 'uppercase' }}>
              Low Risk
            </div>
            <div style={{ fontSize: '1.8rem', fontWeight: 800, color: '#34d399' }}>
              {batchResults.low_risk_count}
            </div>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-dim)' }}>Standard Discharge</div>
          </div>
        </div>
      )}

      {/* Cohort Table */}
      <div className="glass-card" style={{ padding: 20 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
          <h3 style={{ fontSize: '1rem', fontWeight: 700, color: '#ffffff' }}>
            Inpatient Cohort Roster ({patients.length} Patients)
          </h3>
          <span style={{ fontSize: '0.75rem', color: 'var(--text-dim)' }}>Live Clinical Triage Queue</span>
        </div>

        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.84rem' }}>
            <thead>
              <tr style={{ borderBottom: '1px solid var(--border-subtle)', textAlign: 'left', color: 'var(--text-dim)', textTransform: 'uppercase', fontSize: '0.72rem' }}>
                <th style={{ padding: '10px 12px' }}>Patient</th>
                <th style={{ padding: '10px 12px' }}>Demographics</th>
                <th style={{ padding: '10px 12px' }}>Stay / Diag</th>
                <th style={{ padding: '10px 12px' }}>Prior Inpatient</th>
                <th style={{ padding: '10px 12px' }}>Risk Score</th>
                <th style={{ padding: '10px 12px' }}>Risk Tier</th>
                <th style={{ padding: '10px 12px' }}>Action Plan</th>
              </tr>
            </thead>
            <tbody>
              {patients.map((p, idx) => {
                const res = batchResults?.predictions?.[idx];
                const riskPct = res?.readmission_risk_percent;
                const riskLevel = res?.risk_level || 'Pending';

                return (
                  <tr key={idx} style={{ borderBottom: '1px solid rgba(255, 255, 255, 0.04)' }}>
                    <td style={{ padding: '12px', fontWeight: 700, color: '#ffffff' }}>
                      {p.name}
                    </td>
                    <td style={{ padding: '12px', color: 'var(--text-muted)' }}>
                      {p.age}y • {p.gender} • {p.race}
                    </td>
                    <td style={{ padding: '12px', color: 'var(--text-main)' }}>
                      {p.time_in_hospital}d • {p.diag_1_category}
                    </td>
                    <td style={{ padding: '12px' }}>
                      <span style={{ fontWeight: 700, color: p.number_inpatient > 0 ? '#f43f5e' : 'var(--text-dim)' }}>
                        {p.number_inpatient} visits
                      </span>
                    </td>
                    <td style={{ padding: '12px', fontWeight: 700 }}>
                      {riskPct !== undefined ? (
                        <span style={{
                          color: riskPct >= 50 ? '#f43f5e' : riskPct >= 25 ? '#f59e0b' : '#10b981',
                          fontSize: '0.95rem'
                        }}>
                          {riskPct.toFixed(1)}%
                        </span>
                      ) : (
                        <span style={{ color: 'var(--text-dim)' }}>—</span>
                      )}
                    </td>
                    <td style={{ padding: '12px' }}>
                      {res ? (
                        <span className={`badge badge-${riskLevel.toLowerCase()}`}>
                          {riskLevel}
                        </span>
                      ) : (
                        <span style={{ fontSize: '0.75rem', color: 'var(--text-dim)' }}>Ready to run</span>
                      )}
                    </td>
                    <td style={{ padding: '12px', fontSize: '0.78rem', color: 'var(--text-muted)' }}>
                      {res?.clinical_recommendations?.[0]?.action || "Run screening to determine"}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
