import React, { useState, useEffect } from 'react';
import { 
  Activity, AlertTriangle, CheckCircle, ChevronRight, User, Clock, 
  Pill, FileText, Stethoscope, RefreshCw, Zap, ShieldAlert, Award
} from 'lucide-react';
import RiskGauge from './RiskGauge';
import { api } from '../services/api';

const PRESETS = {
  elderlyHighRisk: {
    label: "Complex Geriatric Diabetic (High Risk)",
    data: {
      gender: "Male",
      age: 78,
      race: "AfricanAmerican",
      admission_type_id: 1, // Emergency
      time_in_hospital: 9,
      num_lab_procedures: 68,
      num_procedures: 2,
      num_medications: 26,
      number_diagnoses: 9,
      diag_1_category: "Circulatory",
      number_outpatient: 1,
      number_emergency: 2,
      number_inpatient: 3,
      diabetesMed: true,
      medication_changed: true,
      n_med_changes: 2,
      insulin_changed: true,
      A1C_abnormal: true,
      max_glu_abnormal: false
    }
  },
  routineLowRisk: {
    label: "Elective Routine Care (Low Risk)",
    data: {
      gender: "Female",
      age: 42,
      race: "Caucasian",
      admission_type_id: 3, // Elective
      time_in_hospital: 2,
      num_lab_procedures: 24,
      num_procedures: 1,
      num_medications: 8,
      number_diagnoses: 4,
      diag_1_category: "Respiratory",
      number_outpatient: 0,
      number_emergency: 0,
      number_inpatient: 0,
      diabetesMed: false,
      medication_changed: false,
      n_med_changes: 0,
      insulin_changed: false,
      A1C_abnormal: false,
      max_glu_abnormal: false
    }
  },
  moderateDiabetes: {
    label: "Subacute Diabetes Titration (Moderate)",
    data: {
      gender: "Female",
      age: 64,
      race: "Caucasian",
      admission_type_id: 2, // Urgent
      time_in_hospital: 4,
      num_lab_procedures: 48,
      num_procedures: 0,
      num_medications: 18,
      number_diagnoses: 7,
      diag_1_category: "Diabetes",
      number_outpatient: 2,
      number_emergency: 0,
      number_inpatient: 1,
      diabetesMed: true,
      medication_changed: true,
      n_med_changes: 1,
      insulin_changed: true,
      A1C_abnormal: true,
      max_glu_abnormal: true
    }
  }
};

export default function PredictorView() {
  const [formData, setFormData] = useState(PRESETS.elderlyHighRisk.data);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Run initial prediction on mount
  useEffect(() => {
    handlePredict(formData);
  }, []);

  const handlePredict = async (dataToPredict = formData) => {
    setLoading(true);
    setError(null);
    try {
      const res = await api.predict(dataToPredict);
      setPrediction(res);
    } catch (err) {
      setError(err.message || "Prediction error");
    } finally {
      setLoading(false);
    }
  };

  const handleChange = (e) => {
    const { name, value, type, checked } = e.target;
    let parsedVal = value;
    if (type === 'checkbox') parsedVal = checked;
    else if (type === 'number') parsedVal = parseInt(value, 10) || 0;

    setFormData(prev => ({
      ...prev,
      [name]: parsedVal
    }));
  };

  const applyPreset = (presetKey) => {
    const preset = PRESETS[presetKey].data;
    setFormData(preset);
    handlePredict(preset);
  };

  return (
    <div style={{ maxWidth: 1400, margin: '0 auto', padding: '24px 16px' }}>
      {/* Top Banner & Quick Presets */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 16, marginBottom: 24 }}>
        <div>
          <h2 style={{ fontSize: '1.6rem', fontWeight: 800, color: '#ffffff', letterSpacing: '-0.02em' }}>
            Patient Readmission Risk Assessment
          </h2>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>
            Enter clinical parameters to stratify 30-day post-discharge readmission likelihood.
          </p>
        </div>

        {/* Presets */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
          <span style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-dim)', textTransform: 'uppercase' }}>
            Load Presets:
          </span>
          {Object.entries(PRESETS).map(([key, item]) => (
            <button
              key={key}
              onClick={() => applyPreset(key)}
              className="btn btn-secondary"
              style={{ fontSize: '0.8rem', padding: '6px 12px' }}
            >
              <Zap size={14} color="var(--cyan-primary)" />
              {item.label}
            </button>
          ))}
        </div>
      </div>

      {/* Main Grid: Form Left (7 cols), Results Right (5 cols) */}
      <div className="dashboard-grid">
        {/* Left: Input Parameters Card */}
        <div className="glass-card" style={{ gridColumn: 'span 7', padding: 24 }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', borderBottom: '1px solid var(--border-subtle)', paddingBottom: 14, marginBottom: 20 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
              <User size={20} color="var(--cyan-primary)" />
              <h3 style={{ fontSize: '1.1rem', fontWeight: 700, color: '#ffffff' }}>Clinical Features</h3>
            </div>
            <button
              onClick={() => handlePredict(formData)}
              disabled={loading}
              className="btn btn-primary"
            >
              <RefreshCw size={16} className={loading ? "animate-spin" : ""} />
              {loading ? "Calculating..." : "Compute Risk Score"}
            </button>
          </div>

          <form onSubmit={(e) => { e.preventDefault(); handlePredict(formData); }}>
            {/* Section 1: Demographics */}
            <div style={{ marginBottom: 20 }}>
              <h4 style={{ fontSize: '0.85rem', fontWeight: 700, color: 'var(--cyan-primary)', textTransform: 'uppercase', marginBottom: 12, display: 'flex', alignItems: 'center', gap: 6 }}>
                <User size={15} /> 1. Demographics & Stratification
              </h4>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 14 }}>
                <div className="form-group">
                  <label className="form-label">Patient Age (Years)</label>
                  <input
                    type="number"
                    name="age"
                    min="1"
                    max="110"
                    value={formData.age}
                    onChange={handleChange}
                    className="form-input"
                  />
                </div>
                <div className="form-group">
                  <label className="form-label">Gender</label>
                  <select name="gender" value={formData.gender} onChange={handleChange} className="form-select">
                    <option value="Female">Female</option>
                    <option value="Male">Male</option>
                  </select>
                </div>
                <div className="form-group">
                  <label className="form-label">Ethnicity / Race</label>
                  <select name="race" value={formData.race} onChange={handleChange} className="form-select">
                    <option value="Caucasian">Caucasian</option>
                    <option value="AfricanAmerican">African American</option>
                    <option value="Hispanic">Hispanic</option>
                    <option value="Asian">Asian</option>
                    <option value="Other">Other</option>
                    <option value="Unknown">Unknown</option>
                  </select>
                </div>
              </div>
            </div>

            {/* Section 2: Hospital Encounter Details */}
            <div style={{ marginBottom: 20 }}>
              <h4 style={{ fontSize: '0.85rem', fontWeight: 700, color: 'var(--cyan-primary)', textTransform: 'uppercase', marginBottom: 12, display: 'flex', alignItems: 'center', gap: 6 }}>
                <Clock size={15} /> 2. Encounter & Length of Stay
              </h4>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 14 }}>
                <div className="form-group">
                  <label className="form-label">Admission Type</label>
                  <select name="admission_type_id" value={formData.admission_type_id} onChange={handleChange} className="form-select">
                    <option value="1">1 - Emergency</option>
                    <option value="2">2 - Urgent</option>
                    <option value="3">3 - Elective</option>
                    <option value="4">4 - Newborn</option>
                    <option value="5">5 - Trauma Center</option>
                    <option value="7">7 - Other</option>
                  </select>
                </div>
                <div className="form-group">
                  <label className="form-label">Length of Stay (Days)</label>
                  <input
                    type="number"
                    name="time_in_hospital"
                    min="1"
                    max="14"
                    value={formData.time_in_hospital}
                    onChange={handleChange}
                    className="form-input"
                  />
                </div>
                <div className="form-group">
                  <label className="form-label">Primary Diagnosis</label>
                  <select name="diag_1_category" value={formData.diag_1_category} onChange={handleChange} className="form-select">
                    <option value="Circulatory">Circulatory (Heart / Vascular)</option>
                    <option value="Respiratory">Respiratory (Pulmonary)</option>
                    <option value="Digestive">Digestive (GI)</option>
                    <option value="Diabetes">Diabetes Primary</option>
                    <option value="Injury">Injury / Trauma</option>
                    <option value="Musculoskeletal">Musculoskeletal</option>
                    <option value="Genitourinary">Genitourinary (Renal)</option>
                    <option value="Neoplasms">Neoplasms (Oncology)</option>
                    <option value="Other">Other</option>
                  </select>
                </div>
              </div>
            </div>

            {/* Section 3: Procedures & Utilization */}
            <div style={{ marginBottom: 20 }}>
              <h4 style={{ fontSize: '0.85rem', fontWeight: 700, color: 'var(--cyan-primary)', textTransform: 'uppercase', marginBottom: 12, display: 'flex', alignItems: 'center', gap: 6 }}>
                <Stethoscope size={15} /> 3. Procedures & Prior Healthcare Utilization
              </h4>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 14 }}>
                <div className="form-group">
                  <label className="form-label">Lab Tests Count</label>
                  <input
                    type="number"
                    name="num_lab_procedures"
                    min="1"
                    max="132"
                    value={formData.num_lab_procedures}
                    onChange={handleChange}
                    className="form-input"
                  />
                </div>
                <div className="form-group">
                  <label className="form-label">Clinical Procedures</label>
                  <input
                    type="number"
                    name="num_procedures"
                    min="0"
                    max="6"
                    value={formData.num_procedures}
                    onChange={handleChange}
                    className="form-input"
                  />
                </div>
                <div className="form-group">
                  <label className="form-label">Number of Diagnoses</label>
                  <input
                    type="number"
                    name="number_diagnoses"
                    min="1"
                    max="16"
                    value={formData.number_diagnoses}
                    onChange={handleChange}
                    className="form-input"
                  />
                </div>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 14, marginTop: 4 }}>
                <div className="form-group">
                  <label className="form-label">Prior Inpatient Admissions</label>
                  <input
                    type="number"
                    name="number_inpatient"
                    min="0"
                    max="20"
                    value={formData.number_inpatient}
                    onChange={handleChange}
                    className="form-input"
                    style={{ borderColor: formData.number_inpatient > 0 ? '#f43f5e' : 'var(--border-subtle)' }}
                  />
                </div>
                <div className="form-group">
                  <label className="form-label">Prior Emergency Visits</label>
                  <input
                    type="number"
                    name="number_emergency"
                    min="0"
                    max="40"
                    value={formData.number_emergency}
                    onChange={handleChange}
                    className="form-input"
                  />
                </div>
                <div className="form-group">
                  <label className="form-label">Prior Outpatient Visits</label>
                  <input
                    type="number"
                    name="number_outpatient"
                    min="0"
                    max="40"
                    value={formData.number_outpatient}
                    onChange={handleChange}
                    className="form-input"
                  />
                </div>
              </div>
            </div>

            {/* Section 4: Medications & Glycemic Status */}
            <div>
              <h4 style={{ fontSize: '0.85rem', fontWeight: 700, color: 'var(--cyan-primary)', textTransform: 'uppercase', marginBottom: 12, display: 'flex', alignItems: 'center', gap: 6 }}>
                <Pill size={15} /> 4. Pharmacotherapy & Glycemic Indicators
              </h4>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 14 }}>
                <div className="form-group">
                  <label className="form-label">Total Medications Count</label>
                  <input
                    type="number"
                    name="num_medications"
                    min="1"
                    max="80"
                    value={formData.num_medications}
                    onChange={handleChange}
                    className="form-input"
                  />
                </div>
                <div className="form-group">
                  <label className="form-label">Medication Dosage Changes</label>
                  <input
                    type="number"
                    name="n_med_changes"
                    min="0"
                    max="10"
                    value={formData.n_med_changes}
                    onChange={handleChange}
                    className="form-input"
                  />
                </div>
              </div>

              {/* Checkboxes */}
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 12, marginTop: 8 }}>
                <label style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: '0.86rem', cursor: 'pointer', color: 'var(--text-muted)' }}>
                  <input
                    type="checkbox"
                    name="diabetesMed"
                    checked={formData.diabetesMed}
                    onChange={handleChange}
                    style={{ accentColor: 'var(--cyan-primary)', width: 16, height: 16 }}
                  />
                  Prescribed Diabetes Medications
                </label>

                <label style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: '0.86rem', cursor: 'pointer', color: 'var(--text-muted)' }}>
                  <input
                    type="checkbox"
                    name="insulin_changed"
                    checked={formData.insulin_changed}
                    onChange={handleChange}
                    style={{ accentColor: 'var(--cyan-primary)', width: 16, height: 16 }}
                  />
                  Insulin Regimen Adjusted
                </label>

                <label style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: '0.86rem', cursor: 'pointer', color: 'var(--text-muted)' }}>
                  <input
                    type="checkbox"
                    name="A1C_abnormal"
                    checked={formData.A1C_abnormal}
                    onChange={handleChange}
                    style={{ accentColor: 'var(--cyan-primary)', width: 16, height: 16 }}
                  />
                  HbA1c Result Abnormal (&gt;8%)
                </label>

                <label style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: '0.86rem', cursor: 'pointer', color: 'var(--text-muted)' }}>
                  <input
                    type="checkbox"
                    name="max_glu_abnormal"
                    checked={formData.max_glu_abnormal}
                    onChange={handleChange}
                    style={{ accentColor: 'var(--cyan-primary)', width: 16, height: 16 }}
                  />
                  Serum Glucose Abnormal (&gt;200)
                </label>
              </div>
            </div>
          </form>
        </div>

        {/* Right: Risk Assessment & Clinical Output Card */}
        <div style={{ gridColumn: 'span 5', display: 'flex', flexDirection: 'column', gap: 20 }}>
          {/* Main Risk Result Card */}
          <div className="glass-card" style={{ padding: 24, textAlign: 'center' }}>
            <h3 style={{ fontSize: '1rem', fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: 16 }}>
              Predicted 30-Day Readmission Risk
            </h3>

            {prediction ? (
              <div>
                <RiskGauge
                  percentage={prediction.readmission_risk_percent}
                  riskLevel={prediction.risk_level}
                />

                <div style={{ marginTop: 20, padding: '12px 16px', background: 'rgba(255, 255, 255, 0.03)', borderRadius: 12, border: '1px solid var(--border-subtle)', display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 8 }}>
                  <div>
                    <div style={{ fontSize: '0.72rem', color: 'var(--text-dim)', textTransform: 'uppercase' }}>Classification</div>
                    <div style={{ fontSize: '0.88rem', fontWeight: 700, color: prediction.will_readmit_30d ? '#f43f5e' : '#34d399' }}>
                      {prediction.will_readmit_30d ? 'Likely Readmit' : 'Standard Discharge'}
                    </div>
                  </div>
                  <div>
                    <div style={{ fontSize: '0.72rem', color: 'var(--text-dim)', textTransform: 'uppercase' }}>Threshold</div>
                    <div style={{ fontSize: '0.88rem', fontWeight: 700, color: 'var(--text-main)' }}>
                      {prediction.decision_threshold}
                    </div>
                  </div>
                  <div>
                    <div style={{ fontSize: '0.72rem', color: 'var(--text-dim)', textTransform: 'uppercase' }}>Model</div>
                    <div style={{ fontSize: '0.88rem', fontWeight: 700, color: 'var(--cyan-primary)' }}>
                      {prediction.model_name}
                    </div>
                  </div>
                </div>

                {/* Derived Features badges */}
                <div style={{ display: 'flex', justifyContent: 'center', gap: 8, marginTop: 14, flexWrap: 'wrap' }}>
                  <span className="badge" style={{ background: 'rgba(59, 130, 246, 0.1)', color: '#60a5fa', border: '1px solid rgba(59, 130, 246, 0.2)' }}>
                    Total Past Visits: {prediction.derived_features?.total_past_admissions}
                  </span>
                  <span className="badge" style={{ background: prediction.derived_features?.polypharmacy ? 'rgba(244, 63, 94, 0.1)' : 'rgba(16, 185, 129, 0.1)', color: prediction.derived_features?.polypharmacy ? '#fb7185' : '#34d399', border: '1px solid var(--border-subtle)' }}>
                    {prediction.derived_features?.polypharmacy ? 'Polypharmacy (>20)' : 'Normal Drug Count'}
                  </span>
                  <span className="badge" style={{ background: 'rgba(139, 92, 246, 0.1)', color: '#a78bfa', border: '1px solid rgba(139, 92, 246, 0.2)' }}>
                    Risk Tier: {prediction.derived_features?.admission_risk_level}
                  </span>
                </div>
              </div>
            ) : (
              <div style={{ padding: '40px 0', color: 'var(--text-muted)' }}>
                {loading ? "Analyzing patient features..." : "Enter patient data to compute risk profile."}
              </div>
            )}
          </div>

          {/* Key Contributing Factors */}
          {prediction?.key_contributing_factors?.length > 0 && (
            <div className="glass-card" style={{ padding: 20 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 14 }}>
                <ShieldAlert size={18} color="var(--amber-primary)" />
                <h4 style={{ fontSize: '0.95rem', fontWeight: 700, color: '#ffffff' }}>Key Contributing Risk Drivers</h4>
              </div>

              <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                {prediction.key_contributing_factors.map((factor, idx) => {
                  const isRisk = factor.effect === "Increases Risk";
                  return (
                    <div
                      key={idx}
                      style={{
                        padding: '10px 14px',
                        borderRadius: 10,
                        background: isRisk ? 'rgba(244, 63, 94, 0.06)' : 'rgba(16, 185, 129, 0.06)',
                        borderLeft: `3px solid ${isRisk ? '#f43f5e' : '#10b981'}`
                      }}
                    >
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 2 }}>
                        <span style={{ fontSize: '0.86rem', fontWeight: 600, color: '#ffffff' }}>
                          {factor.feature}
                        </span>
                        <span style={{ fontSize: '0.74rem', fontWeight: 700, color: isRisk ? '#fb7185' : '#34d399' }}>
                          {factor.value} ({isRisk ? '+ Risk' : '- Risk'})
                        </span>
                      </div>
                      <p style={{ fontSize: '0.76rem', color: 'var(--text-muted)', lineHeight: 1.4 }}>
                        {factor.description}
                      </p>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* Clinical Recommendations Checklist */}
          {prediction?.clinical_recommendations?.length > 0 && (
            <div className="glass-card" style={{ padding: 20 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 14 }}>
                <Award size={18} color="var(--cyan-primary)" />
                <h4 style={{ fontSize: '0.95rem', fontWeight: 700, color: '#ffffff' }}>Targeted Clinical Interventions</h4>
              </div>

              <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                {prediction.clinical_recommendations.map((rec, idx) => (
                  <div
                    key={idx}
                    style={{
                      padding: '10px 14px',
                      borderRadius: 10,
                      background: 'rgba(255, 255, 255, 0.03)',
                      border: '1px solid var(--border-subtle)'
                    }}
                  >
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 4 }}>
                      <span className="badge" style={{
                        background: rec.priority === 'Critical' ? 'rgba(244, 63, 94, 0.2)' : 'rgba(59, 130, 246, 0.2)',
                        color: rec.priority === 'Critical' ? '#fb7185' : '#60a5fa',
                        fontSize: '0.68rem',
                        padding: '2px 8px'
                      }}>
                        {rec.priority} Priority • {rec.category}
                      </span>
                    </div>
                    <div style={{ fontSize: '0.85rem', fontWeight: 600, color: '#ffffff', marginBottom: 2 }}>
                      {rec.action}
                    </div>
                    <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                      {rec.rationale}
                    </p>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
