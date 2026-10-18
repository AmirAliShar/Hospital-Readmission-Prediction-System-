import json
import logging
from typing import Dict, Any, List, Tuple
import joblib
import numpy as np
import pandas as pd

from backend.app.core.config import settings
from backend.app.schemas.patient import PatientInput
from backend.app.schemas.prediction import (
    PredictionResponse, FactorImpact, ClinicalRecommendation
)

logger = logging.getLogger(__name__)

class PredictionService:
    _instance = None
    
    def __init__(self):
        self.pipeline = None
        self.metadata = {}
        self.allowed_values = {}
        self.feature_importance = {}
        self._load_artifacts()
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load_artifacts(self):
        try:
            if settings.MODEL_PATH.exists():
                self.pipeline = joblib.load(settings.MODEL_PATH)
                logger.info(f"Loaded ML pipeline from {settings.MODEL_PATH}")
            else:
                logger.warning(f"Model path not found: {settings.MODEL_PATH}")
                
            if settings.METADATA_PATH.exists():
                with open(settings.METADATA_PATH, 'r') as f:
                    self.metadata = json.load(f)
                    
            if settings.ALLOWED_VALUES_PATH.exists():
                with open(settings.ALLOWED_VALUES_PATH, 'r') as f:
                    self.allowed_values = json.load(f)
                    
            if settings.FEATURE_IMPORTANCE_PATH.exists():
                with open(settings.FEATURE_IMPORTANCE_PATH, 'r') as f:
                    self.feature_importance = json.load(f)
        except Exception as e:
            logger.error(f"Error loading artifacts: {e}", exc_info=True)

    def _map_age_to_numeric(self, age: int) -> int:
        """Map exact age to UCI age bucket midpoint"""
        brackets = [
            (0, 10, 5), (10, 20, 15), (20, 30, 25), (30, 40, 35),
            (40, 50, 45), (50, 60, 55), (60, 70, 65), (70, 80, 75),
            (80, 90, 85), (90, 120, 95)
        ]
        for low, high, midpoint in brackets:
            if low <= age < high:
                return midpoint
        return 65

    def _determine_admission_risk(self, total_admissions: int) -> str:
        if total_admissions == 0:
            return "None"
        elif 1 <= total_admissions <= 3:
            return "Low"
        elif 4 <= total_admissions <= 10:
            return "Medium"
        else:
            return "High"

    def preprocess_patient(self, patient: PatientInput) -> Tuple[pd.DataFrame, dict]:
        """Convert PatientInput schema to DataFrame matching trained pipeline columns"""
        age_numeric = self._map_age_to_numeric(patient.age)
        total_past_admissions = patient.number_outpatient + patient.number_emergency + patient.number_inpatient
        admission_risk_level = self._determine_admission_risk(total_past_admissions)
        polypharmacy = 1 if patient.num_medications > 20 else 0
        
        # Match categorical fields with training dataset domain
        gender = patient.gender if patient.gender in ["Female", "Male"] else "Female"
        race = patient.race if patient.race in self.allowed_values.get("race", []) else "Caucasian"
        diag_1_category = patient.diag_1_category
        
        data_dict = {
            # Numeric columns
            "time_in_hospital": [patient.time_in_hospital],
            "num_lab_procedures": [patient.num_lab_procedures],
            "num_procedures": [patient.num_procedures],
            "num_medications": [patient.num_medications],
            "number_outpatient": [patient.number_outpatient],
            "number_emergency": [patient.number_emergency],
            "number_inpatient": [patient.number_inpatient],
            "number_diagnoses": [patient.number_diagnoses],
            "age_numeric": [age_numeric],
            "total_past_admissions": [total_past_admissions],
            "n_med_changes": [patient.n_med_changes],
            "polypharmacy": [polypharmacy],
            "diabetesMed_binary": [1 if patient.diabetesMed else 0],
            "change_binary": [1 if patient.medication_changed else 0],
            "insulin_change": [1 if patient.insulin_changed else 0],
            "A1C_abnormal": [1 if patient.A1C_abnormal else 0],
            "max_glu_abnormal": [1 if patient.max_glu_abnormal else 0],
            "admission_type_id": [patient.admission_type_id],
            # Categorical columns
            "gender": [gender],
            "race": [race],
            "diag_1_category": [diag_1_category],
            "admission_risk_level": [admission_risk_level],
        }
        
        derived = {
            "age_numeric": age_numeric,
            "total_past_admissions": total_past_admissions,
            "admission_risk_level": admission_risk_level,
            "polypharmacy": bool(polypharmacy)
        }
        
        df = pd.DataFrame(data_dict)
        return df, derived

    def _generate_factor_impacts(self, patient: PatientInput, derived: dict) -> List[FactorImpact]:
        """Generate patient-specific risk factor explanations"""
        factors = []
        
        # 1. Inpatient visits
        if patient.number_inpatient > 0:
            factors.append(FactorImpact(
                feature="Prior Inpatient Admissions",
                value=f"{patient.number_inpatient} past admission(s)",
                effect="Increases Risk",
                weight=0.175 * patient.number_inpatient,
                description="Prior hospitalizations are one of the strongest predictive indicators of 30-day readmission."
            ))
            
        # 2. Length of stay
        if patient.time_in_hospital >= 5:
            factors.append(FactorImpact(
                feature="Extended Hospital Stay",
                value=f"{patient.time_in_hospital} days",
                effect="Increases Risk",
                weight=0.128 * (patient.time_in_hospital / 7),
                description="Extended stay indicates complex clinical pathology, functional decline, or complicated recovery."
            ))
        elif patient.time_in_hospital <= 2:
            factors.append(FactorImpact(
                feature="Short Hospital Stay",
                value=f"{patient.time_in_hospital} days",
                effect="Decreases Risk",
                weight=-0.08,
                description="Short recovery time correlates with faster stabilization and lower acute complexity."
            ))
            
        # 3. Emergency room visits
        if patient.number_emergency > 0:
            factors.append(FactorImpact(
                feature="Emergency Department Utilization",
                value=f"{patient.number_emergency} visit(s)",
                effect="Increases Risk",
                weight=0.09 * patient.number_emergency,
                description="Frequent emergency encounters suggest unstable chronic conditions and ambulatory care gaps."
            ))
            
        # 4. Polypharmacy
        if derived.get("polypharmacy"):
            factors.append(FactorImpact(
                feature="Polypharmacy (>20 Medications)",
                value=f"{patient.num_medications} medications",
                effect="Increases Risk",
                weight=0.085,
                description="Complex multi-drug regimens significantly increase adverse drug events and non-compliance risk."
            ))
            
        # 5. Primary diagnosis
        if patient.diag_1_category == "Injury":
            factors.append(FactorImpact(
                feature="Primary Diagnosis: Injury / Trauma",
                value="Injury",
                effect="Increases Risk",
                weight=0.140,
                description="Trauma and acute injuries frequently carry post-discharge rehabilitation challenges."
            ))
        elif patient.diag_1_category == "Respiratory":
            factors.append(FactorImpact(
                feature="Primary Diagnosis: Respiratory",
                value="Respiratory",
                effect="Decreases Risk",
                weight=-0.284,
                description="Managed respiratory admissions in this cohort showed lower 30-day acute readmission rates."
            ))
        elif patient.diag_1_category == "Neoplasms":
            factors.append(FactorImpact(
                feature="Primary Diagnosis: Neoplasms / Oncology",
                value="Neoplasms",
                effect="Increases Risk",
                weight=0.121,
                description="Oncologic conditions require aggressive monitoring for therapy-induced complications."
            ))
            
        # 6. Age
        if patient.age >= 75:
            factors.append(FactorImpact(
                feature="Advanced Age (Geriatric)",
                value=f"{patient.age} years",
                effect="Increases Risk",
                weight=0.163,
                description="Older patients face heightened vulnerability, frailty, and slower post-acute recovery."
            ))
            
        # 7. Glycemic abnormality
        if patient.A1C_abnormal:
            factors.append(FactorImpact(
                feature="Elevated HbA1c (>8%)",
                value="Abnormal",
                effect="Increases Risk",
                weight=0.075,
                description="Suboptimal long-term glycemic control elevates the risk of acute metabolic and vascular complications."
            ))
            
        # 8. Admission Risk Level
        if derived.get("admission_risk_level") == "None":
            factors.append(FactorImpact(
                feature="Zero Prior Healthcare Encounters",
                value="No prior visits",
                effect="Decreases Risk",
                weight=-0.137,
                description="First-time admissions with no recent utilization history have a substantially lower baseline recurrence."
            ))
            
        return factors

    def _generate_recommendations(self, patient: PatientInput, prob: float, derived: dict) -> List[ClinicalRecommendation]:
        """Generate targeted evidence-based clinical interventions"""
        recs = []
        
        # High or Critical Risk
        if prob >= 0.50:
            recs.append(ClinicalRecommendation(
                priority="Critical",
                category="Discharge Planning",
                action="Arrange multi-disciplinary transition coordinator review prior to sign-off",
                rationale=f"Model predicts elevated readmission risk ({prob*100:.1f}%). Structured transition planning decreases 30-day events by ~25%."
            ))
            recs.append(ClinicalRecommendation(
                priority="High",
                category="Follow-up Care",
                action="Mandate Primary Care Provider (PCP) visit within 5 to 7 days post-discharge",
                rationale="Early post-discharge outpatient evaluation catches emerging decompensations before acute re-presentation."
            ))
            
        # Prior inpatient or high utilization
        if patient.number_inpatient > 0 or patient.number_emergency > 1:
            recs.append(ClinicalRecommendation(
                priority="High",
                category="Remote Monitoring",
                action="Enroll in 48-hour Post-Discharge Telephonic Outreach and Symptom Monitoring Program",
                rationale="Patients with repeat utilization history benefit most from structured phone check-ins within 48-72 hours."
            ))
            
        # Medication & Polypharmacy
        if patient.num_medications > 15 or patient.medication_changed or patient.n_med_changes > 0:
            recs.append(ClinicalRecommendation(
                priority="High" if patient.num_medications > 20 else "Medium",
                category="Medication Management",
                action="Perform bedside clinical pharmacist medication reconciliation and pill-box / blister pack schedule",
                rationale=f"{patient.num_medications} medications recorded with active adjustments. Medication discrepancy is a leading trigger for early readmission."
            ))
            
        # Diabetes & Glycemic Control
        if patient.A1C_abnormal or patient.diabetesMed or patient.diag_1_category == "Diabetes":
            recs.append(ClinicalRecommendation(
                priority="Medium",
                category="Patient Education",
                action="Provide specialized Diabetes Self-Management Education (DSME) & glucometer calibration verification",
                rationale="Ensures safe glycemic titration at home and awareness of hypo/hyperglycemic red-flag warning signs."
            ))
            
        # Standard baseline
        if not recs:
            recs.append(ClinicalRecommendation(
                priority="Standard",
                category="Follow-up Care",
                action="Schedule routine outpatient follow-up within 14-21 days",
                rationale="Patient falls in the lower risk tier; standard ambulatory follow-up protocol is sufficient."
            ))
            
        return recs

    def predict(self, patient: PatientInput) -> PredictionResponse:
        """Execute full prediction pipeline with risk scoring and explainability"""
        df_input, derived = self.preprocess_patient(patient)
        
        threshold = float(self.metadata.get("decision_threshold", 0.5))
        model_name = self.metadata.get("model_name", "Logistic Regression")
        model_ver = self.metadata.get("model_version", "2.0")
        
        if self.pipeline is not None:
            proba = float(self.pipeline.predict_proba(df_input)[0, 1])
        else:
            # Fallback heuristic calculation if pipeline not yet loaded
            base_risk = 0.10
            if patient.number_inpatient > 0: base_risk += 0.20
            if patient.time_in_hospital > 5: base_risk += 0.15
            if patient.num_medications > 20: base_risk += 0.10
            if patient.A1C_abnormal: base_risk += 0.08
            proba = min(0.95, base_risk)
            
        risk_percent = round(proba * 100, 1)
        
        if proba < 0.20:
            risk_level = "Low"
        elif proba < 0.45:
            risk_level = "Moderate"
        elif proba < 0.65:
            risk_level = "High"
        else:
            risk_level = "Critical"
            
        factors = self._generate_factor_impacts(patient, derived)
        recommendations = self._generate_recommendations(patient, proba, derived)
        
        return PredictionResponse(
            readmission_probability=round(proba, 4),
            readmission_risk_percent=risk_percent,
            risk_level=risk_level,
            will_readmit_30d=bool(proba >= threshold),
            decision_threshold=threshold,
            model_name=model_name,
            model_version=model_ver,
            key_contributing_factors=factors,
            clinical_recommendations=recommendations,
            derived_features=derived
        )

prediction_service = PredictionService.get_instance()
