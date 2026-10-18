from typing import List, Optional, Literal
from pydantic import BaseModel, Field

class FactorImpact(BaseModel):
    feature: str = Field(..., description="Feature name or human-readable description")
    value: str = Field(..., description="Observed patient value")
    effect: Literal["Increases Risk", "Decreases Risk", "Neutral"] = Field(..., description="Impact direction on readmission risk")
    weight: float = Field(..., description="Model coefficient or feature contribution weight")
    description: str = Field(..., description="Clinical context explaining why this impacts readmission risk")

class ClinicalRecommendation(BaseModel):
    priority: Literal["Critical", "High", "Medium", "Standard"] = Field(..., description="Urgency level")
    category: Literal["Discharge Planning", "Medication Management", "Follow-up Care", "Patient Education", "Remote Monitoring"] = Field(...)
    action: str = Field(..., description="Recommended clinical intervention")
    rationale: str = Field(..., description="Clinical reasoning behind this action")

class PredictionResponse(BaseModel):
    readmission_probability: float = Field(..., ge=0.0, le=1.0, description="Predicted probability of 30-day readmission")
    readmission_risk_percent: float = Field(..., ge=0.0, le=100.0, description="Risk score percentage (0-100%)")
    risk_level: Literal["Low", "Moderate", "High", "Critical"] = Field(..., description="Stratified clinical risk category")
    will_readmit_30d: bool = Field(..., description="Binary classification at decision threshold")
    decision_threshold: float = Field(default=0.5, description="Decision threshold applied")
    model_name: str
    model_version: str
    key_contributing_factors: List[FactorImpact] = Field(default_factory=list, description="Top positive and negative driving factors")
    clinical_recommendations: List[ClinicalRecommendation] = Field(default_factory=list, description="Evidence-based care suggestions")
    derived_features: dict = Field(default_factory=dict, description="Engineered features calculated for this patient")

class BatchPredictionRequest(BaseModel):
    patients: List[dict] = Field(..., min_length=1, max_length=100)

class BatchPredictionResponse(BaseModel):
    total_patients: int
    high_risk_count: int
    moderate_risk_count: int
    low_risk_count: int
    predictions: List[PredictionResponse]
