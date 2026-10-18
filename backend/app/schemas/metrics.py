from typing import Dict, Any, List, Optional
from pydantic import BaseModel

class ModelInfo(BaseModel):
    model_name: str
    model_version: str
    dataset: str
    dataset_rows: int
    dataset_columns: int
    target: str
    positive_rate_percent: float
    feature_count: int
    feature_names_numeric: List[str]
    feature_names_categorical: List[str]
    train_size: int
    test_size: int
    selection_metric: str
    selection_rationale: str
    decision_threshold: float
    training_date: str
    packages: Dict[str, str]

class MetricSet(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    pr_auc: float
    log_loss: float
    brier_score: float
    specificity: float
    confusion_matrix: List[List[int]]

class ModelMetricsResponse(BaseModel):
    best_model: str
    test_metrics: Dict[str, Any]
    validation_metrics: Dict[str, Any]
    feature_importance: Dict[str, float]

class EDASummaryResponse(BaseModel):
    data_quality: Dict[str, Any]
    target_info: Dict[str, Any]
    age_distribution: Dict[str, int]
    age_readmission_rate: Dict[str, float]
    gender_distribution: Dict[str, int]
    gender_readmission_rate: Dict[str, float]
    race_distribution: Dict[str, int]
    race_readmission_rate: Dict[str, float]
    diagnosis_distribution: Dict[str, int]
    diagnosis_readmission_rate: Dict[str, float]
    time_in_hospital_stats: Dict[str, Any]
    num_medications_stats: Dict[str, Any]
    num_lab_procedures_stats: Dict[str, Any]
    emergency_visits_stats: Dict[str, Any]
    outpatient_visits_stats: Dict[str, Any]
    feature_correlations: Dict[str, float]
    charts: List[str]
    evaluation_charts: List[str]
