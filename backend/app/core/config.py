import os
from pathlib import Path
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent.parent.parent
ROOT_DIR = BASE_DIR.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
REPORTS_DIR = BASE_DIR / "reports"
CHARTS_DIR = REPORTS_DIR / "charts"

class Settings(BaseModel):
    PROJECT_NAME: str = "Hospital Readmission Prediction AI"
    VERSION: str = "2.0.0"
    API_PREFIX: str = "/api"
    DESCRIPTION: str = "Advanced Clinical Machine Learning System for 30-Day Hospital Readmission Risk Prediction"
    
    MODEL_PATH: Path = ARTIFACTS_DIR / "model_pipeline.joblib"
    METADATA_PATH: Path = ARTIFACTS_DIR / "model_metadata.json"
    ALLOWED_VALUES_PATH: Path = ARTIFACTS_DIR / "allowed_values.json"
    EDA_SUMMARY_PATH: Path = REPORTS_DIR / "eda_summary.json"
    TEST_METRICS_PATH: Path = REPORTS_DIR / "test_metrics.json"
    FEATURE_IMPORTANCE_PATH: Path = REPORTS_DIR / "feature_importance.json"
    MODEL_COMPARISON_PATH: Path = REPORTS_DIR / "model_comparison.json"
    CHARTS_DIR_PATH: Path = CHARTS_DIR
    
    # Risk thresholds
    LOW_RISK_THRESHOLD: float = 0.30
    HIGH_RISK_THRESHOLD: float = 0.60
    
    CORS_ORIGINS: list[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "*"
    ]

settings = Settings()
