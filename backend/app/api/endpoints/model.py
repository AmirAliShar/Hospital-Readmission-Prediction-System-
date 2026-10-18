from fastapi import APIRouter, HTTPException
from backend.app.services.eda_service import eda_service
from backend.app.services.prediction_service import prediction_service

router = APIRouter()

@router.get("/model/info", tags=["Model"])
async def get_model_info():
    """
    Get current production model metadata, features, and training configuration.
    """
    return prediction_service.metadata

@router.get("/model/metrics", tags=["Model"])
async def get_model_metrics():
    """
    Get evaluation metrics on the held-out patient test set.
    """
    return eda_service.get_metrics_summary()

@router.get("/model/comparison", tags=["Model"])
async def get_model_comparison():
    """
    Compare performance metrics across all evaluated architectures (Logistic Regression, Random Forest, XGBoost, LightGBM).
    """
    return eda_service.get_model_comparison()

@router.get("/model/features", tags=["Model"])
async def get_feature_importance():
    """
    Retrieve ranked feature coefficients and importance scores.
    """
    return {
        "model_name": prediction_service.metadata.get("model_name", "Logistic Regression"),
        "features": eda_service.feature_importance
    }
