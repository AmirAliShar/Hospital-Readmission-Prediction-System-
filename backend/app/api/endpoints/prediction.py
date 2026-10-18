from fastapi import APIRouter, HTTPException
from backend.app.schemas.patient import PatientInput
from backend.app.schemas.prediction import (
    PredictionResponse, BatchPredictionRequest, BatchPredictionResponse
)
from backend.app.services.prediction_service import prediction_service

router = APIRouter()

@router.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_readmission(patient: PatientInput):
    """
    Predict 30-day hospital readmission risk for an individual patient.
    Returns calculated probability, risk tier, key contributing factors, and clinical recommendations.
    """
    try:
        response = prediction_service.predict(patient)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@router.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def batch_predict(batch: BatchPredictionRequest):
    """
    Run predictions on a batch of patients for clinical cohort stratification.
    """
    try:
        predictions = []
        high_cnt = 0
        mod_cnt = 0
        low_cnt = 0
        
        for item in batch.patients:
            patient = PatientInput(**item)
            pred = prediction_service.predict(patient)
            predictions.append(pred)
            if pred.risk_level in ["High", "Critical"]:
                high_cnt += 1
            elif pred.risk_level == "Moderate":
                mod_cnt += 1
            else:
                low_cnt += 1
                
        return BatchPredictionResponse(
            total_patients=len(predictions),
            high_risk_count=high_cnt,
            moderate_risk_count=mod_cnt,
            low_risk_count=low_cnt,
            predictions=predictions
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Batch processing error: {str(e)}")

@router.get("/predict/schema", tags=["Prediction"])
async def get_patient_schema():
    """
    Retrieve input validation ranges, enum values, and defaults to populate client forms.
    """
    return {
        "allowed_values": prediction_service.allowed_values,
        "default_sample": PatientInput().model_dump()
    }
