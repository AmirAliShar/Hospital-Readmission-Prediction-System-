from fastapi import APIRouter
from backend.app.core.config import settings
from backend.app.services.prediction_service import prediction_service

router = APIRouter()

@router.get("/health", tags=["Health"])
async def health_check():
    model_loaded = prediction_service.pipeline is not None
    return {
        "status": "healthy" if model_loaded else "degraded",
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "model_loaded": model_loaded,
        "model_name": prediction_service.metadata.get("model_name", "Unknown"),
        "model_version": prediction_service.metadata.get("model_version", "Unknown")
    }
