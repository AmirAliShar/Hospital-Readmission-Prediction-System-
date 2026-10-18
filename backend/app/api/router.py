from fastapi import APIRouter
from backend.app.api.endpoints import health, prediction, eda, model

api_router = APIRouter()

api_router.include_router(health.router)
api_router.include_router(prediction.router)
api_router.include_router(eda.router)
api_router.include_router(model.router)
