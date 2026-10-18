from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from backend.app.core.config import settings
from backend.app.api.router import api_router

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description=settings.DESCRIPTION,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount generated charts static directory
if settings.CHARTS_DIR_PATH.exists():
    app.mount("/charts", StaticFiles(directory=str(settings.CHARTS_DIR_PATH)), name="charts")

# Include API routes
app.include_router(api_router, prefix=settings.API_PREFIX)

@app.get("/", tags=["Root"])
async def root():
    return {
        "title": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "status": "online",
        "documentation": "/docs",
        "endpoints": {
            "health": f"{settings.API_PREFIX}/health",
            "predict": f"{settings.API_PREFIX}/predict",
            "predict_schema": f"{settings.API_PREFIX}/predict/schema",
            "eda_summary": f"{settings.API_PREFIX}/eda/summary",
            "eda_charts": f"{settings.API_PREFIX}/eda/charts",
            "model_info": f"{settings.API_PREFIX}/model/info",
            "model_metrics": f"{settings.API_PREFIX}/model/metrics",
            "model_comparison": f"{settings.API_PREFIX}/model/comparison",
            "model_features": f"{settings.API_PREFIX}/model/features"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=True)
