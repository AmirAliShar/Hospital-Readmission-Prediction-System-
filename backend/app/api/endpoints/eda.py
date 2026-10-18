from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from backend.app.core.config import settings
from backend.app.services.eda_service import eda_service

router = APIRouter()

@router.get("/eda/summary", tags=["EDA"])
async def get_eda_summary():
    """
    Get comprehensive exploratory data analysis statistics, distributions, and demographic rates.
    """
    summary = eda_service.get_eda_summary()
    if not summary:
        raise HTTPException(status_code=404, detail="EDA summary not found")
    return summary

@router.get("/eda/charts", tags=["EDA"])
async def list_charts():
    """
    List all generated EDA and evaluation chart filenames.
    """
    return eda_service.get_available_charts()

@router.get("/eda/charts/{chart_name}", tags=["EDA"])
async def get_chart_image(chart_name: str):
    """
    Serve a generated analytical visualization chart by filename.
    """
    chart_path = settings.CHARTS_DIR_PATH / chart_name
    if not chart_path.exists():
        raise HTTPException(status_code=404, detail=f"Chart '{chart_name}' not found")
    return FileResponse(chart_path, media_type="image/png")
