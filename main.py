"""
MedPulseAI - Hospital Readmission Risk Prediction System
Root entry point forwarding to the FastAPI application.
Run via:
    uvicorn main:app --reload
or:
    python main.py
"""

from backend.app.main import app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=True)
