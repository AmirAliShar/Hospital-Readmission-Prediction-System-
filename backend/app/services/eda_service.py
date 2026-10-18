import json
import logging
from typing import Dict, Any, Optional
from pathlib import Path

from backend.app.core.config import settings

logger = logging.getLogger(__name__)

class EDAService:
    _instance = None
    
    def __init__(self):
        self.eda_summary = {}
        self.test_metrics = {}
        self.model_comparison = {}
        self.feature_importance = {}
        self.model_metadata = {}
        self._load_reports()
        
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load_reports(self):
        try:
            if settings.EDA_SUMMARY_PATH.exists():
                with open(settings.EDA_SUMMARY_PATH, 'r') as f:
                    self.eda_summary = json.load(f)
            if settings.TEST_METRICS_PATH.exists():
                with open(settings.TEST_METRICS_PATH, 'r') as f:
                    self.test_metrics = json.load(f)
            if settings.MODEL_COMPARISON_PATH.exists():
                with open(settings.MODEL_COMPARISON_PATH, 'r') as f:
                    self.model_comparison = json.load(f)
            if settings.FEATURE_IMPORTANCE_PATH.exists():
                with open(settings.FEATURE_IMPORTANCE_PATH, 'r') as f:
                    self.feature_importance = json.load(f)
            if settings.METADATA_PATH.exists():
                with open(settings.METADATA_PATH, 'r') as f:
                    self.model_metadata = json.load(f)
        except Exception as e:
            logger.error(f"Error loading EDA reports: {e}", exc_info=True)

    def get_eda_summary(self) -> Dict[str, Any]:
        return self.eda_summary

    def get_metrics_summary(self) -> Dict[str, Any]:
        return {
            "best_model": self.model_metadata.get("model_name", "Logistic Regression"),
            "model_version": self.model_metadata.get("model_version", "2.0"),
            "dataset_rows": self.model_metadata.get("dataset_rows", 69987),
            "test_metrics": self.test_metrics,
            "feature_importance": self.feature_importance,
            "model_metadata": self.model_metadata
        }

    def get_model_comparison(self) -> Dict[str, Any]:
        return self.model_comparison

    def get_available_charts(self) -> Dict[str, list]:
        charts = []
        if settings.CHARTS_DIR_PATH.exists():
            charts = [p.name for p in settings.CHARTS_DIR_PATH.glob("*.png")]
        return {
            "eda_charts": self.eda_summary.get("charts", []),
            "evaluation_charts": self.eda_summary.get("evaluation_charts", []),
            "all_charts": sorted(charts)
        }

eda_service = EDAService.get_instance()
