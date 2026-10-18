import pytest
from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "endpoints" in data

def test_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["healthy", "degraded"]
    assert "version" in data

def test_predict_schema():
    response = client.get("/api/predict/schema")
    assert response.status_code == 200
    data = response.json()
    assert "allowed_values" in data
    assert "default_sample" in data

def test_prediction_standard():
    payload = {
        "gender": "Female",
        "age": 65,
        "race": "Caucasian",
        "admission_type_id": 1,
        "time_in_hospital": 4,
        "num_lab_procedures": 43,
        "num_procedures": 1,
        "num_medications": 16,
        "number_diagnoses": 7,
        "diag_1_category": "Circulatory",
        "number_outpatient": 0,
        "number_emergency": 0,
        "number_inpatient": 0,
        "diabetesMed": True,
        "medication_changed": False,
        "n_med_changes": 0,
        "insulin_changed": False,
        "A1C_abnormal": False,
        "max_glu_abnormal": False
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "readmission_probability" in data
    assert 0.0 <= data["readmission_probability"] <= 1.0
    assert "risk_level" in data
    assert data["risk_level"] in ["Low", "Moderate", "High", "Critical"]
    assert len(data["key_contributing_factors"]) >= 0
    assert len(data["clinical_recommendations"]) >= 1

def test_prediction_high_risk():
    high_risk_payload = {
        "gender": "Male",
        "age": 82,
        "race": "AfricanAmerican",
        "admission_type_id": 1,
        "time_in_hospital": 10,
        "num_lab_procedures": 78,
        "num_procedures": 3,
        "num_medications": 28,
        "number_diagnoses": 9,
        "diag_1_category": "Neoplasms",
        "number_outpatient": 2,
        "number_emergency": 3,
        "number_inpatient": 4,
        "diabetesMed": True,
        "medication_changed": True,
        "n_med_changes": 3,
        "insulin_changed": True,
        "A1C_abnormal": True,
        "max_glu_abnormal": True
    }
    response = client.post("/api/predict", json=high_risk_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["readmission_probability"] > 0.35
    assert data["risk_level"] in ["High", "Critical"]

def test_batch_prediction():
    batch_payload = {
        "patients": [
            {
                "gender": "Female", "age": 45, "race": "Caucasian",
                "admission_type_id": 3, "time_in_hospital": 2,
                "num_lab_procedures": 20, "num_procedures": 1, "num_medications": 8,
                "number_diagnoses": 3, "diag_1_category": "Circulatory",
                "number_outpatient": 0, "number_emergency": 0, "number_inpatient": 0,
                "diabetesMed": False, "medication_changed": False, "n_med_changes": 0,
                "insulin_changed": False, "A1C_abnormal": False, "max_glu_abnormal": False
            },
            {
                "gender": "Male", "age": 78, "race": "Caucasian",
                "admission_type_id": 1, "time_in_hospital": 8,
                "num_lab_procedures": 65, "num_procedures": 2, "num_medications": 22,
                "number_diagnoses": 8, "diag_1_category": "Diabetes",
                "number_outpatient": 1, "number_emergency": 2, "number_inpatient": 2,
                "diabetesMed": True, "medication_changed": True, "n_med_changes": 1,
                "insulin_changed": True, "A1C_abnormal": True, "max_glu_abnormal": False
            }
        ]
    }
    response = client.post("/api/predict/batch", json=batch_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_patients"] == 2
    assert len(data["predictions"]) == 2

def test_eda_endpoints():
    res = client.get("/api/eda/summary")
    assert res.status_code == 200
    data = res.json()
    assert "data_quality" in data
    assert "target_info" in data
    assert "age_distribution" in data

    res_charts = client.get("/api/eda/charts")
    assert res_charts.status_code == 200
    assert "all_charts" in res_charts.json()

def test_model_endpoints():
    res_info = client.get("/api/model/info")
    assert res_info.status_code == 200
    assert "model_name" in res_info.json()

    res_metrics = client.get("/api/model/metrics")
    assert res_metrics.status_code == 200
    assert "test_metrics" in res_metrics.json()

    res_comp = client.get("/api/model/comparison")
    assert res_comp.status_code == 200

    res_feat = client.get("/api/model/features")
    assert res_feat.status_code == 200
    assert "features" in res_feat.json()
