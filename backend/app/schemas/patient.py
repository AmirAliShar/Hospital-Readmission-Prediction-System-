from typing import Optional, Literal
from pydantic import BaseModel, Field, field_validator

class PatientInput(BaseModel):
    # Demographics
    gender: Literal["Female", "Male", "Other"] = Field(
        default="Female",
        description="Patient's legal sex/gender"
    )
    age: int = Field(
        default=65,
        ge=0,
        le=110,
        description="Patient's age in years (used to derive age bracket/numeric age)"
    )
    race: Literal["AfricanAmerican", "Asian", "Caucasian", "Hispanic", "Other", "Unknown"] = Field(
        default="Caucasian",
        description="Reported race or ethnicity"
    )
    
    # Encounter / Admission details
    admission_type_id: int = Field(
        default=1,
        ge=1,
        le=8,
        description="Admission type: 1=Emergency, 2=Urgent, 3=Elective, 4=Newborn, 5=Trauma, 6-8=Other"
    )
    time_in_hospital: int = Field(
        default=4,
        ge=1,
        le=14,
        description="Length of hospital stay in days (1-14)"
    )
    
    # Clinical procedures & diagnostics
    num_lab_procedures: int = Field(
        default=43,
        ge=1,
        le=132,
        description="Total number of lab tests performed during the encounter"
    )
    num_procedures: int = Field(
        default=1,
        ge=0,
        le=6,
        description="Number of non-lab clinical/surgical procedures performed"
    )
    num_medications: int = Field(
        default=16,
        ge=1,
        le=81,
        description="Total distinct medications administered"
    )
    number_diagnoses: int = Field(
        default=7,
        ge=1,
        le=16,
        description="Number of diagnostic codes recorded on the chart"
    )
    diag_1_category: Literal[
        "Circulatory", "Diabetes", "Digestive", "Genitourinary", 
        "Injury", "Musculoskeletal", "Neoplasms", "Other", "Respiratory"
    ] = Field(
        default="Circulatory",
        description="Primary diagnostic grouping (e.g. Circulatory, Diabetes, Respiratory)"
    )
    
    # Prior utilization / History
    number_outpatient: int = Field(
        default=0,
        ge=0,
        le=50,
        description="Number of outpatient visits in the preceding 12 months"
    )
    number_emergency: int = Field(
        default=0,
        ge=0,
        le=50,
        description="Number of emergency room visits in the preceding 12 months"
    )
    number_inpatient: int = Field(
        default=0,
        ge=0,
        le=30,
        description="Number of inpatient admissions in the preceding 12 months"
    )
    
    # Glycemic & Medication Indicators
    diabetesMed: bool = Field(
        default=True,
        description="Prescribed any anti-diabetic medication during visit"
    )
    medication_changed: bool = Field(
        default=False,
        description="Was any diabetic medication dosage or regimen changed"
    )
    n_med_changes: int = Field(
        default=0,
        ge=0,
        le=10,
        description="Number of specific medication changes made during encounter"
    )
    insulin_changed: bool = Field(
        default=False,
        description="Was insulin dosage adjusted (up, down, or newly started)"
    )
    A1C_abnormal: bool = Field(
        default=False,
        description="Was HbA1c test result > 8% (or > 7% abnormal)"
    )
    max_glu_abnormal: bool = Field(
        default=False,
        description="Was maximum serum glucose abnormal (> 200 or > 300 mg/dL)"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
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
                "number_emergency": 1,
                "number_inpatient": 1,
                "diabetesMed": True,
                "medication_changed": True,
                "n_med_changes": 1,
                "insulin_changed": True,
                "A1C_abnormal": True,
                "max_glu_abnormal": False
            }
        }
    }
