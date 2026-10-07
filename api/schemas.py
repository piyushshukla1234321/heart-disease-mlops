"""Request / response contracts. Ranges come from cardiorisk.config so the API
accepts exactly what the model was trained on and rejects the rest with HTTP 422."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from cardiorisk.config import VALID_RANGES as R

EXAMPLE_PATIENT = {
    "age": 61,
    "sex": 0,
    "cp": 2,
    "trestbps": 130,
    "chol": 240,
    "fbs": 0,
    "restecg": 0,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 1.0,
    "slope": 1,
    "ca": 0,
    "thal": 3,
}


class PatientFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [EXAMPLE_PATIENT]})

    age: int = Field(..., ge=R["age"][0], le=R["age"][1], description="Age in years")
    sex: Literal[0, 1] = Field(..., description="0 = female, 1 = male")
    cp: Literal[1, 2, 3, 4] = Field(
        ..., description="Chest pain: 1 typical, 2 atypical, 3 non-anginal, 4 asymptomatic")
    trestbps: float = Field(..., ge=R["trestbps"][0], le=R["trestbps"][1],
                            description="Resting blood pressure, mm Hg")
    chol: float = Field(..., ge=R["chol"][0], le=R["chol"][1],
                        description="Serum cholesterol, mg/dl")
    fbs: Literal[0, 1] = Field(..., description="Fasting blood sugar > 120 mg/dl")
    restecg: Literal[0, 1, 2] = Field(..., description="Resting ECG: 0 normal, 1 ST-T, 2 LVH")
    thalach: float = Field(..., ge=R["thalach"][0], le=R["thalach"][1],
                           description="Maximum heart rate achieved")
    exang: Literal[0, 1] = Field(..., description="Exercise-induced angina")
    oldpeak: float = Field(..., ge=R["oldpeak"][0], le=R["oldpeak"][1],
                           description="ST depression induced by exercise")
    slope: Literal[1, 2, 3] = Field(..., description="Peak ST slope: 1 up, 2 flat, 3 down")
    ca: int = Field(..., ge=R["ca"][0], le=R["ca"][1],
                    description="Major vessels coloured by fluoroscopy")
    thal: Literal[3, 6, 7] = Field(..., description="3 normal, 6 fixed defect, 7 reversible")


class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="1 = heart disease predicted, 0 = not")
    label: str
    probability_disease: float = Field(..., ge=0, le=1)
    confidence: float = Field(..., ge=0.5, le=1,
                              description="Probability of the predicted class")
    risk_band: Literal["low", "moderate", "high"]
    model_version: str
    request_id: str


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    instances: list[PatientFeatures] = Field(..., min_length=1, max_length=100)


class BatchResponse(BaseModel):
    predictions: list[PredictionResponse]
