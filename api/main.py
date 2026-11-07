"""Actividad 2: API de inferencia con FastAPI. Sirve el modelo con alias `champion`.

Uso (desde la raíz del repositorio, con MLFLOW_TRACKING_URI definida):
    poetry run uvicorn api.main:app --port 8000
Documentación interactiva: http://localhost:8000/docs

El champion se carga al arrancar. Si el entrenamiento asigna uno nuevo, reinicia la API.
"""

import os
from contextlib import asynccontextmanager

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI
from mlflow.tracking import MlflowClient
from pydantic import BaseModel, ConfigDict, Field

from src.load_data import load_config

cfg = load_config()
NOMBRE_MODELO = cfg["registro"]["nombre_modelo"]
UMBRAL = cfg["metricas"]["umbral"]
estado: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", cfg["mlflow"]["tracking_uri"]))
    estado["modelo"] = mlflow.sklearn.load_model(f"models:/{NOMBRE_MODELO}@champion")
    estado["version"] = MlflowClient().get_model_version_by_alias(NOMBRE_MODELO, "champion").version
    yield
    estado.clear()


app = FastAPI(
    title="API Food Prediction",
    description="Predice si un cliente vuelve a pedir comida en línea (Output = Yes) con el modelo champion.",
    lifespan=lifespan,
)


class Cliente(BaseModel):
    """Datos de un cliente. Los alias coinciden con las columnas del CSV."""

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={"examples": [{
            "Age": 22, "Gender": "Female", "Marital Status": "Single", "Occupation": "Student",
            "Monthly Income": "No Income", "Educational Qualifications": "Graduate", "Family size": 4,
            "Customer Type": "Frequent", "latitude": 12.9766, "longitude": 77.5993,
            "Pin code": 560001, "Feedback": "Positive",
        }]},
    )

    Age: int
    Gender: str
    marital_status: str = Field(alias="Marital Status")
    Occupation: str
    monthly_income: str = Field(alias="Monthly Income")
    educational_qualifications: str = Field(alias="Educational Qualifications")
    family_size: int = Field(alias="Family size")
    customer_type: str = Field(alias="Customer Type")
    latitude: float
    longitude: float
    pin_code: int = Field(alias="Pin code")
    Feedback: str


@app.get("/health")
def health():
    return {"estado": "ok", "modelo": NOMBRE_MODELO, "version": estado.get("version")}


@app.post("/predict")
def predict(cliente: Cliente):
    modelo = estado["modelo"]
    # Solo las columnas con las que se entrenó el champion.
    datos = pd.DataFrame([cliente.model_dump(by_alias=True)])[list(modelo.feature_names_in_)]
    proba = float(modelo.predict_proba(datos)[0, 1])
    return {
        "probabilidad_output": round(proba, 4),
        "prediccion": int(proba >= UMBRAL),
        "vuelve_a_pedir": "Yes" if proba >= UMBRAL else "No",
        "version_modelo": estado["version"],
    }
