"""Actividad 2: API de inferencia con FastAPI. Sirve el modelo con alias `champion`.

Uso (desde la raíz del repositorio, con MLFLOW_TRACKING_URI definida):
    poetry run uvicorn api.main:app --port 8000
Documentación interactiva: http://localhost:8000/docs

El champion se carga al arrancar. Si el entrenamiento asigna uno nuevo, reinicia la API.

La carga está en `cargar_champion()` para que los tests la reemplacen por un modelo falso
y prueben la API sin MLflow (tests/test_api.py).
"""

import os
from contextlib import asynccontextmanager

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from mlflow.tracking import MlflowClient
from pydantic import BaseModel, ConfigDict, Field

from src.load_data import load_config

cfg = load_config()
NOMBRE_MODELO = cfg["registro"]["nombre_modelo"]
UMBRAL = cfg["metricas"]["umbral"]
estado: dict = {}


def cargar_champion() -> tuple[object, str]:
    """Carga el modelo con alias `champion` desde MLflow. Devuelve (modelo, versión)."""
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", cfg["mlflow"]["tracking_uri"]))
    modelo = mlflow.sklearn.load_model(f"models:/{NOMBRE_MODELO}@champion")
    version = MlflowClient().get_model_version_by_alias(NOMBRE_MODELO, "champion").version
    return modelo, str(version)


@asynccontextmanager
async def lifespan(app: FastAPI):
    estado["modelo"], estado["version"] = cargar_champion()
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
            "Pin code": 560001,
        }]},
    )

    Age: int = Field(ge=15, le=80)
    Gender: str
    marital_status: str = Field(alias="Marital Status")
    Occupation: str
    monthly_income: str = Field(alias="Monthly Income")
    educational_qualifications: str = Field(alias="Educational Qualifications")
    family_size: int = Field(alias="Family size", ge=1, le=15)
    customer_type: str = Field(alias="Customer Type")
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    pin_code: int = Field(alias="Pin code")
    # Opcional: es la opinión del cliente DESPUÉS del pedido, no existe al predecir y el
    # modelo ya no la usa (configs/features.yaml, `excluidas`). Se acepta para no romper clientes antiguos.
    Feedback: str | None = None


@app.get("/health")
def health():
    return {"estado": "ok", "modelo": NOMBRE_MODELO, "version": estado.get("version")}


@app.post("/predict")
def predict(cliente: Cliente):
    if "modelo" not in estado:
        raise HTTPException(status_code=503, detail="El modelo champion no está cargado")
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
