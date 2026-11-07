"""Actividad 2: pipeline de inferencia (se despliega para correr una vez al día).

Pasos: cargar champion -> ingesta del lote -> predicción -> guardar CSV -> registrar en MLflow.

El modelo se carga por el alias `champion`, no por número de versión: cuando el entrenamiento
asigna un champion nuevo, la inferencia lo usa sin cambiar código.

Uso (desde la raíz del repositorio, con un champion ya registrado):
    $env:PREFECT_API_URL = "http://localhost:4200/api"
    $env:MLFLOW_TRACKING_URI = "http://localhost:5000"
    poetry run python flows/inferencia_flow.py
"""

import os
import sys
from datetime import date
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient
from prefect import flow, task
from prefect.cache_policies import NO_CACHE

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.load_data import load_config, load_data  # noqa: E402


@task(cache_policy=NO_CACHE)
def cargar_champion(cfg: dict):
    nombre = cfg["registro"]["nombre_modelo"]
    version = MlflowClient().get_model_version_by_alias(nombre, "champion")
    modelo = mlflow.sklearn.load_model(f"models:/{nombre}@champion")
    print(f"Modelo cargado: {nombre} versión {version.version}")
    return modelo, version.version, version.run_id


@task(cache_policy=NO_CACHE)
def ingesta(cfg: dict) -> pd.DataFrame:
    """Simula clientes nuevos con una muestra aleatoria del CSV (cambia en cada ejecución)."""
    df = load_data(config=cfg)
    n = min(cfg["inferencia"]["tamano_lote"], len(df))
    print(f"Lote de {n} clientes")
    return df.sample(n=n)


@task(cache_policy=NO_CACHE)
def predecir(modelo, lote: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    # Usa las mismas columnas con las que se entrenó el champion, aunque features.yaml haya cambiado.
    proba = modelo.predict_proba(lote[list(modelo.feature_names_in_)])[:, 1]
    resultado = lote.copy()
    resultado["probabilidad_output"] = proba
    resultado["prediccion"] = (proba >= cfg["metricas"]["umbral"]).astype(int)
    return resultado


@task(cache_policy=NO_CACHE)
def guardar(resultado: pd.DataFrame, cfg: dict) -> str:
    carpeta = ROOT / cfg["inferencia"]["carpeta_salida"]
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta = carpeta / f"prediccion_{date.today().isoformat()}.csv"
    resultado.to_csv(ruta, index=False)
    print(f"Predicciones guardadas en {ruta}")
    return str(ruta)


@task(cache_policy=NO_CACHE)
def registrar_inferencia(version: str, run_id_entrenamiento: str, ruta: str, resultado: pd.DataFrame) -> None:
    with mlflow.start_run(run_name=f"inferencia-{date.today().isoformat()}"):
        mlflow.set_tag("fase", "inferencia")
        mlflow.set_tag("modelo_version", version)
        mlflow.set_tag("modelo_run_id", run_id_entrenamiento)
        mlflow.log_metric("filas_predichas", len(resultado))
        mlflow.log_metric("probabilidad_media", float(resultado["probabilidad_output"].mean()))
        mlflow.log_metric("tasa_positivos", float(resultado["prediccion"].mean()))
        mlflow.log_artifact(ruta)


@flow(name="inferencia-food", log_prints=True)
def inferencia_flow() -> str:
    cfg = load_config()
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", cfg["mlflow"]["tracking_uri"]))
    mlflow.set_experiment(cfg["mlflow"]["experimento"])

    modelo, version, run_id = cargar_champion(cfg)
    lote = ingesta(cfg)
    resultado = predecir(modelo, lote, cfg)
    ruta = guardar(resultado, cfg)
    registrar_inferencia(version, run_id, ruta, resultado)
    return ruta


if __name__ == "__main__":
    inferencia_flow()
