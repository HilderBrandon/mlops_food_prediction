"""Pipeline de inferencia con drift emulado (actividades 2 y 3). Se despliega para correr una vez al día.

Pasos: cargar champion -> ingesta (lote de la parte de prueba) -> emular drift -> guardar copia
del lote -> predicción -> guardar predicciones -> registrar en MLflow.

- El lote sale de la parte de PRUEBA del dataset (la misma partición del entrenamiento), así
  que son clientes que el modelo no vio al entrenar.
- El nivel de drift se lee de `drift.nivel` en configs/config.yaml (0.0, 0.3 o 0.6). No es un
  parámetro de ejecución: para cambiarlo se edita el YAML.
- La copia del lote (con la etiqueta real `Output`) queda en `drift.carpeta_lotes`. De ahí la
  leen los pipelines de drift. El archivo de predicciones no lleva la etiqueta, como en producción.

El modelo se carga por el alias `champion`, no por número de versión.

Uso (desde la raíz del repositorio, con un champion ya registrado):
    $env:PREFECT_API_URL = "http://localhost:4200/api"
    $env:MLFLOW_TRACKING_URI = "http://localhost:5000"
    poetry run python flows/inferencia_flow.py
"""

import os
import sys
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient
from prefect import flow, task
from prefect.cache_policies import NO_CACHE

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.datos import particionar, validar_datos  # noqa: E402
from src.drift import emular_drift  # noqa: E402
from src.inferencia import marca_tiempo, muestrear_lote, nombre_lote, puntuar  # noqa: E402
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
    """Lote de clientes tomado de la parte de prueba del dataset."""
    df = load_data(config=cfg)
    problemas = validar_datos(df, cfg)
    if problemas:
        raise ValueError("Los datos no pasan la validación: " + "; ".join(problemas))
    _, prueba = particionar(df, cfg)
    lote = muestrear_lote(prueba, cfg["inferencia"]["tamano_lote"], cfg["drift"].get("semilla"))
    print(f"Lote de {len(lote)} clientes tomado de la parte de prueba ({len(prueba)} filas)")
    return lote


@task(cache_policy=NO_CACHE)
def aplicar_drift(lote: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    dcfg = cfg["drift"]
    semilla = dcfg.get("semilla")
    nivel = float(dcfg["nivel"])
    target = cfg["data"]["target"]
    con_drift = emular_drift(
        lote, nivel, dcfg, target,
        cfg["data"]["clase_positiva"], cfg["data"]["clase_negativa"],
        semilla=None if semilla is None else semilla + 1,
    )
    positiva = cfg["data"]["clase_positiva"]
    print(
        f"Drift emulado nivel {nivel}: tasa de '{positiva}' "
        f"{(lote[target] == positiva).mean():.2%} -> {(con_drift[target] == positiva).mean():.2%}"
    )
    return con_drift


@task(cache_policy=NO_CACHE)
def guardar_lote(lote: pd.DataFrame, cfg: dict, marca: str) -> str:
    """Copia del lote que se puntúa, con la etiqueta real, para los pipelines de drift."""
    carpeta = ROOT / cfg["drift"]["carpeta_lotes"]
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta = carpeta / nombre_lote(marca, float(cfg["drift"]["nivel"]))
    lote.assign(nivel_drift=float(cfg["drift"]["nivel"])).to_csv(ruta, index=False)
    print(f"Copia del lote guardada en {ruta}")
    return str(ruta)


@task(cache_policy=NO_CACHE)
def predecir(modelo, lote: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    # La etiqueta real no se le pasa al modelo: en producción todavía no existe.
    sin_etiqueta = lote.drop(columns=[cfg["data"]["target"]], errors="ignore")
    return puntuar(modelo, sin_etiqueta, cfg["metricas"]["umbral"])


@task(cache_policy=NO_CACHE)
def guardar(resultado: pd.DataFrame, cfg: dict, marca: str) -> str:
    carpeta = ROOT / cfg["inferencia"]["carpeta_salida"]
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta = carpeta / f"prediccion_{marca}.csv"
    resultado.to_csv(ruta, index=False)
    print(f"Predicciones guardadas en {ruta}")
    return str(ruta)


@task(cache_policy=NO_CACHE)
def registrar_inferencia(
    version: str, run_id_entrenamiento: str, ruta: str, ruta_lote: str, resultado: pd.DataFrame, nivel: float
) -> None:
    with mlflow.start_run(run_name=f"inferencia-{Path(ruta_lote).stem}"):
        mlflow.set_tag("fase", "inferencia")
        mlflow.set_tag("modelo_version", version)
        mlflow.set_tag("modelo_run_id", run_id_entrenamiento)
        mlflow.set_tag("lote", Path(ruta_lote).name)
        mlflow.log_param("nivel_drift", nivel)
        mlflow.log_metric("filas_predichas", len(resultado))
        mlflow.log_metric("probabilidad_media", float(resultado["probabilidad_output"].mean()))
        mlflow.log_metric("tasa_positivos", float(resultado["prediccion"].mean()))
        mlflow.log_artifact(ruta)
        mlflow.log_artifact(ruta_lote)


@flow(name="inferencia-food", log_prints=True)
def inferencia_flow() -> str:
    cfg = load_config()
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", cfg["mlflow"]["tracking_uri"]))
    mlflow.set_experiment(cfg["mlflow"]["experimento"])
    marca = marca_tiempo()

    modelo, version, run_id = cargar_champion(cfg)
    lote = ingesta(cfg)
    lote = aplicar_drift(lote, cfg)
    ruta_lote = guardar_lote(lote, cfg, marca)
    resultado = predecir(modelo, lote, cfg)
    ruta = guardar(resultado, cfg, marca)
    registrar_inferencia(version, run_id, ruta, ruta_lote, resultado, float(cfg["drift"]["nivel"]))
    return ruta_lote


if __name__ == "__main__":
    inferencia_flow()
