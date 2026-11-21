"""Actividad 3: pipeline de drift de features con Evidently.

Pasos: referencia (parte de entrenamiento) -> lote -> reporte de Evidently -> guardar reporte
-> alerta -> registrar en MLflow.

- Compara la distribución de cada variable de entrada del modelo (configs/features.yaml) en el
  lote frente a los datos de entrenamiento: Kolmogorov-Smirnov para numéricas y chi-cuadrado
  para categóricas (configurable en `drift`).
- Guarda un reporte HTML de Evidently y un resumen JSON por lote en `drift.carpeta_reportes`.
- Si la fracción de variables con drift supera `drift.umbral_fraccion`, emite una ALERTA en el
  log de Prefect y la deja en el artefacto del flow.

Uso (desde la raíz del repositorio, después de la inferencia):
    poetry run python flows/drift_features_flow.py                 # último lote
    poetry run python flows/drift_features_flow.py data/lotes/lote_...csv
"""

import json
import os
import sys
from pathlib import Path

import mlflow
import pandas as pd
from prefect import flow, get_run_logger, task
from prefect.artifacts import create_markdown_artifact
from prefect.cache_policies import NO_CACHE

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.datos import particionar  # noqa: E402
from src.drift import calcular_drift_features, tabla_markdown_features, ultimo_lote  # noqa: E402
from src.load_data import load_config, load_data  # noqa: E402
from src.modelo import load_features, resolver_features  # noqa: E402


@task(cache_policy=NO_CACHE)
def cargar_referencia(cfg: dict) -> pd.DataFrame:
    entrenamiento, _ = particionar(load_data(config=cfg), cfg)
    print(f"Referencia: parte de entrenamiento ({len(entrenamiento)} filas)")
    return entrenamiento


@task(cache_policy=NO_CACHE)
def cargar_lote(cfg: dict, ruta_lote: str | None) -> tuple[pd.DataFrame, Path]:
    ruta = Path(ruta_lote) if ruta_lote else ultimo_lote(ROOT / cfg["drift"]["carpeta_lotes"])
    lote = pd.read_csv(ruta)
    print(f"Lote: {ruta.name} ({len(lote)} filas)")
    return lote, ruta


@task(cache_policy=NO_CACHE)
def calcular(referencia: pd.DataFrame, lote: pd.DataFrame, cfg: dict):
    _, numericas, categoricas = resolver_features(load_features())
    return calcular_drift_features(referencia, lote, numericas, categoricas, cfg["drift"])


@task(cache_policy=NO_CACHE)
def guardar_reporte(snapshot, resumen: dict, ruta_lote: Path, cfg: dict) -> tuple[str, str]:
    carpeta = ROOT / cfg["drift"]["carpeta_reportes"]
    carpeta.mkdir(parents=True, exist_ok=True)
    html = carpeta / f"{ruta_lote.stem}_features.html"
    resumen_json = carpeta / f"{ruta_lote.stem}_features.json"
    snapshot.save_html(str(html))
    resumen_json.write_text(json.dumps({"lote": ruta_lote.name, **resumen}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Reporte de Evidently: {html}")
    return str(html), str(resumen_json)


@task(cache_policy=NO_CACHE)
def alertar(resumen: dict, ruta_lote: Path) -> None:
    logger = get_run_logger()
    mensaje = (
        f"{len(resumen['variables_con_drift'])} de {resumen['n_variables']} variables con drift "
        f"({resumen['fraccion_con_drift']:.0%}); umbral {resumen['umbral_fraccion']:.0%}"
    )
    if resumen["alerta"]:
        logger.warning(f"ALERTA de drift de features en {ruta_lote.name}: {mensaje}. "
                       f"Variables: {', '.join(resumen['variables_con_drift'])}")
    else:
        logger.info(f"Sin alerta en {ruta_lote.name}: {mensaje}")


@task(cache_policy=NO_CACHE)
def registrar(resumen: dict, ruta_lote: Path, html: str, resumen_json: str, cfg: dict) -> None:
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", cfg["mlflow"]["tracking_uri"]))
    mlflow.set_experiment(cfg["mlflow"]["experimento"])
    with mlflow.start_run(run_name=f"drift-features-{ruta_lote.stem}"):
        mlflow.set_tag("fase", "drift-features")
        mlflow.set_tag("lote", ruta_lote.name)
        mlflow.set_tag("alerta", str(resumen["alerta"]))
        mlflow.log_param("umbral_fraccion", resumen["umbral_fraccion"])
        mlflow.log_metric("fraccion_con_drift", resumen["fraccion_con_drift"])
        mlflow.log_metric("variables_con_drift", len(resumen["variables_con_drift"]))
        mlflow.log_artifact(html)
        mlflow.log_artifact(resumen_json)


@flow(name="drift-features", log_prints=True)
def drift_features_flow(ruta_lote: str | None = None) -> dict:
    cfg = load_config()
    referencia = cargar_referencia(cfg)
    lote, ruta = cargar_lote(cfg, ruta_lote)
    snapshot, resumen = calcular(referencia, lote, cfg)
    html, resumen_json = guardar_reporte(snapshot, resumen, ruta, cfg)
    alertar(resumen, ruta)
    registrar(resumen, ruta, html, resumen_json, cfg)
    create_markdown_artifact(
        key="drift-features",
        markdown=(
            f"# Drift de features: {ruta.name}\n\n"
            f"{'**ALERTA:** ' if resumen['alerta'] else ''}"
            f"{len(resumen['variables_con_drift'])} de {resumen['n_variables']} variables con drift "
            f"({resumen['fraccion_con_drift']:.0%}, umbral {resumen['umbral_fraccion']:.0%}).\n\n"
            f"{tabla_markdown_features(resumen)}\n\nReporte: `{html}`\n"
        ),
    )
    return resumen


if __name__ == "__main__":
    drift_features_flow(sys.argv[1] if len(sys.argv) > 1 else None)
