"""Actividad 3: pipeline de drift de evento (variable objetivo `Output`).

Pasos: referencia (parte de entrenamiento) -> lote -> etiquetas reales -> prueba chi-cuadrado
-> guardar reporte -> alerta -> registrar en MLflow.

- Compara la tasa de clientes que vuelven a pedir (`Output = Yes`) en el lote con la del
  entrenamiento.
- El lote debe traer las etiquetas reales. En producción llegan días después de la predicción;
  si no han llegado, el flow falla con un mensaje claro en vez de calcular algo sin sentido.
- Si el p-valor es menor que `drift.alpha_objetivo`, emite una ALERTA en el log de Prefect.

Uso (desde la raíz del repositorio, después de la inferencia):
    poetry run python flows/drift_objetivo_flow.py                 # último lote
    poetry run python flows/drift_objetivo_flow.py data/lotes/lote_...csv
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
from flows.drift_features_flow import cargar_lote, cargar_referencia  # noqa: E402
from src.drift import drift_objetivo, etiquetas_del_lote  # noqa: E402
from src.load_data import load_config  # noqa: E402
from src.modelo import preparar_objetivo  # noqa: E402


@task(cache_policy=NO_CACHE)
def comparar(referencia: pd.DataFrame, lote: pd.DataFrame, cfg: dict) -> dict:
    target, positiva = cfg["data"]["target"], cfg["data"]["clase_positiva"]
    y_lote, sin_etiqueta = etiquetas_del_lote(lote, target, positiva)  # falla si no hay etiquetas
    if sin_etiqueta:
        get_run_logger().warning(f"{sin_etiqueta} filas aún sin etiqueta: se comparan solo las {len(y_lote)} etiquetadas")
    y_ref = preparar_objetivo(referencia, target, positiva)
    return {**drift_objetivo(y_ref, y_lote, cfg["drift"]["alpha_objetivo"]), "filas_sin_etiqueta": sin_etiqueta}


@task(cache_policy=NO_CACHE)
def guardar_reporte(resultado: dict, ruta_lote: Path, cfg: dict) -> str:
    carpeta = ROOT / cfg["drift"]["carpeta_reportes"]
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta = carpeta / f"{ruta_lote.stem}_objetivo.json"
    ruta.write_text(json.dumps({"lote": ruta_lote.name, **resultado}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Reporte de drift de evento: {ruta}")
    return str(ruta)


@task(cache_policy=NO_CACHE)
def alertar(resultado: dict, ruta_lote: Path) -> None:
    logger = get_run_logger()
    mensaje = (
        f"tasa de 'Yes' {resultado['tasa_referencia']:.1%} (entrenamiento) -> {resultado['tasa_lote']:.1%} (lote), "
        f"p-valor {resultado['p_valor']:.4f}"
    )
    if resultado["drift"]:
        logger.warning(f"ALERTA de drift de evento en {ruta_lote.name}: {mensaje}")
    else:
        logger.info(f"Sin drift de evento en {ruta_lote.name}: {mensaje}")


@task(cache_policy=NO_CACHE)
def registrar(resultado: dict, ruta_lote: Path, ruta_reporte: str, cfg: dict) -> None:
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", cfg["mlflow"]["tracking_uri"]))
    mlflow.set_experiment(cfg["mlflow"]["experimento"])
    with mlflow.start_run(run_name=f"drift-objetivo-{ruta_lote.stem}"):
        mlflow.set_tag("fase", "drift-objetivo")
        mlflow.set_tag("lote", ruta_lote.name)
        mlflow.set_tag("alerta", str(resultado["drift"]))
        mlflow.log_param("alpha", resultado["alpha"])
        mlflow.log_metric("tasa_referencia", resultado["tasa_referencia"])
        mlflow.log_metric("tasa_lote", resultado["tasa_lote"])
        mlflow.log_metric("p_valor", resultado["p_valor"])
        mlflow.log_artifact(ruta_reporte)


@flow(name="drift-objetivo", log_prints=True)
def drift_objetivo_flow(ruta_lote: str | None = None) -> dict:
    cfg = load_config()
    referencia = cargar_referencia(cfg)
    lote, ruta = cargar_lote(cfg, ruta_lote)
    resultado = comparar(referencia, lote, cfg)
    ruta_reporte = guardar_reporte(resultado, ruta, cfg)
    alertar(resultado, ruta)
    registrar(resultado, ruta, ruta_reporte, cfg)
    create_markdown_artifact(
        key="drift-objetivo",
        markdown=(
            f"# Drift de evento: {ruta.name}\n\n"
            f"| | Entrenamiento | Lote |\n|---|---|---|\n"
            f"| Filas | {resultado['n_referencia']} | {resultado['n_lote']} |\n"
            f"| Tasa de 'Yes' | {resultado['tasa_referencia']:.1%} | {resultado['tasa_lote']:.1%} |\n\n"
            f"p-valor (chi-cuadrado): **{resultado['p_valor']:.4f}** (alpha {resultado['alpha']}). "
            f"{'**ALERTA: drift de evento.**' if resultado['drift'] else 'Sin drift de evento.'}\n"
        ),
    )
    return resultado


if __name__ == "__main__":
    drift_objetivo_flow(sys.argv[1] if len(sys.argv) > 1 else None)
