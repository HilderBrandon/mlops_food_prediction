"""Actividad 2: pipeline de entrenamiento.

Pasos: ingesta -> entrenamiento -> evaluación -> registro.

- Entrena cada modelo de `candidatos` (configs/config.yaml) con las variables elegidas en
  configs/features.yaml. Cada candidato crea un run en MLflow con el tag `fase = entrenamiento`.
- El mejor candidato (mayor AUC) se compara con el champion actual. Solo se registra como
  versión nueva y recibe el alias `champion` si supera `registro.umbral_auc` y el AUC del
  champion vigente (más `registro.mejora_minima`). Si no, el run queda en MLflow y el alias no cambia.

Uso (desde la raíz del repositorio, con los servicios levantados):
    $env:PREFECT_API_URL = "http://localhost:4200/api"
    $env:MLFLOW_TRACKING_URI = "http://localhost:5000"
    poetry run python flows/entrenamiento_flow.py
"""

import os
import sys
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.exceptions import MlflowException
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from prefect import flow, task
from prefect.artifacts import create_markdown_artifact
from prefect.cache_policies import NO_CACHE
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.load_data import load_config, load_data  # noqa: E402
from src.modelo import (  # noqa: E402
    calcular_metricas,
    construir_modelo,
    debe_promover,
    load_features,
    preparar_objetivo,
    resolver_features,
)

# Los modelos los entrenamos nosotros: se confía en los tipos internos de sklearn que usan
# los árboles (random forest, gradient boosting) y la calibración del SVM.
SKOPS_TRUSTED = [
    "sklearn.tree._tree.Tree",
    "sklearn.calibration._CalibratedClassifier",
    "sklearn.calibration._SigmoidCalibration",
]


def configurar_mlflow(cfg: dict) -> None:
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", cfg["mlflow"]["tracking_uri"]))
    mlflow.set_experiment(cfg["mlflow"]["experimento"])


@task(cache_policy=NO_CACHE)
def ingesta(cfg: dict) -> pd.DataFrame:
    df = load_data(config=cfg)
    print(f"Ingesta: {df.shape[0]} filas x {df.shape[1]} columnas")
    return df


@task(cache_policy=NO_CACHE)
def dividir(df: pd.DataFrame, cfg: dict, variables: list[str]) -> tuple:
    y = preparar_objetivo(df, cfg["data"]["target"], cfg["data"]["clase_positiva"])
    X_train, X_test, y_train, y_test = train_test_split(
        df[variables], y,
        test_size=cfg["division"]["test_size"],
        random_state=cfg["division"]["random_state"],
        stratify=y,
    )
    print(f"Train: {len(X_train)} filas | Test: {len(X_test)} filas | % positivos test: {y_test.mean():.2%}")
    return X_train, X_test, y_train, y_test


@task(cache_policy=NO_CACHE)
def entrenar(candidato: dict, numericas: list[str], categoricas: list[str], X_train, y_train):
    modelo = construir_modelo(candidato["tipo"], candidato.get("parametros"), numericas, categoricas)
    modelo.fit(X_train, y_train)
    print(f"Entrenado {candidato['nombre_run']} ({candidato['tipo']})")
    return modelo


@task(cache_policy=NO_CACHE)
def evaluar(modelo, candidato: dict, alias: str, cfg: dict, X_test, y_test) -> dict:
    """Calcula métricas en test y crea el run de entrenamiento en MLflow."""
    umbral = cfg["metricas"]["umbral"]
    proba = modelo.predict_proba(X_test)[:, 1]
    metricas = calcular_metricas(y_test, proba, umbral)

    configurar_mlflow(cfg)
    with mlflow.start_run(run_name=f"{candidato['nombre_run']}-{alias.lower()}") as run:
        mlflow.set_tag("fase", "entrenamiento")
        mlflow.log_params({
            "modelo": candidato["tipo"],
            **(candidato.get("parametros") or {}),
            "semilla": cfg["division"]["random_state"],
            "test_size": cfg["division"]["test_size"],
            "umbral": umbral,
            "features": alias,
            "n_features": X_test.shape[1],
        })
        mlflow.log_metrics(metricas)
        mlflow.log_text("\n".join(X_test.columns), "features.txt")
        info = mlflow.sklearn.log_model(
            modelo,
            name="model",
            signature=infer_signature(X_test, proba),
            input_example=X_test.head(3),
            skops_trusted_types=SKOPS_TRUSTED,
        )
    print(f"{candidato['nombre_run']}: {metricas} | run {run.info.run_id}")
    return {"nombre": candidato["nombre_run"], "run_id": run.info.run_id, "model_uri": info.model_uri, "metricas": metricas}


@task(cache_policy=NO_CACHE)
def registrar(mejor: dict, cfg: dict) -> str | None:
    """Registra el mejor modelo como nueva versión champion solo si mejora."""
    configurar_mlflow(cfg)
    client = MlflowClient()
    nombre = cfg["registro"]["nombre_modelo"]
    auc = mejor["metricas"]["auc"]

    try:
        champion = client.get_model_version_by_alias(nombre, "champion")
        auc_champion = client.get_run(champion.run_id).data.metrics["auc"]
        print(f"Champion actual: versión {champion.version} (AUC {auc_champion:.4f})")
    except MlflowException:
        champion, auc_champion = None, None
        print("Todavía no hay champion")

    promover, motivo = debe_promover(
        auc, auc_champion, cfg["registro"]["umbral_auc"], cfg["registro"].get("mejora_minima", 0.0)
    )
    print(motivo)
    if not promover:
        print("El alias champion no cambia")
        return None

    version = mlflow.register_model(mejor["model_uri"], nombre)
    client.set_registered_model_alias(nombre, "champion", version.version)
    client.set_model_version_tag(nombre, version.version, "estado", "aprobado")
    print(f"Versión {version.version} registrada como champion (AUC {auc:.4f})")
    return version.version


@flow(name="entrenamiento-food", log_prints=True)
def entrenamiento_flow() -> str | None:
    cfg = load_config()
    alias, numericas, categoricas = resolver_features(load_features())
    print(f"Features '{alias}': {len(numericas) + len(categoricas)} variables")

    df = ingesta(cfg)
    X_train, X_test, y_train, y_test = dividir(df, cfg, numericas + categoricas)

    resultados = []
    for candidato in cfg["candidatos"]:
        modelo = entrenar(candidato, numericas, categoricas, X_train, y_train)
        resultados.append(evaluar(modelo, candidato, alias, cfg, X_test, y_test))

    mejor = max(resultados, key=lambda r: r["metricas"]["auc"])
    print(f"Mejor candidato: {mejor['nombre']} (AUC {mejor['metricas']['auc']:.4f})")
    version = registrar(mejor, cfg)

    columnas = ["auc", "accuracy", "precision", "recall", "f1"]
    ranking = sorted(resultados, key=lambda r: r["metricas"]["auc"], reverse=True)
    filas = "\n".join(
        f"| {i} | {'**' + r['nombre'] + '**' if r is mejor else r['nombre']} | "
        + " | ".join(f"{r['metricas'][c]:.4f}" for c in columnas)
        + f" | `{r['run_id'][:8]}` |"
        for i, r in enumerate(ranking, 1)
    )
    create_markdown_artifact(
        key="entrenamiento-food-resumen",
        markdown=(
            f"# Entrenamiento ({alias})\n\n"
            f"| # | Candidato | {' | '.join(c.upper() for c in columnas)} | Run MLflow |\n"
            f"|---|---|{'---|' * len(columnas)}---|\n{filas}\n\n"
            f"Mejor: **{mejor['nombre']}**. Nueva versión champion: **{version or 'ninguna'}**\n"
        ),
    )
    return version


if __name__ == "__main__":
    entrenamiento_flow()
