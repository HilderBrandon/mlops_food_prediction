"""Pipeline de entrenamiento del proyecto (opcional en la Actividad 1).

Tareas: cargar datos -> entrenar y registrar en MLflow -> resumen en Prefect.

    $env:PREFECT_API_URL = "http://localhost:4200/api"
    $env:MLFLOW_TRACKING_URI = "http://localhost:5000"
    poetry run python flows/food_flow.py
"""

import os
import sys
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from prefect import flow, task
from prefect.artifacts import create_markdown_artifact
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.load_data import load_config, load_data  # noqa: E402


@task
def cargar_datos(config: dict) -> pd.DataFrame:
    df = load_data(config=config)
    print(f"Dataset cargado: {df.shape[0]} filas x {df.shape[1]} columnas")
    return df


@task
def entrenar(df: pd.DataFrame, config: dict) -> dict:
    target = config["data"]["target"]
    tr = config["train"]

    X = df.drop(columns=[target])
    y = (df[target].astype(str).str.strip().str.lower() == "yes").astype(int)

    categoricas = X.select_dtypes(exclude="number").columns.tolist()
    modelo = Pipeline([
        ("prep", ColumnTransformer(
            [("cat", OneHotEncoder(handle_unknown="ignore"), categoricas)],
            remainder="passthrough",
        )),
        ("rf", RandomForestClassifier(n_estimators=tr["n_estimators"], random_state=tr["random_state"])),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=tr["test_size"], random_state=tr["random_state"], stratify=y
    )
    modelo.fit(X_train, y_train)
    pred = modelo.predict(X_test)
    proba = modelo.predict_proba(X_test)[:, 1]
    metricas = {
        "accuracy": accuracy_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "auc": roc_auc_score(y_test, proba),
    }

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment(config["mlflow"]["experiment"])
    with mlflow.start_run(run_name="random-forest") as run:
        mlflow.log_params({**tr, "n_filas": len(df), "n_columnas": df.shape[1]})
        mlflow.log_metrics(metricas)
        # El modelo lo creamos nosotros, así que confiamos en el tipo Tree de sklearn.
        mlflow.sklearn.log_model(modelo, name="model", skops_trusted_types=["sklearn.tree._tree.Tree"])
        run_id = run.info.run_id

    print(f"Métricas: {metricas}")
    return {"metricas": metricas, "run_id": run_id, "n_filas": len(df)}


@task
def resumen(resultado: dict) -> None:
    filas = "\n".join(f"| {k} | {v:.3f} |" for k, v in resultado["metricas"].items())
    create_markdown_artifact(
        key="food-prediction-resumen",
        description="Food prediction",
        markdown=(
            "# Food prediction\n\n"
            f"- Modelo: RandomForestClassifier\n- Filas: {resultado['n_filas']}\n"
            f"- Run de MLflow: `{resultado['run_id']}`\n\n"
            f"| Métrica | Valor |\n|---|---|\n{filas}\n"
        ),
    )


@flow(name="food-prediction", log_prints=True)
def food_prediction() -> dict:
    config = load_config()
    df = cargar_datos(config)
    resultado = entrenar(df, config)
    resumen(resultado)
    return resultado


if __name__ == "__main__":
    food_prediction()
