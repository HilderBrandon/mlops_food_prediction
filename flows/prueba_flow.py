"""Flow corto para comprobar que Prefect recibe runs y MLflow registra experimentos.

Uso (desde la raíz del repositorio, con los servicios levantados):

    $env:PREFECT_API_URL = "http://localhost:4200/api"
    $env:MLFLOW_TRACKING_URI = "http://localhost:5000"
    poetry run python flows/prueba_flow.py
"""

import os

import mlflow
from prefect import flow, task

EXPERIMENTO = "prueba-validacion"
RUN_NAME = "prueba-mlflow"


@task
def sumar(n: int) -> int:
    return sum(range(1, n + 1))


@task
def registrar_en_mlflow(n: int, suma: int) -> str:
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment(EXPERIMENTO)
    with mlflow.start_run(run_name=RUN_NAME) as run:
        mlflow.log_param("n", n)
        mlflow.log_metric("suma", suma)
        return run.info.run_id


@flow(name="prueba", log_prints=True)
def prueba(n: int = 5) -> int:
    suma = sumar(n)
    print(f"Suma de 1 a {n} = {suma}")
    run_id = registrar_en_mlflow(n, suma)
    print(f"Run de MLflow registrado: {run_id} (experimento '{EXPERIMENTO}')")
    return suma


if __name__ == "__main__":
    prueba()
