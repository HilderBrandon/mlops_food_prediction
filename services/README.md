# Servicios: MLflow y Prefect

Usa **una** de las dos opciones, no las dos a la vez (ambas usan los puertos 5000 y 4200).

## Opción A: Docker (recomendada)

Con Docker Desktop abierto, desde esta carpeta (`services/`):

```powershell
docker compose up -d      # levanta MLflow y Prefect
docker compose ps         # ambos deben estar "running"
docker compose logs -f    # ver logs (Ctrl+C para salir)
docker compose down       # apagar
```

- MLflow: http://localhost:5000 (datos en `services/mlflow_store/`, fuera de Git)
- Prefect: http://localhost:4200 (datos en `services/prefect_store/`, fuera de Git)

## Opción B: sin Docker

Abre dos terminales en la raíz del repositorio:

```powershell
# Terminal 1: MLflow
poetry run mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db --artifacts-destination ./mlflow_store

# Terminal 2: Prefect
poetry run prefect server start --port 4200
```
