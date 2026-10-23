# MLOps Project: Online Food Ordering Behavior Prediction

Proyecto del curso de **MLOps** de la **Maestría en Ciencia de Datos (Universidad Santo Tomás)**. Implementa un flujo reproducible para predecir si un cliente vuelve a pedir (`Output`) con el dataset [Online Food Ordering Dataset](https://www.kaggle.com/datasets/srisyra02/online-food-ordering-dataset).

## Requisitos

| Herramienta | Versión | Comprobar |
|---|---|---|
| Python | 3.12 a 3.14 | `python --version` |
| Poetry | 2.x | `poetry --version` |
| Git | reciente | `git --version` |
| Docker Desktop | reciente | `docker --version` |

## Estructura

```text
mlops_food_prediction/
├── configs/config.yaml      # Rutas, columnas esperadas e hiperparámetros
├── data/
│   ├── raw/                 # CSV de Kaggle (no se versiona)
│   └── processed/           # Datos transformados (no se versiona)
├── flows/
│   ├── prueba_flow.py       # Flow de validación de Prefect + MLflow
│   └── food_flow.py         # Pipeline de entrenamiento del proyecto (opcional A1)
├── models/                  # Modelos locales (no se versiona)
├── notebooks/01_eda.ipynb   # Análisis exploratorio
├── services/                # docker compose de MLflow (5000) y Prefect (4200)
├── src/
│   ├── load_data.py         # Cargue y validación del dataset
│   ├── preprocessing.py
│   ├── train.py
│   └── evaluate.py
├── tests/                   # Pruebas con pytest
├── pyproject.toml           # Dependencias declaradas (versiones fijadas)
└── poetry.lock              # Versiones exactas instaladas (no editar a mano)
```

Cada carpeta lleva un `__init__.py` para que Git la versione aunque esté vacía.

## 1. Preparar el entorno

```powershell
git clone https://github.com/HilderBrandon/mlops_food_prediction.git
cd mlops_food_prediction
poetry env use 3.12
poetry install
poetry run python -c "import prefect, mlflow, sklearn; print('ok')"
```

`pyproject.toml` declara lo que se necesita y `poetry.lock` registra lo que se instaló. Si cambias dependencias usa `poetry add "paquete==x.y.z"` y sube ambos archivos. Si Poetry avisa de que el lock no coincide, `poetry lock` lo ejecuta solo quien cambió las dependencias.

> `fastapi` está fijado en 0.154.0 porque las versiones más nuevas rompen el servidor de Prefect 3.8.8 cuando se ejecuta sin Docker.

## 2. Datos

1. Descarga el CSV desde [Kaggle](https://www.kaggle.com/datasets/srisyra02/online-food-ordering-dataset).
2. Guárdalo como `data/raw/online food delivery dataset.csv` (388 filas).
3. Comprueba el cargue:

```powershell
poetry run python src/load_data.py
```

`load_data()` elimina la columna sin nombre que crea la coma final de cada fila, valida que exista la columna objetivo `Output` y avisa si falta alguna columna esperada (lista en `configs/config.yaml`).

## 3. Levantar MLflow y Prefect

```powershell
cd services
docker compose up -d
cd ..
```

- MLflow: http://localhost:5000
- Prefect: http://localhost:4200

> **macOS:** el puerto 5000 lo ocupa el Receptor AirPlay. Desactívalo en Ajustes del Sistema → General → AirDrop y Handoff → Receptor AirPlay, o MLflow no podrá arrancar.

Si Docker no funciona, usa la opción B de [services/README.md](services/README.md).

## 4. Validar los servicios con el flow de prueba

Desde la raíz del repositorio, en la misma terminal:

```powershell
$env:PREFECT_API_URL = "http://localhost:4200/api"
$env:MLFLOW_TRACKING_URI = "http://localhost:5000"
poetry run python flows/prueba_flow.py
```

En macOS/Linux: `export PREFECT_API_URL=http://localhost:4200/api` y `export MLFLOW_TRACKING_URI=http://localhost:5000`.

Resultado esperado:

- En http://localhost:4200/runs, el run del flow **prueba** queda en *Completed* y en sus logs aparece `Suma de 1 a 5 = 15`.
- En http://localhost:5000, el experimento **prueba-validacion** tiene un run **prueba-mlflow** con el parámetro `n` y la métrica `suma`.

## 5. Pipeline de entrenamiento (opcional)

Con las mismas variables de entorno y el CSV en `data/raw/`:

```powershell
poetry run python flows/food_flow.py
```

El flow **food-prediction** tiene tres tareas: cargar datos (registra filas y columnas en los logs y en MLflow), entrenar un RandomForest y registrar parámetros, métricas (accuracy, recall, f1, AUC) y modelo en el experimento `food-prediction` de MLflow, y crear en Prefect un artefacto Markdown con el resumen.

## Pruebas

```powershell
poetry run pytest
```
