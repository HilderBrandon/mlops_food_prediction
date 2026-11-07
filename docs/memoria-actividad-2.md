# Memoria 2: Pipeline y empaquetado

**Proyecto:** Online Food Ordering Behavior Prediction · **Repositorio:** https://github.com/HilderBrandon/mlops_food_prediction

> Borrador. Completar las secciones marcadas con ✏️ (capturas, contribución individual y uso de IA).

## 1. Objetivo

Pasar del flow de entrenamiento de la actividad 1 a un sistema con dos pipelines en Prefect, entrenamiento e inferencia, conectadas por el Model Registry de MLflow mediante el alias `champion`, y servir ese modelo con una API de FastAPI. El caso de uso es predecir si un cliente vuelve a pedir comida en línea (`Output = Yes`) con el dataset de Kaggle *Online Food Ordering* (388 filas, 78 % positivos).

## 2. Descripción del trabajo

1. **Configuración separada del código.** `configs/config.yaml` contiene datos, división train/test, candidatos con sus hiperparámetros, umbral de clasificación, experimento de MLflow, nombre del modelo, umbral de AUC, mejora mínima, tamaño de lote, carpeta de salida y cron. `configs/features.yaml` elige las variables con `seleccion: ALL | NUMERIC | CUSTOM`.
2. **Lógica del modelo** en `src/modelo.py`, sin Prefect ni MLflow: resolución de variables, pipeline de sklearn (escalado + one-hot + estimador), métricas y regla de promoción. Tiene pruebas unitarias en `tests/test_modelo.py`.
3. **Pipeline de entrenamiento** (`flows/entrenamiento_flow.py`, flow `entrenamiento-food`): `ingesta → dividir → entrenar → evaluar → registrar`. Cada candidato crea un run con tag `fase = entrenamiento`, parámetros (`features`, `n_features`, hiperparámetros, semilla), métricas y el modelo. El mejor candidato solo se registra si supera el AUC mínimo y al champion vigente.
4. **Pipeline de inferencia** (`flows/inferencia_flow.py`, flow `inferencia-food`): carga `models:/food-modelo@champion`, puntúa un lote aleatorio que simula clientes nuevos, guarda `data/predictions/prediccion_<fecha>.csv` y registra un run `fase = inferencia` con la versión y el run de entrenamiento del modelo usado.
5. **Despliegues** (`flows/desplegar.py`): `entrenamiento-food/entrenamiento-manual` sin horario e `inferencia-food/inferencia-diaria` con cron `0 6 * * *`.
6. **API** (`api/main.py`): carga el champion al arrancar y expone `GET /health` y `POST /predict`.

```text
api/main.py · configs/{config,features}.yaml · flows/{entrenamiento,inferencia}_flow.py · flows/desplegar.py · src/modelo.py · tests/test_modelo.py
```

## 3. Decisiones técnicas

| Decisión | Alternativas | Justificación |
|---|---|---|
| Métrica de promoción: AUC | Accuracy, F1 | Las clases están desbalanceadas (78/22). El AUC no depende del umbral y la accuracy de un modelo que siempre dice "Yes" ya es 0,78. |
| Umbral AUC 0,80 y `mejora_minima` configurable | Solo comparar con el champion | Evita promover modelos malos cuando no hay champion y permite exigir mejoras mínimas para no cambiar el modelo por ruido. |
| Varios candidatos por ejecución; solo el mejor compite | Un único modelo | Con 388 filas el entrenamiento es barato y la comparación queda trazada en MLflow. |
| Registrar versión solo si se promueve | Registrar siempre y mover el alias | Lo pide la actividad y mantiene el registry limpio: cada versión fue champion alguna vez. |
| Rutas resueltas desde la raíz del repo (`Path(__file__)`) | Rutas relativas al directorio actual | Los flows y la API funcionan aunque se lancen desde otra carpeta o desde Prefect. |
| Inferencia con `modelo.feature_names_in_` | Leer `features.yaml` | La inferencia usa las columnas del champion aunque `features.yaml` cambie después. |
| División estratificada, `test_size` 0,25 | 0,2 sin estratificar | Con pocos negativos, estratificar garantiza que el test tenga ambas clases (97 filas, ~22 negativos). |
| `cache_policy=NO_CACHE` en las tasks | Caché por defecto de Prefect | Las entradas son DataFrames y modelos; no tiene sentido cachearlos y evita avisos de hashing. |

## 4. Evidencias ✏️

Añadir capturas de:

- MLflow, experimento `food-prediction-mlops`: los 9 runs de entrenamiento (3 candidatos × ALL/NUMERIC/CUSTOM) con tag `fase = entrenamiento`, y el run `inferencia-<fecha>`.
- MLflow, Model Registry: `food-modelo` con versiones 1 y 2 y el alias `champion` en la versión 2.
- Prefect: flow runs de `entrenamiento-food` e `inferencia-food`, y la página *Deployments* con los dos despliegues y el horario diario.
- Swagger de la API (`/docs`) con la predicción de ejemplo.

## 5. Resultados

Métricas en test (97 filas) con `random_state = 42`:

| Features | Candidato | AUC | Accuracy | F1 |
|---|---|---|---|---|
| ALL (12) | Regresión logística | 0,873 | 0,897 | 0,935 |
| ALL (12) | **Random forest** | **0,914** | 0,918 | 0,949 |
| ALL (12) | Gradient boosting | 0,906 | 0,918 | 0,948 |
| NUMERIC (5) | Regresión logística | 0,745 | 0,773 | 0,869 |
| NUMERIC (5) | **Random forest** | **0,882** | 0,814 | 0,889 |
| NUMERIC (5) | Gradient boosting | 0,818 | 0,794 | 0,875 |
| CUSTOM (9) | Regresión logística | 0,853 | 0,897 | 0,935 |
| CUSTOM (9) | **Random forest** | **0,919** | 0,907 | 0,942 |
| CUSTOM (9) | Gradient boosting | 0,895 | 0,887 | 0,928 |

Evolución del champion:

| Ejecución | Mejor AUC | Champion previo | Resultado |
|---|---|---|---|
| 1 · ALL | 0,9139 | ninguno | Versión 1 → `champion` |
| 2 · NUMERIC | 0,8818 | v1 (0,9139) | No supera al champion: alias sin cambios |
| 3 · CUSTOM | 0,9194 | v1 (0,9139) | Versión 2 → `champion` |

Así se cumple el criterio de éxito: una nueva ejecución generó la versión 2, el alias pasó a ella y, al reiniciar la API, `/health` devuelve `"version": "2"` sin cambiar código.

Inferencia: lote de 50 clientes guardado en `data/predictions/prediccion_2026-11-07.csv`, con la versión 2.

Predicción de ejemplo de la API:

```json
POST /predict
{"Age": 22, "Gender": "Female", "Marital Status": "Single", "Occupation": "Student",
 "Monthly Income": "No Income", "Educational Qualifications": "Graduate", "Family size": 4,
 "Customer Type": "Frequent", "latitude": 12.9766, "longitude": 77.5993, "Pin code": 560001,
 "Feedback": "Positive"}

→ {"probabilidad_output": 0.9807, "prediccion": 1, "vuelve_a_pedir": "Yes", "version_modelo": "2"}
```

Un cliente casado, empleado, con ingresos altos y feedback negativo obtuvo 0,192 (`"No"`).

**Lectura:** las variables categóricas aportan mucho (NUMERIC cae a 0,88). Quitar ubicación (`latitude`, `longitude`, `Pin code`) en CUSTOM no empeora e incluso mejora ligeramente el AUC, lo que sugiere que esas variables añaden ruido. Con solo 97 filas de test, diferencias de ~0,005 en AUC están dentro de la variación esperable; por eso se dejó `mejora_minima` configurable.

## 6. Problemas encontrados

- **Puerto 5000 en macOS** ocupado por el Receptor AirPlay: MLflow quedó publicado en 5001 y se usó `MLFLOW_TRACKING_URI=http://localhost:5001` (tiene prioridad sobre el YAML).
- **Modelos de árboles y skops:** MLflow rechaza por defecto el tipo `sklearn.tree._tree.Tree`; se añadió `skops_trusted_types` al registrar.
- **Aviso de esquema con enteros** al inferir la firma del modelo: es informativo; el dataset no tiene nulos.
- **Al cerrar `desplegar.py`** Prefect pausa los despliegues; vuelven a activarse al lanzarlo de nuevo.
- ✏️ Añadir los problemas que haya tenido el grupo.

## 7. Conclusiones

La separación en `src/` (lógica), `flows/` (orquestación), `configs/` (parámetros) y `api/` (servicio) permite cambiar modelos, variables y umbrales sin tocar código, y el alias `champion` desacopla el entrenamiento del consumo. Con más tiempo: validación cruzada en vez de un único split (el test es pequeño), recarga del champion en la API sin reiniciar, ingesta de datos realmente nuevos, monitoreo de drift (actividad siguiente) y un worker/work pool en lugar de `serve` para no depender de una terminal abierta.

## 8. Uso de herramientas de IA ✏️

- Herramientas: Claude Code (asistente de programación) para generar la estructura de flows, API, pruebas y este borrador a partir del enunciado y el ejemplo de referencia.
- ✏️ Qué se hizo sin IA (p. ej. EDA, elección del dataset, revisión de resultados).
- ✏️ Qué salidas de la IA se revisaron, corrigieron o descartaron y por qué.
- ✏️ Contribución individual de cada integrante.
