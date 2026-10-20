# MLOps Project: Online Food Ordering Behavior Prediction

## 📋 Descripción del Proyecto
Este proyecto forma parte de las actividades prácticas del curso de **MLOps** de la **Maestría en Ciencia de Datos (Universidad Santo Tomás)**. Su objetivo principal es implementar un flujo de trabajo reproducible (Pipeline) para un modelo de Machine Learning aplicado al análisis y predicción del comportamiento del consumidor utilizando el dataset [Online Food Ordering Dataset](https://www.kaggle.com/datasets/srisyra02/online-food-ordering-dataset) [cite: srisyra02/online-food-ordering-dataset].

---

## 🗂️ Estructura del Repositorio
La estructura del proyecto sigue las mejores prácticas de ingeniería de software y MLOps:

```text
mlops_food_prediction/
│
├── data/
│   ├── raw/                 # Datos originales descargados de Kaggle (online food delivery dataset.csv)
│   └── processed/           # Datos limpios y transformados (cleaned_food_data.csv)
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py     # Script de limpieza, codificación y preparación de datos
│   ├── train.py             # Script de entrenamiento y validación del modelo
│   └── evaluate.py          # Métricas de rendimiento y evaluación
│
├── notebooks/
│   └── 01_eda.ipynb         # Análisis Exploratorio de Datos inicial
│
├── models/                  # Artefactos del modelo versionados localmente (model_v1.pkl)
├── requirements.txt         # Dependencias del proyecto
├── .gitignore               # Archivos y carpetas excluidos del versionamiento de Git
└── README.md                # Documentación principal del proyecto
```

---

## ⚙️ Configuración y Ejecución

### 1. Clonar e inicializar entorno
Asegúrate de tener instalado Python y las dependencias necesarias:
```bash
pip install -r requirements.txt
```

### 2. Preprocesamiento de Datos
Ejecuta el script para transformar las variables categóricas y demográficas:
```bash
python src/preprocessing.py
```

### 3. Entrenamiento del Modelo
Ejecuta el script de entrenamiento para generar el clasificador de Random Forest y guardar el artefacto en `models/`:
```bash
python src/train.py
```

---

## 📊 Resultados del Modelo
* **Algoritmo:** Random Forest Classifier
* **Métricas de Evaluación:** Precisión, Recall y F1-Score con un rendimiento óptimo en la validación del conjunto de prueba.
* **Artefacto Versionado:** `models/model_v1.pkl`
