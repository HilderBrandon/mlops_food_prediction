"""Lógica del modelo: selección de variables, pipeline de sklearn y métricas.

No usa Prefect ni MLflow, así que se puede probar sin servicios levantados.
"""

from pathlib import Path

import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
FEATURES_PATH = ROOT / "configs" / "features.yaml"

ESTIMADORES = {
    "logistic_regression": LogisticRegression,
    "random_forest": RandomForestClassifier,
    "gradient_boosting": GradientBoostingClassifier,
}


def load_features(path: Path = FEATURES_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolver_features(cfg_features: dict) -> tuple[str, list[str], list[str]]:
    """Devuelve (alias, numericas, categoricas) según `seleccion` en features.yaml."""
    alias = str(cfg_features["seleccion"]).upper()
    numericas = list(cfg_features["numericas"])
    categoricas = list(cfg_features["categoricas"])

    if alias == "ALL":
        elegidas = numericas + categoricas
    elif alias == "NUMERIC":
        elegidas = numericas
    elif alias == "CUSTOM":
        elegidas = list(cfg_features.get("custom") or [])
        desconocidas = [v for v in elegidas if v not in numericas + categoricas]
        if desconocidas:
            raise ValueError(f"Variables desconocidas en 'custom': {desconocidas}")
    else:
        raise ValueError(f"seleccion '{alias}' no válida: usa ALL, NUMERIC o CUSTOM")

    if not elegidas:
        raise ValueError("La selección de features no tiene ninguna variable")
    return alias, [v for v in numericas if v in elegidas], [v for v in categoricas if v in elegidas]


def preparar_objetivo(df: pd.DataFrame, target: str, clase_positiva: str) -> pd.Series:
    """Convierte la columna objetivo (Yes/No) en 1/0."""
    return (df[target].astype(str).str.strip() == clase_positiva).astype(int)


def construir_modelo(tipo: str, parametros: dict, numericas: list[str], categoricas: list[str]) -> Pipeline:
    """Preprocesado (escalado + one-hot) seguido del estimador elegido en la configuración."""
    if tipo not in ESTIMADORES:
        raise ValueError(f"Modelo '{tipo}' no soportado. Opciones: {list(ESTIMADORES)}")

    transformadores = []
    if numericas:
        transformadores.append(("num", StandardScaler(), numericas))
    if categoricas:
        transformadores.append(("cat", OneHotEncoder(handle_unknown="ignore"), categoricas))
    return Pipeline([
        ("preprocesado", ColumnTransformer(transformadores)),
        ("modelo", ESTIMADORES[tipo](**(parametros or {}))),
    ])


def calcular_metricas(y_true, proba, umbral: float) -> dict:
    pred = (proba >= umbral).astype(int)
    return {
        "auc": float(roc_auc_score(y_true, proba)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
    }


def debe_promover(auc: float, auc_champion: float | None, umbral_auc: float, mejora_minima: float = 0.0) -> tuple[bool, str]:
    """Regla de promoción: supera el mínimo y al champion actual (más la mejora mínima)."""
    if auc < umbral_auc:
        return False, f"AUC {auc:.4f} por debajo del mínimo {umbral_auc}"
    if auc_champion is not None and auc <= auc_champion + mejora_minima:
        return False, f"AUC {auc:.4f} no supera al champion ({auc_champion:.4f}) + {mejora_minima}"
    if auc_champion is None:
        return True, f"AUC {auc:.4f} supera el mínimo {umbral_auc} y no hay champion previo"
    return True, f"AUC {auc:.4f} supera el mínimo {umbral_auc} y al champion ({auc_champion:.4f})"
