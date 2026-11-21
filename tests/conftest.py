"""Datos y configuración compartidos por los tests. Nada aquí necesita MLflow, Prefect ni el CSV."""

import numpy as np
import pandas as pd
import pytest

from src.load_data import load_config

CATEGORIAS = {
    "Gender": ["Male", "Female"],
    "Marital Status": ["Single", "Married", "Prefer not to say"],
    "Occupation": ["Student", "Employee", "Self Employeed", "House wife"],
    "Monthly Income": ["No Income", "Below Rs.10000", "10001 to 25000", "25001 to 50000", "More than 50000"],
    "Educational Qualifications": ["Graduate", "Post Graduate", "Ph.D", "School"],
    "Customer Type": ["Frequent", "Occasional", "New"],
    "Feedback": ["Positive", "Negative"],
}


def datos_sinteticos(n: int = 400, semilla: int = 0) -> pd.DataFrame:
    """Dataset con las mismas columnas y rangos que el CSV de Kaggle (valores inventados)."""
    rng = np.random.default_rng(semilla)
    df = pd.DataFrame({
        "Age": rng.integers(18, 34, n),
        "Family size": rng.integers(1, 7, n),
        "latitude": rng.uniform(12.86, 13.10, n).round(4),
        "longitude": rng.uniform(77.48, 77.76, n).round(4),
        "Pin code": rng.integers(560001, 560110, n),
        **{col: rng.choice(valores, n) for col, valores in CATEGORIAS.items()},
        "Output": rng.choice(["Yes", "No"], n, p=[0.77, 0.23]),
    })
    return df


@pytest.fixture
def cfg() -> dict:
    return load_config()


@pytest.fixture
def datos() -> pd.DataFrame:
    return datos_sinteticos()
