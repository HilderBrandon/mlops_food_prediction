import numpy as np
import pandas as pd
import pytest

from src.modelo import construir_modelo, debe_promover, load_features, resolver_features

FEATURES = {
    "numericas": ["Age", "Family size"],
    "categoricas": ["Gender", "Feedback"],
    "custom": ["Age", "Gender"],
}


@pytest.mark.parametrize(
    ("seleccion", "esperado"),
    [("ALL", (["Age", "Family size"], ["Gender", "Feedback"])),
     ("NUMERIC", (["Age", "Family size"], [])),
     ("CUSTOM", (["Age"], ["Gender"]))],
)
def test_resolver_features(seleccion, esperado):
    alias, num, cat = resolver_features({**FEATURES, "seleccion": seleccion})
    assert alias == seleccion
    assert (num, cat) == esperado


def test_resolver_features_custom_desconocida():
    with pytest.raises(ValueError):
        resolver_features({**FEATURES, "seleccion": "CUSTOM", "custom": ["NoExiste"]})


def test_features_yaml_valido():
    alias, num, cat = resolver_features(load_features())
    assert alias in {"ALL", "NUMERIC", "CUSTOM"}
    assert num or cat


def test_debe_promover():
    assert debe_promover(0.9, None, 0.8)[0]
    assert not debe_promover(0.7, None, 0.8)[0]
    assert not debe_promover(0.85, 0.86, 0.8)[0]
    assert debe_promover(0.87, 0.86, 0.8)[0]
    assert not debe_promover(0.87, 0.86, 0.8, mejora_minima=0.02)[0]


@pytest.mark.parametrize("tipo", ["logistic_regression", "random_forest", "gradient_boosting", "svm"])
def test_construir_modelo_entrena(tipo):
    rng = np.random.default_rng(0)
    X = pd.DataFrame({
        "Age": rng.integers(18, 35, 40),
        "Gender": rng.choice(["Male", "Female"], 40),
    })
    y = np.tile([0, 1], 20)
    modelo = construir_modelo(tipo, {}, ["Age"], ["Gender"]).fit(X, y)
    assert modelo.predict_proba(X).shape == (40, 2)
    assert list(modelo.feature_names_in_) == ["Age", "Gender"]
