"""Tests de la API sin MLflow: `cargar_champion` se reemplaza por un modelo falso."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

import api.main as api

CLIENTE = {
    "Age": 22, "Gender": "Female", "Marital Status": "Single", "Occupation": "Student",
    "Monthly Income": "No Income", "Educational Qualifications": "Graduate", "Family size": 4,
    "Customer Type": "Frequent", "latitude": 12.9766, "longitude": 77.5993, "Pin code": 560001,
}


class ModeloFalso:
    """Imita un pipeline de sklearn: tiene `feature_names_in_` y `predict_proba`."""

    feature_names_in_ = np.array(["Age", "Occupation", "Family size"])

    def __init__(self, proba: float):
        self.proba = proba
        self.recibido = None

    def predict_proba(self, X):
        self.recibido = X
        return np.array([[1 - self.proba, self.proba]] * len(X))


@pytest.fixture
def cliente_http(monkeypatch):
    def _crear(proba: float = 0.8):
        modelo = ModeloFalso(proba)
        monkeypatch.setattr(api, "cargar_champion", lambda: (modelo, "7"))
        return TestClient(api.app), modelo
    return _crear


def test_health(cliente_http):
    client, _ = cliente_http()
    with client:
        respuesta = client.get("/health")
    assert respuesta.status_code == 200
    assert respuesta.json() == {"estado": "ok", "modelo": api.NOMBRE_MODELO, "version": "7"}


def test_predict_positivo(cliente_http):
    client, modelo = cliente_http(proba=0.8)
    with client:
        respuesta = client.post("/predict", json=CLIENTE)
    assert respuesta.status_code == 200
    assert respuesta.json() == {"probabilidad_output": 0.8, "prediccion": 1, "vuelve_a_pedir": "Yes", "version_modelo": "7"}
    # El modelo solo recibe sus columnas, en su orden.
    assert list(modelo.recibido.columns) == ["Age", "Occupation", "Family size"]


def test_predict_negativo(cliente_http):
    client, _ = cliente_http(proba=0.2)
    with client:
        cuerpo = client.post("/predict", json=CLIENTE).json()
    assert cuerpo["prediccion"] == 0
    assert cuerpo["vuelve_a_pedir"] == "No"


def test_feedback_es_opcional(cliente_http):
    """Feedback no existe al predecir: la API funciona sin él y lo acepta por compatibilidad."""
    client, _ = cliente_http()
    with client:
        assert client.post("/predict", json=CLIENTE).status_code == 200
        assert client.post("/predict", json={**CLIENTE, "Feedback": "Positive"}).status_code == 200


@pytest.mark.parametrize(("campo", "valor"), [("Age", 5), ("Age", "veinte"), ("Family size", 0), ("latitude", 120)])
def test_predict_rechaza_datos_invalidos(cliente_http, campo, valor):
    client, _ = cliente_http()
    with client:
        assert client.post("/predict", json={**CLIENTE, campo: valor}).status_code == 422


def test_predict_rechaza_campo_faltante(cliente_http):
    client, _ = cliente_http()
    sin_ocupacion = {k: v for k, v in CLIENTE.items() if k != "Occupation"}
    with client:
        assert client.post("/predict", json=sin_ocupacion).status_code == 422


def test_predict_sin_modelo_cargado_devuelve_503():
    api.estado.clear()
    client = TestClient(api.app)  # sin `with`: no se ejecuta el arranque, no hay modelo
    assert client.post("/predict", json=CLIENTE).status_code == 503
