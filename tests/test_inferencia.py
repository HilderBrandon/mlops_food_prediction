"""Tests de la inferencia: lote de la parte de prueba, puntuación y nombres de archivo."""

from datetime import datetime

import pytest

from src.datos import particionar
from src.inferencia import marca_tiempo, muestrear_lote, nombre_lote, puntuar
from src.modelo import construir_modelo, preparar_objetivo


def test_lote_sale_solo_de_la_parte_de_prueba(datos, cfg):
    train, test = particionar(datos, cfg)
    lote = muestrear_lote(test, 50, semilla=42)
    assert len(lote) == 50
    assert set(lote.index) <= set(test.index)
    assert set(lote.index).isdisjoint(train.index)


def test_lote_con_semilla_es_reproducible(datos):
    assert list(muestrear_lote(datos, 30, 7).index) == list(muestrear_lote(datos, 30, 7).index)


def test_lote_no_supera_el_tamano_de_prueba(datos):
    assert len(muestrear_lote(datos.head(20), 50, 1)) == 20


def test_tamano_lote_invalido(datos):
    with pytest.raises(ValueError):
        muestrear_lote(datos, 0)


@pytest.fixture
def modelo_entrenado(datos):
    modelo = construir_modelo("random_forest", {"n_estimators": 20, "random_state": 0}, ["Age"], ["Occupation"])
    return modelo.fit(datos[["Age", "Occupation"]], preparar_objetivo(datos, "Output", "Yes"))


def test_puntuar_agrega_probabilidad_y_prediccion(modelo_entrenado, datos):
    resultado = puntuar(modelo_entrenado, datos.head(10), umbral=0.5)
    assert {"probabilidad_output", "prediccion"} <= set(resultado.columns)
    assert resultado["probabilidad_output"].between(0, 1).all()
    assert (resultado["prediccion"] == (resultado["probabilidad_output"] >= 0.5)).all()
    assert "probabilidad_output" not in datos.columns  # no modifica el lote original


def test_puntuar_ignora_columnas_que_el_modelo_no_usa(modelo_entrenado, datos):
    """El lote trae Output y Feedback, pero el modelo solo recibe sus propias columnas."""
    assert len(puntuar(modelo_entrenado, datos.head(5), 0.5)) == 5


def test_puntuar_falla_si_falta_una_columna_del_modelo(modelo_entrenado, datos):
    with pytest.raises(KeyError, match="Occupation"):
        puntuar(modelo_entrenado, datos.drop(columns=["Occupation"]), 0.5)


def test_nombres_de_archivo():
    marca = marca_tiempo(datetime(2026, 11, 21, 6, 30, 0))
    assert marca == "20261121-063000"
    assert nombre_lote(marca, 0.6) == "lote_20261121-063000_nivel-0.6.csv"
    assert nombre_lote(marca, 0.0) == "lote_20261121-063000_nivel-0.0.csv"
