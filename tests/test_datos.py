"""Tests de datos: esquema, nulos, rangos, valores permitidos y partición."""

from pathlib import Path

import numpy as np
import pytest

from src.datos import particionar, validar_datos
from src.load_data import ROOT, load_data


def test_datos_validos_no_tienen_problemas(datos, cfg):
    assert validar_datos(datos, cfg) == []


def test_esquema_detecta_columna_faltante(datos, cfg):
    problemas = validar_datos(datos.drop(columns=["Age"]), cfg)
    assert any("Faltan columnas" in p and "Age" in p for p in problemas)


def test_detecta_nulos(datos, cfg):
    datos.loc[[0, 5], "Occupation"] = np.nan
    problemas = validar_datos(datos, cfg)
    assert any("'Occupation' tiene 2 valores nulos" in p for p in problemas)


def test_detecta_valores_fuera_de_rango(datos, cfg):
    datos.loc[0, "Age"] = 150
    datos.loc[1, "Family size"] = 0
    problemas = validar_datos(datos, cfg)
    assert any("'Age'" in p and "fuera de" in p for p in problemas)
    assert any("'Family size'" in p and "fuera de" in p for p in problemas)


def test_detecta_columna_numerica_con_texto(datos, cfg):
    datos["Age"] = datos["Age"].astype(str)
    assert any("'Age' debería ser numérica" in p for p in validar_datos(datos, cfg))


def test_detecta_valores_no_permitidos(datos, cfg):
    datos.loc[0, "Output"] = "Maybe"
    assert any("'Output' tiene valores no permitidos: ['Maybe']" in p for p in validar_datos(datos, cfg))


def test_particion_tamanos_y_estratificacion(datos, cfg):
    train, test = particionar(datos, cfg)
    assert len(train) + len(test) == len(datos)
    assert len(test) == pytest.approx(len(datos) * cfg["division"]["test_size"], abs=1)
    assert set(train.index).isdisjoint(test.index)
    tasa = lambda d: (d["Output"] == "Yes").mean()  # noqa: E731
    assert tasa(train) == pytest.approx(tasa(test), abs=0.02)


def test_particion_es_reproducible(datos, cfg):
    """La inferencia y el drift dependen de reconstruir la misma partición del entrenamiento."""
    train_1, test_1 = particionar(datos, cfg)
    train_2, test_2 = particionar(datos.copy(), cfg)
    assert list(train_1.index) == list(train_2.index)
    assert list(test_1.index) == list(test_2.index)


CSV_REAL = ROOT / "data" / "raw" / "online food delivery dataset.csv"


@pytest.mark.skipif(not Path(CSV_REAL).exists(), reason="El CSV de Kaggle no está en data/raw (no se versiona)")
def test_csv_real_pasa_la_validacion(cfg):
    df = load_data(config=cfg)
    assert validar_datos(df, cfg) == []
    assert len(df) == 388
