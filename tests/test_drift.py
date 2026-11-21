"""Tests de la lógica de drift: emulación, drift de features, drift de evento y alerta.

Ninguno necesita MLflow ni Prefect. Los de Evidently se saltan si no está instalado.
"""

import numpy as np
import pandas as pd
import pytest

from src.datos import particionar
from src.drift import (
    drift_objetivo,
    emular_drift,
    es_drift,
    etiquetas_del_lote,
    hay_alerta,
    resumir_evidently,
    tabla_markdown_features,
    ultimo_lote,
)
from src.inferencia import muestrear_lote
from src.modelo import load_features, preparar_objetivo, resolver_features
from tests.conftest import datos_sinteticos


def _emular(lote, nivel, cfg, semilla=1):
    d = cfg["data"]
    return emular_drift(lote, nivel, cfg["drift"], d["target"], d["clase_positiva"], d["clase_negativa"], semilla)


# --- Emulación ---------------------------------------------------------------------------

def test_nivel_cero_no_cambia_el_lote(datos, cfg):
    pd.testing.assert_frame_equal(_emular(datos, 0.0, cfg), datos)


@pytest.mark.parametrize("nivel", [-0.1, 1.5])
def test_nivel_fuera_de_rango(datos, cfg, nivel):
    with pytest.raises(ValueError, match="entre 0.0 y 1.0"):
        _emular(datos, nivel, cfg)


def test_emulacion_no_modifica_el_original(datos, cfg):
    copia = datos.copy()
    _emular(datos, 0.6, cfg)
    pd.testing.assert_frame_equal(datos, copia)


def test_numericas_se_desplazan_segun_el_nivel(datos, cfg):
    factor = cfg["drift"]["numericas"]["Age"]
    std = datos["Age"].std(ddof=0)
    for nivel in (0.3, 0.6):
        salto = _emular(datos, nivel, cfg)["Age"].mean() - datos["Age"].mean()
        assert salto == pytest.approx(nivel * factor * std, abs=0.5)  # redondeo a enteros
    assert pd.api.types.is_integer_dtype(_emular(datos, 0.6, cfg)["Age"])


def test_categoricas_cambian_una_fraccion_igual_al_nivel(datos, cfg):
    for col, valor in cfg["drift"]["categoricas"].items():
        antes = (datos[col] == valor).mean()
        despues = (_emular(datos, 0.6, cfg)[col] == valor).mean()
        assert despues >= 0.6  # al menos el 60 % de las filas tiene ese valor
        assert despues > antes


def test_tasa_del_objetivo_baja_con_el_nivel(datos, cfg):
    tasa = lambda d: (d["Output"] == "Yes").mean()  # noqa: E731
    t0, t3, t6 = (tasa(_emular(datos, n, cfg)) for n in (0.0, 0.3, 0.6))
    assert t0 > t3 > t6
    # Con 0.6 y objetivo_invertido 0.5 se invierte el 30 % de los "Yes".
    assert t6 == pytest.approx(t0 * (1 - 0.6 * cfg["drift"]["objetivo_invertido"]), abs=0.01)


def test_emulacion_reproducible_con_semilla(datos, cfg):
    pd.testing.assert_frame_equal(_emular(datos, 0.6, cfg, 5), _emular(datos, 0.6, cfg, 5))


# --- Drift de features: lectura del resultado de Evidently --------------------------------

def _resultado_evidently(valores: dict, metodo="ks", umbral=0.05) -> dict:
    """Misma forma que `snapshot.dict()` de Evidently 0.7 (DataDriftPreset)."""
    metricas = [{
        "id": "a1", "metric_name": "DriftedColumnsCount(drift_share=0.3)",
        "config": {"type": "evidently:metric_v2:DriftedColumnsCount", "drift_share": 0.3},
        "value": {"count": float(sum(v < umbral for v in valores.values())),
                  "share": sum(v < umbral for v in valores.values()) / len(valores)},
    }]
    for col, v in valores.items():
        metricas.append({
            "id": f"id-{col}", "metric_name": f"ValueDrift(column={col},method={metodo},threshold={umbral})",
            "config": {"type": "evidently:metric_v2:ValueDrift", "column": col, "method": metodo, "threshold": umbral},
            "value": v,
        })
    return {"metrics": metricas, "tests": []}


def test_resumen_cuenta_variables_con_drift_y_alerta():
    resultado = _resultado_evidently({"Age": 0.001, "Family size": 0.01, "Gender": 0.6, "Occupation": 0.2})
    resumen = resumir_evidently(resultado, umbral_test=0.05, umbral_fraccion=0.3)
    assert resumen["variables_con_drift"] == ["Age", "Family size"]
    assert resumen["fraccion_con_drift"] == 0.5
    assert resumen["fraccion_evidently"] == 0.5
    assert resumen["alerta"] is True


def test_resumen_sin_drift_no_alerta():
    resumen = resumir_evidently(_resultado_evidently({"Age": 0.4, "Gender": 0.9}), 0.05, 0.3)
    assert resumen["variables_con_drift"] == []
    assert resumen["alerta"] is False


def test_resumen_falla_sin_metricas_por_variable():
    with pytest.raises(ValueError, match="ValueDrift"):
        resumir_evidently({"metrics": [], "tests": []}, 0.05, 0.3)


def test_es_drift_distingue_p_valor_de_distancia():
    assert es_drift(0.01, "ks", 0.05) is True          # p-valor bajo: drift
    assert es_drift(0.30, "chisquare", 0.05) is False
    assert es_drift(0.20, "wasserstein", 0.1) is True  # distancia alta: drift
    assert es_drift(0.05, "wasserstein", 0.1) is False


@pytest.mark.parametrize(("fraccion", "esperado"), [(0.0, False), (0.3, False), (0.31, True), (0.5, True)])
def test_alerta_solo_cuando_se_supera_el_umbral(fraccion, esperado):
    assert hay_alerta(fraccion, 0.3) is esperado


def test_tabla_markdown(cfg):
    resumen = resumir_evidently(_resultado_evidently({"Age": 0.001, "Gender": 0.6}), 0.05, 0.3)
    tabla = tabla_markdown_features(resumen)
    assert "| Age | ks | 0.0010 | **sí** |" in tabla
    assert "| Gender | ks | 0.6000 | no |" in tabla


# --- Drift de evento -----------------------------------------------------------------------

def test_rechaza_lote_sin_columna_de_etiquetas(datos):
    with pytest.raises(ValueError, match="no trae etiquetas reales"):
        etiquetas_del_lote(datos.drop(columns=["Output"]), "Output", "Yes")


def test_rechaza_lote_con_etiquetas_vacias(datos):
    datos["Output"] = np.nan
    with pytest.raises(ValueError, match="no trae etiquetas reales"):
        etiquetas_del_lote(datos, "Output", "Yes")


def test_etiquetas_parciales_usa_solo_las_disponibles(datos):
    datos.loc[datos.index[:10], "Output"] = np.nan
    y, sin_etiqueta = etiquetas_del_lote(datos, "Output", "Yes")
    assert sin_etiqueta == 10
    assert len(y) == len(datos) - 10
    assert set(y.unique()) <= {0, 1}


def test_drift_objetivo_sin_cambio():
    resultado = drift_objetivo([1] * 230 + [0] * 70, [1] * 38 + [0] * 12, alpha=0.05)
    assert resultado["tasa_referencia"] == pytest.approx(0.7667, abs=1e-3)
    assert resultado["drift"] is False


def test_drift_objetivo_detecta_caida_de_la_tasa():
    resultado = drift_objetivo([1] * 230 + [0] * 70, [1] * 20 + [0] * 30, alpha=0.05)
    assert resultado["tasa_lote"] == 0.4
    assert resultado["diferencia"] < 0
    assert resultado["drift"] is True


def test_drift_objetivo_con_una_sola_clase_no_falla():
    assert drift_objetivo([1] * 10, [1] * 5, alpha=0.05)["drift"] is False


def test_drift_objetivo_con_emulacion(cfg):
    """Con el mismo lote, el nivel 0.6 cambia la tasa del objetivo lo bastante para detectarse."""
    datos = datos_sinteticos(800, semilla=3)
    train, test = particionar(datos, cfg)
    lote = muestrear_lote(test, 150, semilla=42)
    y_ref = preparar_objetivo(train, "Output", "Yes")
    resultados = {}
    for nivel in (0.0, 0.6):
        y, _ = etiquetas_del_lote(_emular(lote, nivel, cfg), "Output", "Yes")
        resultados[nivel] = drift_objetivo(y_ref, y, cfg["drift"]["alpha_objetivo"])
    assert resultados[0.0]["drift"] is False
    assert resultados[0.6]["drift"] is True


# --- Utilidades ----------------------------------------------------------------------------

def test_ultimo_lote_elige_el_mas_reciente(tmp_path):
    for nombre in ("lote_20261120-060000_nivel-0.0.csv", "lote_20261121-060000_nivel-0.6.csv",
                   "lote_20261121-055900_nivel-0.3.csv"):
        (tmp_path / nombre).write_text("Age\n20\n")
    assert ultimo_lote(tmp_path).name == "lote_20261121-060000_nivel-0.6.csv"


def test_ultimo_lote_sin_lotes(tmp_path):
    with pytest.raises(FileNotFoundError, match="Ejecuta primero la inferencia"):
        ultimo_lote(tmp_path)


# --- Integración con Evidently (se salta si no está instalado) -----------------------------

def test_evidently_solo_alerta_con_drift_fuerte(cfg, tmp_path):
    pytest.importorskip("evidently")
    from src.drift import calcular_drift_features

    # Semilla elegida con margen: con 0.0 ningún p-valor baja de 0.10; con 0.6 las cuatro
    # variables alteradas quedan por debajo de 0.001.
    datos = datos_sinteticos(800, semilla=6)
    train, test = particionar(datos, cfg)
    lote = muestrear_lote(test, 100, semilla=42)
    _, numericas, categoricas = resolver_features(load_features())

    resultados = {}
    for nivel in (0.0, 0.6):
        snapshot, resumen = calcular_drift_features(train, _emular(lote, nivel, cfg), numericas, categoricas, cfg["drift"])
        html = tmp_path / f"reporte_{nivel}.html"
        snapshot.save_html(str(html))
        assert html.stat().st_size > 0
        resultados[nivel] = resumen

    assert resultados[0.0]["alerta"] is False
    assert resultados[0.0]["variables_con_drift"] == []
    assert resultados[0.6]["alerta"] is True
    assert resultados[0.6]["fraccion_con_drift"] > resultados[0.0]["fraccion_con_drift"]
    # Las variables que la emulación altera son las que Evidently marca con drift.
    alteradas = set(cfg["drift"]["numericas"]) | set(cfg["drift"]["categoricas"])
    assert set(resultados[0.6]["variables_con_drift"]) == alteradas
