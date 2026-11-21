"""Lógica de la inferencia: tomar un lote de la parte de prueba, puntuarlo y nombrar archivos.

No usa Prefect ni MLflow, así que se puede probar sin servicios levantados.
"""

from datetime import datetime

import pandas as pd


def muestrear_lote(prueba: pd.DataFrame, tamano: int, semilla: int | None = None) -> pd.DataFrame:
    """Toma `tamano` filas de la parte de prueba (sin reemplazo).

    Con una semilla fija el lote es siempre el mismo, lo que permite comparar los niveles de
    drift sobre los mismos clientes. Con `semilla = None` el lote cambia en cada ejecución.
    """
    if tamano <= 0:
        raise ValueError(f"tamano_lote debe ser positivo, recibido {tamano}")
    n = min(tamano, len(prueba))
    return prueba.sample(n=n, random_state=semilla)


def puntuar(modelo, lote: pd.DataFrame, umbral: float) -> pd.DataFrame:
    """Añade `probabilidad_output` y `prediccion` al lote.

    Usa las columnas con las que se entrenó el modelo (`feature_names_in_`), aunque el lote
    traiga otras (por ejemplo, la etiqueta real o Feedback).
    """
    columnas = list(modelo.feature_names_in_)
    faltantes = [c for c in columnas if c not in lote.columns]
    if faltantes:
        raise KeyError(f"Al lote le faltan columnas que usa el modelo: {faltantes}")
    proba = modelo.predict_proba(lote[columnas])[:, 1]
    resultado = lote.copy()
    resultado["probabilidad_output"] = proba
    resultado["prediccion"] = (proba >= umbral).astype(int)
    return resultado


def marca_tiempo(ahora: datetime | None = None) -> str:
    """Marca para nombrar archivos, p. ej. 20261121-063000 (permite varias ejecuciones al día)."""
    return (ahora or datetime.now()).strftime("%Y%m%d-%H%M%S")


def nombre_lote(marca: str, nivel: float) -> str:
    """Nombre del archivo con la copia del lote, p. ej. lote_20261121-063000_nivel-0.6.csv."""
    return f"lote_{marca}_nivel-{nivel:.1f}.csv"
