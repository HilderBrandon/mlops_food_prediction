"""Lógica de drift: emulación (inferencia) y detección (pipelines de drift).

No usa Prefect ni MLflow. Evidently solo se importa dentro de `calcular_drift_features`,
así que el resto de funciones se prueba sin tenerlo instalado.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

# Métodos de Evidently cuyo resultado es un p-valor: hay drift si el valor es MENOR que el umbral.
# Los demás (wasserstein, psi, jensenshannon...) son distancias: hay drift si el valor es MAYOR o igual.
METODOS_P_VALOR = {
    "ks", "chisquare", "z", "fisher_exact", "g_test", "anderson", "cramer_von_mises",
    "mannw", "epps_singleton", "t_test", "empirical_mmd",
}


# --- Emulación ---------------------------------------------------------------------------

def emular_drift(
    lote: pd.DataFrame,
    nivel: float,
    cfg_drift: dict,
    target: str,
    clase_positiva: str,
    clase_negativa: str,
    semilla: int | None = None,
) -> pd.DataFrame:
    """Devuelve una copia del lote con drift emulado. Con `nivel = 0.0` el lote no cambia.

    - Numéricas (`cfg_drift["numericas"]`, {columna: factor}): la media sube
      `nivel × factor` desviaciones estándar. Las columnas enteras se redondean.
    - Categóricas (`cfg_drift["categoricas"]`, {columna: valor}): una fracción `nivel`
      de las filas pasa a ese valor.
    - Objetivo: una fracción `nivel × objetivo_invertido` de las filas positivas pasa a
      negativa, así que baja la tasa de positivos.
    """
    if not 0.0 <= nivel <= 1.0:
        raise ValueError(f"drift.nivel debe estar entre 0.0 y 1.0, recibido {nivel}")
    salida = lote.copy()
    if nivel == 0.0 or len(salida) == 0:
        return salida

    rng = np.random.default_rng(semilla)
    n = len(salida)

    for col, factor in (cfg_drift.get("numericas") or {}).items():
        desplazamiento = nivel * float(factor) * float(salida[col].std(ddof=0))
        nuevos = salida[col] + desplazamiento
        if pd.api.types.is_integer_dtype(salida[col]):
            nuevos = nuevos.round().astype(salida[col].dtype)
        salida[col] = nuevos

    for col, valor in (cfg_drift.get("categoricas") or {}).items():
        filas = rng.choice(n, size=round(nivel * n), replace=False)
        salida.iloc[filas, salida.columns.get_loc(col)] = valor

    if target in salida.columns:
        positivas = np.flatnonzero((salida[target] == clase_positiva).to_numpy())
        k = round(nivel * float(cfg_drift.get("objetivo_invertido", 0.0)) * len(positivas))
        if k > 0:
            filas = rng.choice(positivas, size=k, replace=False)
            salida.iloc[filas, salida.columns.get_loc(target)] = clase_negativa

    return salida


# --- Drift de features (Evidently) ---------------------------------------------------------

def calcular_drift_features(
    referencia: pd.DataFrame,
    lote: pd.DataFrame,
    numericas: list[str],
    categoricas: list[str],
    cfg_drift: dict,
):
    """Compara cada variable del lote con la referencia usando Evidently.

    Devuelve (snapshot, resumen). El snapshot se guarda como HTML con `snapshot.save_html`.
    """
    from evidently import DataDefinition, Dataset, Report
    from evidently.presets import DataDriftPreset

    columnas = numericas + categoricas
    faltantes = [c for c in columnas if c not in lote.columns or c not in referencia.columns]
    if faltantes:
        raise KeyError(f"Columnas ausentes en la referencia o en el lote: {faltantes}")

    definicion = DataDefinition(numerical_columns=numericas, categorical_columns=categoricas)
    reporte = Report([
        DataDriftPreset(
            columns=columnas,
            drift_share=cfg_drift["umbral_fraccion"],
            num_method=cfg_drift.get("metodo_numericas"),
            cat_method=cfg_drift.get("metodo_categoricas"),
            threshold=cfg_drift["umbral_test"],
        )
    ])
    snapshot = reporte.run(
        current_data=Dataset.from_pandas(lote[columnas].reset_index(drop=True), data_definition=definicion),
        reference_data=Dataset.from_pandas(referencia[columnas].reset_index(drop=True), data_definition=definicion),
    )
    resumen = resumir_evidently(snapshot.dict(), cfg_drift["umbral_test"], cfg_drift["umbral_fraccion"])
    return snapshot, resumen


def es_drift(valor: float, metodo: str, umbral: float) -> bool:
    """Decide si una variable tiene drift según el tipo de método (p-valor o distancia)."""
    metodo = str(metodo or "")
    if metodo in METODOS_P_VALOR or "p_value" in metodo.lower():
        return valor < umbral
    return valor >= umbral


def resumir_evidently(resultado: dict, umbral_test: float, umbral_fraccion: float) -> dict:
    """Extrae del `snapshot.dict()` de Evidently el resultado por variable y la fracción con drift.

    Evidently devuelve una lista de métricas: un `DriftedColumnsCount` (conteo y fracción) y un
    `ValueDrift` por variable, con su método, su umbral y su valor (p-valor o distancia).
    """
    columnas, conteo = {}, None
    for metrica in resultado.get("metrics", []):
        config = metrica.get("config") or {}
        tipo = str(config.get("type", "")).split(":")[-1]
        if tipo == "ValueDrift":
            metodo = config.get("method")
            umbral = config.get("threshold")
            umbral = umbral_test if umbral is None else float(umbral)
            valor = float(metrica["value"])
            columnas[config["column"]] = {
                "metodo": metodo,
                "valor": round(valor, 6),
                "umbral": umbral,
                "drift": es_drift(valor, metodo, umbral),
            }
        elif tipo == "DriftedColumnsCount":
            conteo = metrica.get("value")

    if not columnas:
        raise ValueError("El resultado de Evidently no trae métricas ValueDrift por variable")

    con_drift = sorted(c for c, r in columnas.items() if r["drift"])
    fraccion = len(con_drift) / len(columnas)
    return {
        "columnas": columnas,
        "variables_con_drift": con_drift,
        "n_variables": len(columnas),
        "fraccion_con_drift": round(fraccion, 4),
        "fraccion_evidently": None if conteo is None else conteo.get("share"),
        "umbral_fraccion": umbral_fraccion,
        "alerta": hay_alerta(fraccion, umbral_fraccion),
    }


def hay_alerta(fraccion: float, umbral: float) -> bool:
    """La alerta salta cuando la fracción de variables con drift SUPERA el umbral."""
    return fraccion > umbral


# --- Drift de evento (target) --------------------------------------------------------------

def etiquetas_del_lote(lote: pd.DataFrame, target: str, clase_positiva: str) -> tuple[pd.Series, int]:
    """Devuelve (etiquetas 0/1 disponibles, filas sin etiqueta). Rechaza lotes sin etiquetas.

    En producción la etiqueta real (si el cliente volvió a pedir) llega días después de la
    predicción. Si todavía no ha llegado, el drift de evento no se puede calcular.
    """
    if target not in lote.columns or lote[target].isna().all():
        raise ValueError(
            f"El lote no trae etiquetas reales en la columna '{target}'. El drift de evento "
            "necesita saber si cada cliente volvió a pedir; ejecútalo cuando lleguen las etiquetas."
        )
    disponibles = lote[target].dropna().astype(str).str.strip()
    y = (disponibles == clase_positiva).astype(int)
    return y, int(lote[target].isna().sum())


def drift_objetivo(y_referencia, y_lote, alpha: float) -> dict:
    """Compara la tasa de positivos del lote con la del entrenamiento (chi-cuadrado 2×2)."""
    ref = pd.Series(y_referencia).astype(int)
    lote = pd.Series(y_lote).astype(int)
    if len(lote) == 0:
        raise ValueError("El lote no tiene etiquetas para comparar")
    tabla = np.array([
        [(ref == 1).sum(), (ref == 0).sum()],
        [(lote == 1).sum(), (lote == 0).sum()],
    ])
    if (tabla.sum(axis=0) == 0).any():
        p_valor = 1.0  # todas las filas son de una sola clase en ambos conjuntos: no hay cambio medible
    else:
        p_valor = float(chi2_contingency(tabla)[1])
    return {
        "n_referencia": int(len(ref)),
        "n_lote": int(len(lote)),
        "tasa_referencia": round(float(ref.mean()), 4),
        "tasa_lote": round(float(lote.mean()), 4),
        "diferencia": round(float(lote.mean() - ref.mean()), 4),
        "p_valor": round(p_valor, 6),
        "alpha": alpha,
        "drift": p_valor < alpha,
    }


# --- Utilidades -----------------------------------------------------------------------------

def ultimo_lote(carpeta: Path) -> Path:
    """Lote más reciente de la carpeta (los nombres empiezan con fecha y hora, así que ordenan)."""
    lotes = sorted(Path(carpeta).glob("lote_*.csv"))
    if not lotes:
        raise FileNotFoundError(
            f"No hay lotes en '{carpeta}'. Ejecuta primero la inferencia (flows/inferencia_flow.py)."
        )
    return lotes[-1]


def tabla_markdown_features(resumen: dict) -> str:
    """Tabla Markdown con el resultado por variable (para el artefacto de Prefect)."""
    filas = "\n".join(
        f"| {col} | {r['metodo']} | {r['valor']:.4f} | {'**sí**' if r['drift'] else 'no'} |"
        for col, r in resumen["columnas"].items()
    )
    return "| Variable | Método | Valor | Drift |\n|---|---|---|---|\n" + filas
