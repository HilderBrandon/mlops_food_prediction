"""Partición train/test compartida y validación de calidad de los datos.

No usa Prefect ni MLflow, así que se puede probar sin servicios levantados.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

from src.modelo import preparar_objetivo


def particionar(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Divide el dataset completo en (entrenamiento, prueba), estratificado por el objetivo.

    Usa `division.test_size` y `division.random_state`, igual que el entrenamiento. La
    partición solo depende de la semilla y del objetivo, así que la inferencia y los
    pipelines de drift reconstruyen exactamente la misma parte de entrenamiento y de prueba.
    """
    y = preparar_objetivo(df, cfg["data"]["target"], cfg["data"]["clase_positiva"])
    train, test = train_test_split(
        df,
        test_size=cfg["division"]["test_size"],
        random_state=cfg["division"]["random_state"],
        stratify=y,
    )
    return train, test


def validar_datos(df: pd.DataFrame, cfg: dict) -> list[str]:
    """Revisa esquema, nulos, rangos y valores permitidos. Devuelve la lista de problemas.

    Una lista vacía significa que los datos pasan todas las reglas de `validacion`.
    """
    problemas = []
    esperadas = cfg["data"].get("expected_columns", [])
    reglas = cfg.get("validacion", {})

    # Esquema: columnas esperadas presentes.
    faltantes = [c for c in esperadas if c not in df.columns]
    if faltantes:
        problemas.append(f"Faltan columnas: {faltantes}")

    # Nulos en las columnas esperadas.
    presentes = [c for c in esperadas if c in df.columns]
    nulos = df[presentes].isna().sum()
    for col, n in nulos[nulos > 0].items():
        problemas.append(f"'{col}' tiene {n} valores nulos")

    # Rangos (y tipo numérico) de las columnas numéricas.
    for col, (minimo, maximo) in reglas.get("rangos", {}).items():
        if col not in df.columns:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            problemas.append(f"'{col}' debería ser numérica y es {df[col].dtype}")
            continue
        fuera = df[col].dropna()
        fuera = fuera[(fuera < minimo) | (fuera > maximo)]
        if len(fuera):
            problemas.append(f"'{col}' tiene {len(fuera)} valores fuera de [{minimo}, {maximo}]")

    # Valores permitidos de las columnas categóricas.
    for col, permitidos in reglas.get("valores_permitidos", {}).items():
        if col not in df.columns:
            continue
        raros = sorted(set(df[col].dropna().astype(str)) - {str(v) for v in permitidos})
        if raros:
            problemas.append(f"'{col}' tiene valores no permitidos: {raros}")

    return problemas
