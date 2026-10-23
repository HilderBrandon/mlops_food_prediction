"""Cargue del dataset real (Online Food Ordering, Kaggle).

Uso:
    poetry run python src/load_data.py
"""

import logging
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "config.yaml"

logger = logging.getLogger(__name__)


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_data(path: str | Path | None = None, config: dict | None = None) -> pd.DataFrame:
    """Lee el CSV, quita columnas vacías y valida las columnas esperadas.

    El CSV original termina cada fila con una coma, lo que crea una columna
    sin nombre ("Unnamed: N"). Esa columna se elimina aquí.
    """
    config = config or load_config()
    data_cfg = config["data"]
    path = Path(path or ROOT / data_cfg["raw_path"])

    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró el dataset en '{path}'. Descárgalo de "
            "https://www.kaggle.com/datasets/srisyra02/online-food-ordering-dataset "
            "y guárdalo en data/raw/ (ver README)."
        )

    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]

    unnamed = [c for c in df.columns if c.startswith("Unnamed")]
    if unnamed:
        logger.info("Eliminando columnas sin nombre: %s", unnamed)
        df = df.drop(columns=unnamed)

    target = data_cfg["target"]
    if target not in df.columns:
        raise ValueError(f"Falta la columna objetivo '{target}'. Columnas: {list(df.columns)}")

    missing = [c for c in data_cfg.get("expected_columns", []) if c not in df.columns]
    if missing:
        logger.warning("Columnas esperadas que no están en el CSV: %s", missing)

    logger.info("Dataset cargado: %d filas x %d columnas", *df.shape)
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    data = load_data()
    print(f"Filas: {data.shape[0]}, columnas: {data.shape[1]}")
    print(data.head())
