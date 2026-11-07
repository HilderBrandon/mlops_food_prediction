"""Actividad 2: registra los dos despliegues en Prefect y los sirve.

- entrenamiento-food/entrenamiento-manual: sin horario, se lanza a mano.
- inferencia-food/inferencia-diaria: todos los días según `inferencia.cron` de config.yaml.

Uso (desde la raíz del repositorio). Deja la terminal abierta: mientras lo esté,
Prefect puede lanzar las ejecuciones programadas.
    poetry run python flows/desplegar.py
"""

import sys
from pathlib import Path

from prefect import serve

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flows.entrenamiento_flow import entrenamiento_flow  # noqa: E402
from flows.inferencia_flow import inferencia_flow  # noqa: E402
from src.load_data import load_config  # noqa: E402

if __name__ == "__main__":
    cron = load_config()["inferencia"]["cron"]
    serve(
        entrenamiento_flow.to_deployment(name="entrenamiento-manual"),
        inferencia_flow.to_deployment(name="inferencia-diaria", cron=cron),
    )
