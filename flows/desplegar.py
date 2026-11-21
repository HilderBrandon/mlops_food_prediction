"""Registra los despliegues en Prefect y los sirve (actividades 2 y 3).

- entrenamiento-food/entrenamiento-manual: sin horario, se lanza a mano.
- inferencia-food/inferencia-diaria: todos los días según `inferencia.cron` de config.yaml.
- drift-features/drift-features-diario y drift-objetivo/drift-objetivo-diario: según
  `drift.cron` (30 minutos después de la inferencia), sobre el último lote guardado.

Uso (desde la raíz del repositorio). Deja la terminal abierta: mientras lo esté,
Prefect puede lanzar las ejecuciones programadas.
    poetry run python flows/desplegar.py
"""

import sys
from pathlib import Path

from prefect import serve

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flows.drift_features_flow import drift_features_flow  # noqa: E402
from flows.drift_objetivo_flow import drift_objetivo_flow  # noqa: E402
from flows.entrenamiento_flow import entrenamiento_flow  # noqa: E402
from flows.inferencia_flow import inferencia_flow  # noqa: E402
from src.load_data import load_config  # noqa: E402

if __name__ == "__main__":
    cfg = load_config()
    serve(
        entrenamiento_flow.to_deployment(name="entrenamiento-manual"),
        inferencia_flow.to_deployment(name="inferencia-diaria", cron=cfg["inferencia"]["cron"]),
        drift_features_flow.to_deployment(name="drift-features-diario", cron=cfg["drift"]["cron"]),
        drift_objetivo_flow.to_deployment(name="drift-objetivo-diario", cron=cfg["drift"]["cron"]),
    )
