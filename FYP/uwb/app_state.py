"""
Shared application state for the UWB dashboard server.
"""

import time

from .config import config
from .engine import PositioningEngine
from .experiment import ExperimentManager
from .mqtt_handler import MQTTHandler

engine = PositioningEngine()
experiment = ExperimentManager(engine)
mqtt_handler = MQTTHandler(engine)


def build_stream_payload(results: dict) -> dict:
    return {
        "timestamp": time.time(),
        "mode": config.mode,
        "fusion_mode": config.fusion_mode,
        "positions": {
            name: {
                "position": (
                    {"x": result.position.x, "y": result.position.y}
                    if result.position else None
                ),
                "distances": result.distances,
                "error": result.error,
            }
            for name, result in results.items()
        },
        "raw_distances": engine.raw_distances,
        "rx_powers": engine.rx_powers,
        "statistics": engine.get_statistics(),
        "fusion_ab": engine.get_fusion_ab_metrics(),
        "ground_truth": (
            {"x": engine.ground_truth.x, "y": engine.ground_truth.y}
            if engine.ground_truth else None
        ),
        "experiment": experiment.get_status(),
    }
