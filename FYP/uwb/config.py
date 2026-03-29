"""
Configuration for the UWB Indoor Positioning System.
Edit defaults here or update at runtime via the dashboard API.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class Config:
    # ---- MODE: "1D" or "2D" ----
    mode: str = "2D"

    # For 1D: distance between A1 and A2 (meters)
    line_length: float = 6.5

    # For 2D: room dimensions (meters)
    room_width: float = 6.5
    room_height: float = 7.3

    # Anchor positions (x, y) in meters
    anchor_positions: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "A1": (0.0, 0.0),
        "A2": (6.5, 0.0),
        "A3": (0.0, 7.30),
        "A4": (6.5, 7.30),
    })

    # Smoothing: sliding window for distance averaging
    distance_window_size: int = 10

    # Ignore anchors that have not updated within this timeout (seconds)
    stale_anchor_timeout_s: float = 15.0

    # EMA smoothing factors per algorithm
    smoothing_factors: Dict[str, float] = field(default_factory=lambda: {
        "WCL": 0.30,
        "TRI": 0.20,
        "BCCP": 0.20,
        "FUSED": 0.25,
    })

    # WCL weight exponent
    wcl_exponent: float = 2.0

    # Fusion mode for FUSED output: "fixed" or "adaptive"
    fusion_mode: str = "fixed"

    # Pillar / obstacle model (4 vertices in clockwise or counter-clockwise order)
    pillar_enabled: bool = False
    pillar_vertices: List[Tuple[float, float]] = field(default_factory=lambda: [
        (5.8, 2.8),
        (6.3, 2.8),
        (6.3, 4.1),
        (5.8, 4.1),
    ])
    pillar_near_threshold_m: float = 0.8
    pillar_soft_strength: float = 0.7

    # MQTT
    mqtt_broker: str = "localhost"
    mqtt_port: int = 1883

    # Server
    server_host: str = "0.0.0.0"
    server_port: int = 8000
    update_rate_hz: float = 10.0


# Global singleton — imported by other modules
config = Config()
