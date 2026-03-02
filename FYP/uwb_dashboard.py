#!/usr/bin/env python3
"""
UWB Indoor Positioning System - Real-Time Dashboard
====================================================
Adapted from the BLE IPS Dashboard for UWB ranging.

KEY DIFFERENCE FROM BLE VERSION:
  - UWB provides DIRECT distance measurements via Time of Flight
  - No RSSI-to-distance conversion / path loss model needed
  - Much higher accuracy (~10cm vs ~1m)
  - Tag publishes distances (not anchors publishing RSSI)

SUPPORTS:
  - 1D mode: 2 anchors on a line (current setup)
  - 2D mode: 3+ anchors (future, just add anchors + switch mode)

ARCHITECTURE:
  UWB Anchors ←ranging→ Tag →WiFi/MQTT→ This Server →SSE→ Web Dashboard

MQTT TOPICS:
  uwb/distances  - Combined: {"A1": 3.45, "A2": 5.12, "ts": 12345}
  uwb/range      - Per-anchor: {"anchor": "A1", "distance": 3.45, ...}

Author: Wayne Wong
Project: EE4002D Indoor Tracking - UWB Module
"""

import asyncio
import json
import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import logging

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import paho.mqtt.client as mqtt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class Config:
    # ---- MODE: "1D" or "2D" ----
    mode: str = "2D"

    # For 1D: distance between A1 and A2 (meters)
    line_length: float = 6.0

    # For 2D: room dimensions (meters)
    room_width: float = 6
    room_height: float = 6

    # Anchor positions (x, y) in meters
    anchor_positions: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "A1": (0.0, 0.0),
        "A2": (6, 0.0),
        "A3": (0.0, 6.0),
        "A4": (6.0, 6.0),
    })

    # Smoothing: sliding window for distance averaging
    distance_window_size: int = 10

    # EMA smoothing for position output
    smoothing_factors: Dict[str, float] = field(default_factory=lambda: {
        "WCL": 0.30,
        "TRI": 0.20,
        "BCCP": 0.20,
        "FUSED": 0.25,
    })

    # WCL weight exponent
    wcl_exponent: float = 2.0

    # MQTT
    mqtt_broker: str = "localhost"
    mqtt_port: int = 1883

    # Server
    server_host: str = "0.0.0.0"
    server_port: int = 8000
    update_rate_hz: float = 10.0


config = Config()


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class Position:
    x: float
    y: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class GroundTruth:
    x: float
    y: float
    set_time: float = field(default_factory=time.time)


@dataclass
class AlgorithmResult:
    name: str
    position: Optional[Position]
    distances: Dict[str, float]
    error: Optional[float] = None


# =============================================================================
# POSITIONING ALGORITHMS
# =============================================================================

class PositioningAlgorithms:
    """All algorithms work for both 1D and 2D"""

    @staticmethod
    def position_1d_linear(distances: Dict[str, float]) -> Optional[Position]:
        """
        1D positioning with 2 anchors.
        A1 at x=x1, A2 at x=x2, distances d1, d2.
        x = (L² + d1² - d2²) / (2L) + x1
        """
        aids = list(config.anchor_positions.keys())
        if len(aids) < 2:
            return None
        a1, a2 = aids[0], aids[1]
        if a1 not in distances or a2 not in distances:
            return None

        d1 = distances[a1]
        d2 = distances[a2]
        x1 = config.anchor_positions[a1][0]
        x2 = config.anchor_positions[a2][0]
        L = abs(x2 - x1)
        if L < 0.01:
            return None

        x = (L**2 + d1**2 - d2**2) / (2 * L)
        # Offset if A1 is not at origin
        x += min(x1, x2)
        return Position(x=x, y=0.0)

    @staticmethod
    def weighted_centroid(distances: Dict[str, float]) -> Optional[Position]:
        """WCL - works for 1D (2 anchors) and 2D (3+ anchors)"""
        if len(distances) < 2:
            return None

        total_w = 0.0
        wx = 0.0
        wy = 0.0

        for aid, d in distances.items():
            if aid not in config.anchor_positions or d <= 0:
                continue
            pos = config.anchor_positions[aid]
            w = 1.0 / (d ** config.wcl_exponent)
            wx += w * pos[0]
            wy += w * pos[1]
            total_w += w

        if total_w == 0:
            return None
        return Position(x=wx / total_w, y=wy / total_w)

    @staticmethod
    def least_squares_trilateration(distances: Dict[str, float]) -> Optional[Position]:
        """Least squares - 1D uses linear formula, 2D uses full LS"""
        anchors = [(aid, config.anchor_positions[aid], d)
                   for aid, d in distances.items()
                   if aid in config.anchor_positions and d > 0]

        if len(anchors) < 2:
            return None

        # 2 anchors → use 1D linear
        if len(anchors) == 2:
            return PositioningAlgorithms.position_1d_linear(distances)

        # 3+ anchors → full 2D least squares (same as your BLE version)
        ref_id, (x_n, y_n), r_n = anchors[-1]
        A = []
        b = []
        for i in range(len(anchors) - 1):
            _, (x_i, y_i), r_i = anchors[i]
            A.append([2 * (x_n - x_i), 2 * (y_n - y_i)])
            b.append(r_i**2 - r_n**2 - x_i**2 - y_i**2 + x_n**2 + y_n**2)

        try:
            ata = [[0, 0], [0, 0]]
            atb = [0, 0]
            for idx, row in enumerate(A):
                ata[0][0] += row[0] * row[0]
                ata[0][1] += row[0] * row[1]
                ata[1][0] += row[1] * row[0]
                ata[1][1] += row[1] * row[1]
                atb[0] += row[0] * b[idx]
                atb[1] += row[1] * b[idx]

            det = ata[0][0] * ata[1][1] - ata[0][1] * ata[1][0]
            if abs(det) < 1e-10:
                return None

            x = (ata[1][1] * atb[0] - ata[0][1] * atb[1]) / det
            y = (-ata[1][0] * atb[0] + ata[0][0] * atb[1]) / det

            # Constrain to room
            x = max(0, min(config.room_width, x))
            y = max(0, min(config.room_height, y))
            return Position(x=x, y=y)
        except Exception:
            return None

    @staticmethod
    def bccp(distances: Dict[str, float]) -> Optional[Position]:
        """BCCP - circle intersections. Needs 3+ for 2D, uses 1D for 2."""
        anchors = [(aid, config.anchor_positions[aid], d)
                   for aid, d in distances.items()
                   if aid in config.anchor_positions and d > 0]

        if len(anchors) < 2:
            return None
        if len(anchors) == 2:
            return PositioningAlgorithms.position_1d_linear(distances)
        if len(anchors) < 3:
            return None

        def circle_intersect(c1, r1, c2, r2):
            dx = c2[0] - c1[0]
            dy = c2[1] - c1[1]
            d = math.sqrt(dx*dx + dy*dy)
            if d > r1 + r2 or d < abs(r1 - r2) or d == 0:
                ratio = r1 / (r1 + r2) if (r1 + r2) > 0 else 0.5
                return [(c1[0] + ratio * dx, c1[1] + ratio * dy)]
            a = (r1**2 - r2**2 + d**2) / (2 * d)
            h = math.sqrt(max(0, r1**2 - a**2))
            px = c1[0] + a * dx / d
            py = c1[1] + a * dy / d
            if h == 0:
                return [(px, py)]
            return [(px + h * dy / d, py - h * dx / d),
                    (px - h * dy / d, py + h * dx / d)]

        points = []
        for i in range(len(anchors)):
            for j in range(i + 1, len(anchors)):
                _, c1, r1 = anchors[i]
                _, c2, r2 = anchors[j]
                pts = circle_intersect(c1, r1, c2, r2)
                if len(pts) == 2:
                    others = [anchors[k][1] for k in range(len(anchors)) if k != i and k != j]
                    if others:
                        cx = sum(c[0] for c in others) / len(others)
                        cy = sum(c[1] for c in others) / len(others)
                        d0 = (pts[0][0]-cx)**2 + (pts[0][1]-cy)**2
                        d1 = (pts[1][0]-cx)**2 + (pts[1][1]-cy)**2
                        points.append(pts[0] if d0 < d1 else pts[1])
                    else:
                        points.append(pts[0])
                elif pts:
                    points.append(pts[0])

        if not points:
            return None
        x = sum(p[0] for p in points) / len(points)
        y = sum(p[1] for p in points) / len(points)
        x = max(0, min(config.room_width, x))
        y = max(0, min(config.room_height, y))
        return Position(x=x, y=y)

    @staticmethod
    def fused(wcl: Optional[Position], tri: Optional[Position],
              bccp: Optional[Position]) -> Optional[Position]:
        """Weighted fusion of available algorithms"""
        positions = []
        weights = []
        if wcl:
            positions.append(wcl); weights.append(0.2)
        if tri:
            positions.append(tri); weights.append(0.6)
        if bccp:
            positions.append(bccp); weights.append(0.2)
        if not positions:
            return None
        tw = sum(weights)
        x = sum(w * p.x for w, p in zip(weights, positions)) / tw
        y = sum(w * p.y for w, p in zip(weights, positions)) / tw
        return Position(x=x, y=y)


# =============================================================================
# POSITIONING ENGINE
# =============================================================================

class PositioningEngine:
    def __init__(self):
        self.distance_history: Dict[str, deque] = {}
        for aid in config.anchor_positions:
            self.distance_history[aid] = deque(maxlen=config.distance_window_size)

        self.smoothed_positions: Dict[str, Optional[Position]] = {
            k: None for k in ["WCL", "TRI", "BCCP", "FUSED"]
        }
        self.ground_truth: Optional[GroundTruth] = None
        self.error_history: Dict[str, List[float]] = {
            k: [] for k in ["WCL", "TRI", "BCCP", "FUSED"]
        }
        self.current_distances: Dict[str, float] = {}
        self.raw_distances: Dict[str, float] = {}
        self.rx_powers: Dict[str, float] = {}
        self.last_update: Dict[str, float] = {}

    def add_distance(self, anchor_id: str, distance: float, rx_power: float = 0.0):
        """Add a UWB distance measurement"""
        if anchor_id not in self.distance_history:
            self.distance_history[anchor_id] = deque(maxlen=config.distance_window_size)
            # Auto-add anchor position if not configured (place at origin)
            if anchor_id not in config.anchor_positions:
                logger.warning(f"Unknown anchor {anchor_id} - add its position in config!")

        self.distance_history[anchor_id].append(distance)
        self.raw_distances[anchor_id] = distance
        self.rx_powers[anchor_id] = rx_power
        self.last_update[anchor_id] = time.time()

    def get_smoothed_distances(self) -> Dict[str, float]:
        """Sliding window average with trimmed mean"""
        distances = {}
        now = time.time()

        for aid, history in self.distance_history.items():
            if not history:
                continue
            if aid in self.last_update and now - self.last_update[aid] > 5.0:
                continue

            values = list(history)
            if len(values) >= 3:
                s = sorted(values)
                trimmed = s[1:-1]
                distances[aid] = sum(trimmed) / len(trimmed)
            else:
                distances[aid] = sum(values) / len(values)

        self.current_distances = distances
        return distances

    def smooth_position(self, name: str, new_pos: Optional[Position]) -> Optional[Position]:
        if new_pos is None:
            return self.smoothed_positions.get(name)
        prev = self.smoothed_positions.get(name)
        alpha = config.smoothing_factors.get(name, 0.3)
        if prev is None:
            self.smoothed_positions[name] = new_pos
            return new_pos
        smoothed = Position(
            x=alpha * new_pos.x + (1 - alpha) * prev.x,
            y=alpha * new_pos.y + (1 - alpha) * prev.y
        )
        self.smoothed_positions[name] = smoothed
        return smoothed

    def calc_error(self, pos: Optional[Position]) -> Optional[float]:
        if pos is None or self.ground_truth is None:
            return None
        return math.sqrt((pos.x - self.ground_truth.x)**2 +
                        (pos.y - self.ground_truth.y)**2)

    def compute_all(self) -> Dict[str, AlgorithmResult]:
        distances = self.get_smoothed_distances()
        results = {}

        for name, algo_fn in [
            ("WCL", PositioningAlgorithms.weighted_centroid),
            ("TRI", PositioningAlgorithms.least_squares_trilateration),
            ("BCCP", PositioningAlgorithms.bccp),
        ]:
            raw = algo_fn(distances)
            smoothed = self.smooth_position(name, raw)
            err = self.calc_error(smoothed)
            if err is not None:
                self.error_history[name].append(err)
            results[name] = AlgorithmResult(name, smoothed, distances.copy(), err)

        fused_raw = PositioningAlgorithms.fused(
            results["WCL"].position,
            results["TRI"].position,
            results["BCCP"].position
        )
        fused_smoothed = self.smooth_position("FUSED", fused_raw)
        fused_err = self.calc_error(fused_smoothed)
        if fused_err is not None:
            self.error_history["FUSED"].append(fused_err)
        results["FUSED"] = AlgorithmResult("FUSED", fused_smoothed, distances.copy(), fused_err)

        return results

    def set_ground_truth(self, x, y, auto_reset=True):
        if auto_reset:
            self.clear_errors()
        self.ground_truth = GroundTruth(x=x, y=y)

    def clear_ground_truth(self):
        self.ground_truth = None

    def clear_errors(self):
        for k in self.error_history:
            self.error_history[k] = []

    def get_statistics(self):
        stats = {}
        for name, errors in self.error_history.items():
            if errors:
                n = len(errors)
                stats[name] = {
                    "count": n,
                    "mean": sum(errors) / n,
                    "rmse": math.sqrt(sum(e**2 for e in errors) / n),
                    "min": min(errors),
                    "max": max(errors),
                }
            else:
                stats[name] = {"count": 0, "mean": None, "rmse": None, "min": None, "max": None}
        return stats

    def update_config(self, new_config: dict):
        if "mode" in new_config:
            config.mode = new_config["mode"]
        if "line_length" in new_config:
            config.line_length = float(new_config["line_length"])
        if "room_width" in new_config:
            config.room_width = float(new_config["room_width"])
        if "room_height" in new_config:
            config.room_height = float(new_config["room_height"])
        if "anchor_positions" in new_config:
            for aid, pos in new_config["anchor_positions"].items():
                config.anchor_positions[aid] = tuple(pos)
            for aid in config.anchor_positions:
                if aid not in self.distance_history:
                    self.distance_history[aid] = deque(maxlen=config.distance_window_size)
        if "smoothing_factors" in new_config:
            config.smoothing_factors.update(new_config["smoothing_factors"])


# =============================================================================
# MQTT HANDLER
# =============================================================================

class MQTTHandler:
    def __init__(self, engine: PositioningEngine):
        self.engine = engine
        self.client = mqtt.Client()
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.connected = False

    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            logger.info(f"MQTT connected to {config.mqtt_broker}:{config.mqtt_port}")
            # Subscribe to both topics
            client.subscribe("uwb/distances")
            client.subscribe("uwb/range")
            self.connected = True
        else:
            logger.error(f"MQTT connection failed: {rc}")

    def _on_message(self, client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode())

            if msg.topic == "uwb/distances":
                # Combined format: {"A1": 3.45, "A2": 5.12, "ts": 12345}
                for key, val in payload.items():
                    if key.startswith("A") and isinstance(val, (int, float)):
                        self.engine.add_distance(key, float(val))

            elif msg.topic == "uwb/range":
                # Per-anchor format: {"anchor": "A1", "distance": 3.45, ...}
                anchor_id = payload.get("anchor")
                distance = payload.get("distance")
                rx_power = payload.get("rx_power", 0.0)
                if anchor_id and distance is not None:
                    self.engine.add_distance(anchor_id, float(distance), float(rx_power))

        except json.JSONDecodeError:
            pass
        except Exception as e:
            logger.error(f"MQTT message error: {e}")

    def connect(self):
        try:
            self.client.connect(config.mqtt_broker, config.mqtt_port, 60)
            self.client.loop_start()
        except Exception as e:
            logger.error(f"MQTT connection error: {e}")

    def disconnect(self):
        self.client.loop_stop()
        self.client.disconnect()


# =============================================================================
# FASTAPI APP
# =============================================================================

app = FastAPI(title="UWB IPS Dashboard")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

engine = PositioningEngine()
mqtt_handler = MQTTHandler(engine)


# =============================================================================
# HTML DASHBOARD
# =============================================================================

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>UWB IPS Dashboard</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Instrument+Sans:wght@400;500;600;700&display=swap');
*{margin:0;padding:0;box-sizing:border-box}
:root{
  --bg:#f5f5f0;--card:#fff;--border:#ddd8d0;
  --text:#1c1917;--text2:#57534e;--muted:#a8a29e;
  --wcl:#2563eb;--tri:#ea580c;--bccp:#7c3aed;--fused:#dc2626;--gt:#16a34a;
  --a1:#0891b2;--a2:#7c3aed;--a3:#db2777;--a4:#ca8a04;
}
body{font-family:'Instrument Sans',sans-serif;background:var(--bg);color:var(--text)}
.container{max-width:1600px;margin:0 auto;padding:16px}
header{display:flex;justify-content:space-between;align-items:center;padding:16px 0;border-bottom:2px solid var(--border);margin-bottom:20px}
.logo{display:flex;align-items:center;gap:12px}
.logo-icon{width:40px;height:40px;background:#1c1917;border-radius:6px;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:13px;color:#fff;font-family:'DM Mono',monospace;letter-spacing:1px}
.logo h1{font-size:22px;font-weight:700;letter-spacing:-.5px}
.logo span{color:var(--text2);font-size:13px}
.badges{display:flex;gap:10px;align-items:center}
.badge{padding:5px 14px;border-radius:4px;font-size:12px;font-weight:600;font-family:'DM Mono',monospace;letter-spacing:.5px}
.badge-mode{background:#1c1917;color:#fff}
.badge-status{background:var(--card);border:1px solid var(--border);display:flex;align-items:center;gap:6px}
.dot{width:7px;height:7px;border-radius:50%;background:var(--gt);animation:pulse 2s infinite}
.dot.off{background:var(--fused);animation:none}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}

.grid{display:grid;grid-template-columns:1fr 340px;gap:20px}
.card{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:16px;margin-bottom:16px}
.card-title{font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:1.5px;color:var(--muted);margin-bottom:14px;font-family:'DM Mono',monospace}

/* Plot */
.plot{position:relative;width:100%;background:#fafaf7;border-radius:6px;overflow:hidden;cursor:crosshair;border:1px solid var(--border)}
.plot svg{width:100%;height:100%}
.grid-line{stroke:var(--border);stroke-width:.5;opacity:.4}
.grid-line.major{stroke-width:1;opacity:.6}
.axis-label{fill:var(--muted);font-size:10px;font-family:'DM Mono',monospace}

.legend{display:flex;flex-wrap:wrap;gap:16px;margin-top:14px}
.legend-item{display:flex;align-items:center;gap:6px;font-size:12px;font-family:'DM Mono',monospace}
.legend-dot{width:10px;height:10px;border-radius:2px}

/* Algo cards */
.algo-card{display:grid;grid-template-columns:auto 1fr auto;gap:10px;align-items:center;padding:12px;border-radius:6px;border:1px solid var(--border);margin-bottom:8px;border-left:3px solid}
.algo-icon{width:34px;height:34px;border-radius:4px;display:flex;align-items:center;justify-content:center;font-weight:600;font-size:11px;font-family:'DM Mono',monospace}
.algo-info h4{font-size:13px;font-weight:600;margin-bottom:1px}
.algo-info .coords{font-family:'DM Mono',monospace;font-size:11px;color:var(--text2)}
.algo-err{text-align:right}
.algo-err .val{font-family:'DM Mono',monospace;font-size:16px;font-weight:600}
.algo-err .lbl{font-size:9px;color:var(--muted);text-transform:uppercase;letter-spacing:.5px}

/* Distance bars */
.dist-bar{padding:10px 12px;border-radius:6px;border:1px solid var(--border);margin-bottom:8px}
.dist-header{display:flex;justify-content:space-between;align-items:center;margin-bottom:6px}
.dist-name{font-size:12px;font-weight:600;font-family:'DM Mono',monospace}
.dist-val{font-family:'DM Mono',monospace;font-size:14px;font-weight:500}
.dist-raw{font-size:10px;color:var(--muted);font-family:'DM Mono',monospace}
.bar-bg{height:4px;background:#e7e5e0;border-radius:2px}
.bar-fill{height:100%;border-radius:2px;transition:width .3s}

/* Ground truth */
.gt-panel{background:#f0fdf4;border:1px solid #bbf7d0}
.gt-inputs{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:10px}
.input-group label{font-size:10px;text-transform:uppercase;letter-spacing:.5px;color:var(--text2);display:block;margin-bottom:3px;font-family:'DM Mono',monospace}
.input-group input{width:100%;background:#fff;border:1px solid var(--border);border-radius:4px;padding:8px 10px;font-family:'DM Mono',monospace;font-size:13px}
.input-group input:focus{outline:none;border-color:var(--gt)}
.gt-btns{display:flex;gap:6px}
.btn{flex:1;padding:8px 14px;border:none;border-radius:4px;font-family:'Instrument Sans',sans-serif;font-size:12px;font-weight:600;cursor:pointer}
.btn-primary{background:#1c1917;color:#fff}
.btn-primary:hover{background:#292524}
.btn-secondary{background:#fff;color:var(--text);border:1px solid var(--border)}

/* Stats */
table{width:100%;border-collapse:collapse;font-size:12px}
th{text-align:left;padding:6px 8px;color:var(--muted);font-weight:500;border-bottom:1px solid var(--border);font-family:'DM Mono',monospace;font-size:10px;text-transform:uppercase;letter-spacing:.5px}
td{padding:6px 8px;font-family:'DM Mono',monospace;border-bottom:1px solid var(--border)}

/* Debug */
.debug{padding:10px;background:#fafaf7;border-radius:4px;font-family:'DM Mono',monospace;font-size:11px;border:1px solid var(--border);margin-top:12px}
.debug-title{font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:1px;margin-bottom:6px}

.hint{position:absolute;bottom:8px;left:50%;transform:translateX(-50%);font-size:10px;color:var(--text2);background:rgba(255,255,255,.9);padding:3px 10px;border-radius:3px;border:1px solid var(--border);font-family:'DM Mono',monospace}
.tooltip{position:absolute;background:#fff;border:1px solid var(--border);border-radius:4px;padding:6px 10px;font-size:11px;font-family:'DM Mono',monospace;pointer-events:none;z-index:100;opacity:0;transition:opacity .15s}
.tooltip.show{opacity:1}

@media(max-width:1100px){.grid{grid-template-columns:1fr}}
</style>
</head>
<body>
<div class="container">
<header>
  <div class="logo">
    <div class="logo-icon">UWB</div>
    <div><h1>UWB Indoor Positioning</h1><span>EE4002D Real-Time Dashboard</span></div>
  </div>
  <div class="badges">
    <div class="badge badge-mode" id="modeBadge">1D</div>
    <div class="badge badge-status"><div class="dot" id="statusDot"></div><span id="statusText">Connecting</span></div>
  </div>
</header>

<div class="grid">
  <div>
    <div class="card">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:14px">
        <span class="card-title">Position Plot</span>
        <span id="updateRate" style="font-family:'DM Mono';font-size:11px;color:var(--muted)">0 Hz</span>
      </div>
      <div class="plot" id="posPlot">
        <svg id="plotSvg" preserveAspectRatio="xMidYMid meet"></svg>
        <div class="tooltip" id="tooltip"></div>
        <div class="hint">click to set ground truth</div>
      </div>
      <div class="legend">
        <div class="legend-item"><div class="legend-dot" style="background:var(--wcl)"></div>WCL</div>
        <div class="legend-item"><div class="legend-dot" style="background:var(--tri)"></div>Trilateration</div>
        <div class="legend-item"><div class="legend-dot" style="background:var(--bccp)"></div>BCCP</div>
        <div class="legend-item"><div class="legend-dot" style="background:var(--fused)"></div>Fused</div>
        <div class="legend-item"><div class="legend-dot" style="background:var(--gt)"></div>Ground Truth</div>
      </div>
    </div>
    <div class="card">
      <span class="card-title">UWB Distances (Time of Flight)</span>
      <div id="distBars"></div>
      <div class="debug">
        <div class="debug-title">Distance Debug</div>
        <div id="debugInfo">Waiting for data...</div>
      </div>
    </div>
  </div>

  <div>
    <div class="card">
      <span class="card-title">Algorithm Positions</span>
      <div id="algoCards"></div>
    </div>
    <div class="card gt-panel">
      <span class="card-title">Ground Truth</span>
      <div class="gt-inputs">
        <div class="input-group"><label>X (m)</label><input type="number" id="gtX" step="0.01"></div>
        <div class="input-group"><label>Y (m)</label><input type="number" id="gtY" step="0.01" value="0"></div>
      </div>
      <div class="gt-btns">
        <button class="btn btn-primary" id="setGtBtn">Set GT</button>
        <button class="btn btn-secondary" id="clearGtBtn">Clear</button>
        <button class="btn btn-secondary" id="resetBtn">Reset Stats</button>
      </div>
    </div>
    <div class="card">
      <span class="card-title">Error Statistics</span>
      <table><thead><tr><th>Algo</th><th>RMSE</th><th>Mean</th><th>N</th></tr></thead>
      <tbody id="statsBody"></tbody></table>
    </div>
    <div class="card">
      <span class="card-title">Configuration</span>
      <div class="gt-inputs" style="margin-bottom:8px">
        <div class="input-group"><label>Mode</label>
          <select id="cfgMode" style="width:100%;padding:8px;border:1px solid var(--border);border-radius:4px;font-family:'DM Mono',monospace;font-size:13px">
            <option value="1D">1D (2 anchors)</option>
            <option value="2D">2D (3+ anchors)</option>
          </select>
        </div>
        <div class="input-group"><label>Room Width (m)</label><input type="number" id="cfgWidth" step="0.1"></div>
      </div>
      <div class="gt-inputs" style="margin-bottom:8px">
        <div class="input-group"><label>Room Height (m)</label><input type="number" id="cfgHeight" step="0.1"></div>
        <div class="input-group"><label>Line Length (1D)</label><input type="number" id="cfgLen" step="0.1"></div>
      </div>
      <div id="anchorConfig" style="margin-bottom:8px"></div>
      <button class="btn btn-primary" id="applyBtn" style="width:100%;margin-top:8px">Apply Config</button>
    </div>
  </div>
</div>
</div>

<script>
const C={WCL:'#2563eb',TRI:'#ea580c',BCCP:'#7c3aed',FUSED:'#dc2626',GT:'#22c55e',A1:'#0891b2',A2:'#7c3aed',A3:'#db2777',A4:'#ca8a04'};
const ALGOS=[{id:'WCL',name:'Weighted Centroid'},{id:'TRI',name:'Trilateration'},{id:'BCCP',name:'Barycentric'},{id:'FUSED',name:'Fused'}];
const PAD=50;

let mode='1D',lineLen=5,roomW=6,roomH=5.8,anchors={},gt=null,curData=null;
let updCount=0,lastUpdTime=Date.now();

function dimW(){return mode==='1D'?lineLen:roomW;}
function dimH(){return mode==='1D'?1.5:roomH;}
function plotW(){return dimW()*100+2*PAD;}
function plotH(){return dimH()*100+2*PAD;}
function sx(){return(plotW()-2*PAD)/dimW();}
function sy(){return(plotH()-2*PAD)/dimH();}

function m2px(x,y){
  if(mode==='1D')return{x:PAD+x*sx(),y:plotH()/2};
  return{x:PAD+x*sx(),y:plotH()-PAD-y*sy()};
}
function px2m(px,py){
  if(mode==='1D')return{x:(px-PAD)/sx(),y:0};
  return{x:(px-PAD)/sx(),y:(plotH()-PAD-py)/sy()};
}

function initPlot(){
  const svg=document.getElementById('plotSvg');svg.innerHTML='';
  const W=plotW(),H=plotH();
  svg.setAttribute('viewBox',`0 0 ${W} ${H}`);
  document.getElementById('posPlot').style.aspectRatio=`${W}/${H}`;

  const defs=document.createElementNS('http://www.w3.org/2000/svg','defs');
  defs.innerHTML=`<filter id="glow"><feGaussianBlur stdDeviation="3" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>`;
  svg.appendChild(defs);

  const g=document.createElementNS('http://www.w3.org/2000/svg','g');

  if(mode==='1D'){
    const y=H/2;
    // axis line
    let l=document.createElementNS('http://www.w3.org/2000/svg','line');
    l.setAttribute('x1',PAD);l.setAttribute('y1',y);l.setAttribute('x2',W-PAD);l.setAttribute('y2',y);
    l.setAttribute('stroke','#c0c0b8');l.setAttribute('stroke-width','2');g.appendChild(l);
    // ticks
    for(let x=0;x<=lineLen;x+=0.5){
      const px=PAD+x*sx();const major=x===Math.round(x);
      let t=document.createElementNS('http://www.w3.org/2000/svg','line');
      t.setAttribute('x1',px);t.setAttribute('y1',y-(major?10:5));
      t.setAttribute('x2',px);t.setAttribute('y2',y+(major?10:5));
      t.setAttribute('stroke','#a8a29e');t.setAttribute('stroke-width',major?'1.5':'0.5');
      g.appendChild(t);
      if(major){
        let lb=document.createElementNS('http://www.w3.org/2000/svg','text');
        lb.setAttribute('x',px);lb.setAttribute('y',y+28);
        lb.setAttribute('class','axis-label');lb.setAttribute('text-anchor','middle');
        lb.textContent=x+'m';g.appendChild(lb);
      }
    }
  } else {
    for(let x=0;x<=dimW();x++){
      const p=m2px(x,0);
      let l=document.createElementNS('http://www.w3.org/2000/svg','line');
      l.setAttribute('x1',p.x);l.setAttribute('y1',PAD);l.setAttribute('x2',p.x);l.setAttribute('y2',H-PAD);
      l.setAttribute('class',x%2===0?'grid-line major':'grid-line');g.appendChild(l);
      if(x%2===0){let t=document.createElementNS('http://www.w3.org/2000/svg','text');
        t.setAttribute('x',p.x);t.setAttribute('y',H-PAD+18);t.setAttribute('class','axis-label');
        t.setAttribute('text-anchor','middle');t.textContent=x+'m';g.appendChild(t);}
    }
    for(let y=0;y<=dimH();y++){
      const p=m2px(0,y);
      let l=document.createElementNS('http://www.w3.org/2000/svg','line');
      l.setAttribute('x1',PAD);l.setAttribute('y1',p.y);l.setAttribute('x2',W-PAD);l.setAttribute('y2',p.y);
      l.setAttribute('class',y%2===0?'grid-line major':'grid-line');g.appendChild(l);
      if(y%2===0){let t=document.createElementNS('http://www.w3.org/2000/svg','text');
        t.setAttribute('x',PAD-8);t.setAttribute('y',p.y+4);t.setAttribute('class','axis-label');
        t.setAttribute('text-anchor','end');t.textContent=y+'m';g.appendChild(t);}
    }
  }
  svg.appendChild(g);
  ['circles','positions','anchorMarkers'].forEach(id=>{
    const el=document.createElementNS('http://www.w3.org/2000/svg','g');
    el.id=id;svg.appendChild(el);
  });
  drawAnchors();
}

function drawAnchors(){
  const g=document.getElementById('anchorMarkers');g.innerHTML='';
  Object.entries(anchors).forEach(([id,pos])=>{
    const p=m2px(pos[0],pos[1]);const c=C[id]||'#888';
    let gl=document.createElementNS('http://www.w3.org/2000/svg','circle');
    gl.setAttribute('cx',p.x);gl.setAttribute('cy',p.y);gl.setAttribute('r','18');
    gl.setAttribute('fill',c);gl.setAttribute('opacity','0.15');g.appendChild(gl);
    let ci=document.createElementNS('http://www.w3.org/2000/svg','circle');
    ci.setAttribute('cx',p.x);ci.setAttribute('cy',p.y);ci.setAttribute('r','11');
    ci.setAttribute('fill',c);g.appendChild(ci);
    let t=document.createElementNS('http://www.w3.org/2000/svg','text');
    t.setAttribute('x',p.x);t.setAttribute('y',p.y+4);t.setAttribute('text-anchor','middle');
    t.setAttribute('fill','#fff');t.setAttribute('font-size','9');t.setAttribute('font-weight','bold');
    t.setAttribute('font-family','DM Mono, monospace');t.textContent=id;g.appendChild(t);
  });
}

function drawCircles(dists){
  const g=document.getElementById('circles');g.innerHTML='';
  if(!dists)return;
  Object.entries(dists).forEach(([id,d])=>{
    if(!anchors[id])return;
    const p=m2px(anchors[id][0],anchors[id][1]);
    const r=d*sx();
    let c=document.createElementNS('http://www.w3.org/2000/svg','circle');
    c.setAttribute('cx',p.x);c.setAttribute('cy',p.y);c.setAttribute('r',r);
    c.setAttribute('fill','none');c.setAttribute('stroke',C[id]||'#888');
    c.setAttribute('stroke-width','1.5');c.setAttribute('stroke-dasharray','6,4');
    c.setAttribute('opacity','0.35');g.appendChild(c);
  });
}

function drawPositions(pos){
  const g=document.getElementById('positions');g.innerHTML='';
  if(gt){
    const p=m2px(gt.x,gt.y);const s=12;
    [[-s,-s,s,s],[s,-s,-s,s]].forEach(([x1,y1,x2,y2])=>{
      let l=document.createElementNS('http://www.w3.org/2000/svg','line');
      l.setAttribute('x1',p.x+x1);l.setAttribute('y1',p.y+y1);
      l.setAttribute('x2',p.x+x2);l.setAttribute('y2',p.y+y2);
      l.setAttribute('stroke',C.GT);l.setAttribute('stroke-width','3');g.appendChild(l);
    });
  }
  if(!pos)return;
  ['WCL','TRI','BCCP','FUSED'].forEach(a=>{
    const r=pos[a];if(!r||!r.position)return;
    const p=m2px(r.position.x,r.position.y);
    let gl=document.createElementNS('http://www.w3.org/2000/svg','circle');
    gl.setAttribute('cx',p.x);gl.setAttribute('cy',p.y);gl.setAttribute('r','12');
    gl.setAttribute('fill',C[a]);gl.setAttribute('opacity','0.25');
    gl.setAttribute('filter','url(#glow)');g.appendChild(gl);
    let ci=document.createElementNS('http://www.w3.org/2000/svg','circle');
    ci.setAttribute('cx',p.x);ci.setAttribute('cy',p.y);
    ci.setAttribute('r',a==='FUSED'?'9':'6');
    ci.setAttribute('fill',C[a]);g.appendChild(ci);
  });
}

function initAlgoCards(){
  const c=document.getElementById('algoCards');c.innerHTML='';
  ALGOS.forEach(a=>{
    c.innerHTML+=`<div class="algo-card" style="border-left-color:${C[a.id]}">
      <div class="algo-icon" style="background:${C[a.id]}15;color:${C[a.id]}">${a.id.slice(0,3)}</div>
      <div class="algo-info"><h4>${a.name}</h4><div class="coords" id="c-${a.id}">X: --.--</div></div>
      <div class="algo-err"><div class="val" id="e-${a.id}" style="color:${C[a.id]}">--</div><div class="lbl">Error(m)</div></div>
    </div>`;
  });
}

function initDistBars(){
  const c=document.getElementById('distBars');c.innerHTML='';
  Object.keys(anchors).forEach(id=>{
    c.innerHTML+=`<div class="dist-bar">
      <div class="dist-header"><span class="dist-name" style="color:${C[id]||'#888'}">${id}</span>
      <div><span class="dist-val" id="dv-${id}">-- m</span>
      <div class="dist-raw" id="dr-${id}">raw: --</div></div></div>
      <div class="bar-bg"><div class="bar-fill" id="db-${id}" style="background:${C[id]||'#888'};width:0%"></div></div>
    </div>`;
  });
}

function updateUI(){
  if(!curData)return;
  const pos=curData.positions;
  const dists=pos&&pos.WCL?pos.WCL.distances:null;

  drawCircles(dists);
  drawPositions(pos);

  if(pos)Object.entries(pos).forEach(([a,r])=>{
    const ce=document.getElementById('c-'+a);
    const ee=document.getElementById('e-'+a);
    if(ce&&r&&r.position){
      ce.textContent=mode==='1D'?`X: ${r.position.x.toFixed(3)}m`:`X:${r.position.x.toFixed(2)} Y:${r.position.y.toFixed(2)}`;
      ee.textContent=r.error!=null?r.error.toFixed(3):'--';
    }
  });

  // Distance bars
  if(dists){
    const maxD=mode==='1D'?lineLen:Math.max(roomW,roomH);
    Object.entries(dists).forEach(([id,d])=>{
      const v=document.getElementById('dv-'+id);
      const b=document.getElementById('db-'+id);
      const raw=document.getElementById('dr-'+id);
      if(v)v.textContent=d.toFixed(3)+' m';
      if(b)b.style.width=Math.min(100,d/maxD*100)+'%';
      if(raw&&curData.raw_distances&&curData.raw_distances[id])
        raw.textContent='raw: '+curData.raw_distances[id].toFixed(3)+'m';
    });
  }

  // Debug
  const dbg=document.getElementById('debugInfo');
  if(dists&&Object.keys(dists).length>=2){
    let html='';
    Object.entries(dists).forEach(([id,d])=>{
      html+=`<span style="color:${C[id]||'#888'}">${id}</span>:${d.toFixed(3)}m &nbsp;`;
    });
    // Check triangle inequality for 1D
    if(mode==='1D'){
      const ids=Object.keys(dists);
      if(ids.length>=2){
        const d1=dists[ids[0]],d2=dists[ids[1]];
        const sum=d1+d2;
        html+=`<br>d1+d2=${sum.toFixed(2)}m, line=${lineLen.toFixed(1)}m `;
        html+=sum>=lineLen?'<span style="color:#16a34a">✓ valid</span>':'<span style="color:#dc2626">✗ too short</span>';
      }
    }
    dbg.innerHTML=html;
  }

  // Stats table
  const tbody=document.getElementById('statsBody');tbody.innerHTML='';
  const st=curData.statistics;
  ALGOS.forEach(a=>{
    const s=st?st[a.id]:null;
    const r=document.createElement('tr');
    r.innerHTML=`<td style="color:${C[a.id]}">${a.id}</td>
      <td>${s&&s.rmse!=null?s.rmse.toFixed(3):'--'}</td>
      <td>${s&&s.mean!=null?s.mean.toFixed(3):'--'}</td>
      <td>${s?s.count:0}</td>`;
    tbody.appendChild(r);
  });
}

// SSE
function connectSSE(){
  const es=new EventSource('/api/stream');
  es.onopen=()=>{document.getElementById('statusDot').classList.remove('off');document.getElementById('statusText').textContent='Connected';};
  es.onmessage=e=>{
    try{curData=JSON.parse(e.data);updateUI();
      updCount++;const now=Date.now();
      if(now-lastUpdTime>=1000){document.getElementById('updateRate').textContent=updCount+' Hz';updCount=0;lastUpdTime=now;}
    }catch(err){console.error(err);}
  };
  es.onerror=()=>{document.getElementById('statusDot').classList.add('off');document.getElementById('statusText').textContent='Disconnected';
    setTimeout(()=>{es.close();connectSSE();},2000);};
}

// Events
function setupEvents(){
  document.getElementById('posPlot').addEventListener('click',e=>{
    const svg=document.getElementById('plotSvg');
    const rect=svg.getBoundingClientRect();
    const scX=plotW()/rect.width,scY=plotH()/rect.height;
    const m=px2m((e.clientX-rect.left)*scX,(e.clientY-rect.top)*scY);
    const x=Math.max(0,Math.min(dimW(),m.x));
    const y=mode==='1D'?0:Math.max(0,Math.min(dimH(),m.y));
    document.getElementById('gtX').value=x.toFixed(2);
    document.getElementById('gtY').value=y.toFixed(2);
    setGT(x,y);
  });
  // Tooltip
  const svg=document.getElementById('plotSvg');
  const tt=document.getElementById('tooltip');
  const plot=document.getElementById('posPlot');
  svg.addEventListener('mousemove',e=>{
    const rect=svg.getBoundingClientRect();
    const m=px2m((e.clientX-rect.left)*(plotW()/rect.width),(e.clientY-rect.top)*(plotH()/rect.height));
    if(m.x>=0&&m.x<=dimW()){
      tt.textContent=mode==='1D'?`X: ${m.x.toFixed(2)}m`:`X:${m.x.toFixed(2)} Y:${m.y.toFixed(2)}`;
      tt.style.left=(e.clientX-plot.getBoundingClientRect().left+12)+'px';
      tt.style.top=(e.clientY-plot.getBoundingClientRect().top+12)+'px';
      tt.classList.add('show');
    }else tt.classList.remove('show');
  });
  svg.addEventListener('mouseleave',()=>tt.classList.remove('show'));

  document.getElementById('setGtBtn').addEventListener('click',()=>{
    const x=parseFloat(document.getElementById('gtX').value);
    const y=parseFloat(document.getElementById('gtY').value||0);
    if(!isNaN(x))setGT(x,y);
  });
  document.getElementById('clearGtBtn').addEventListener('click',()=>{gt=null;fetch('/api/ground-truth',{method:'DELETE'});updateUI();});
  document.getElementById('resetBtn').addEventListener('click',()=>fetch('/api/reset-stats',{method:'POST'}));
  document.getElementById('applyBtn').addEventListener('click',applyConfig);
}

async function setGT(x,y){
  gt={x,y};
  await fetch('/api/ground-truth',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({x,y,auto_reset:true})});
  updateUI();
}

async function applyConfig(){
  const newMode=document.getElementById('cfgMode').value;
  const w=parseFloat(document.getElementById('cfgWidth').value);
  const h=parseFloat(document.getElementById('cfgHeight').value);
  const len=parseFloat(document.getElementById('cfgLen').value);
  const cfg={mode:newMode,room_width:w,room_height:h,line_length:len};

  // Read anchor positions from dynamic inputs
  const ap={};
  document.querySelectorAll('.anchor-pos-row').forEach(row=>{
    const id=row.dataset.anchor;
    const ax=parseFloat(row.querySelector('.anc-x').value);
    const ay=parseFloat(row.querySelector('.anc-y').value);
    if(id&&!isNaN(ax)&&!isNaN(ay)) ap[id]=[ax,ay];
  });
  if(Object.keys(ap).length>0) cfg.anchor_positions=ap;

  await fetch('/api/config',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(cfg)});
  await loadConfig();
  initPlot();initDistBars();initAlgoCards();
}

async function loadConfig(){
  try{
    const r=await fetch('/api/config');const c=await r.json();
    mode=c.mode||'1D';lineLen=c.line_length||5;roomW=c.room_width||8;roomH=c.room_height||8;
    anchors={};
    if(c.anchor_positions)Object.entries(c.anchor_positions).forEach(([id,pos])=>{anchors[id]=pos;});
    if(c.ground_truth)gt=c.ground_truth;

    // Update config panel inputs
    document.getElementById('modeBadge').textContent=mode;
    document.getElementById('cfgMode').value=mode;
    document.getElementById('cfgWidth').value=roomW;
    document.getElementById('cfgHeight').value=roomH;
    document.getElementById('cfgLen').value=lineLen;

    // Build dynamic anchor position inputs
    const ac=document.getElementById('anchorConfig');ac.innerHTML='';
    Object.entries(anchors).forEach(([id,pos])=>{
      const row=document.createElement('div');
      row.className='anchor-pos-row';row.dataset.anchor=id;
      row.style.cssText='display:grid;grid-template-columns:40px 1fr 1fr;gap:6px;align-items:center;margin-bottom:6px';
      row.innerHTML=`<span style="font-family:'DM Mono',monospace;font-size:12px;font-weight:600;color:${C[id]||'#888'}">${id}</span>
        <input type="number" class="anc-x" step="0.1" value="${pos[0]}" style="width:100%;padding:6px 8px;border:1px solid var(--border);border-radius:4px;font-family:'DM Mono',monospace;font-size:12px" placeholder="X">
        <input type="number" class="anc-y" step="0.1" value="${pos[1]}" style="width:100%;padding:6px 8px;border:1px solid var(--border);border-radius:4px;font-family:'DM Mono',monospace;font-size:12px" placeholder="Y">`;
      ac.appendChild(row);
    });

    if(mode==='1D'){document.getElementById('gtY').parentElement.style.display='none';}
    else{document.getElementById('gtY').parentElement.style.display='';}
  }catch(e){console.error(e);}
}

async function init(){
  await loadConfig();
  initPlot();initDistBars();initAlgoCards();
  setupEvents();connectSSE();
}
document.addEventListener('DOMContentLoaded',init);
</script>
</body></html>"""


# =============================================================================
# API ROUTES
# =============================================================================

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    return DASHBOARD_HTML


@app.get("/api/stream")
async def stream():
    async def gen():
        while True:
            try:
                results = engine.compute_all()
                data = {
                    "timestamp": time.time(),
                    "mode": config.mode,
                    "positions": {
                        n: {"position": {"x": r.position.x, "y": r.position.y} if r.position else None,
                            "distances": r.distances, "error": r.error}
                        for n, r in results.items()
                    },
                    "raw_distances": engine.raw_distances,
                    "rx_powers": engine.rx_powers,
                    "statistics": engine.get_statistics(),
                    "ground_truth": {"x": engine.ground_truth.x, "y": engine.ground_truth.y} if engine.ground_truth else None,
                }
                yield f"data: {json.dumps(data)}\n\n"
            except Exception as e:
                logger.error(f"Stream error: {e}")
            await asyncio.sleep(1.0 / config.update_rate_hz)
    return StreamingResponse(gen(), media_type="text/event-stream",
                            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"})


@app.get("/api/config")
async def get_config():
    return {
        "mode": config.mode,
        "line_length": config.line_length,
        "room_width": config.room_width,
        "room_height": config.room_height,
        "anchor_positions": config.anchor_positions,
        "smoothing_factors": config.smoothing_factors,
        "ground_truth": {"x": engine.ground_truth.x, "y": engine.ground_truth.y} if engine.ground_truth else None
    }


@app.post("/api/config")
async def update_config(request: Request):
    engine.update_config(await request.json())
    return {"status": "ok"}


@app.post("/api/ground-truth")
async def set_gt(request: Request):
    d = await request.json()
    engine.set_ground_truth(float(d["x"]), float(d.get("y", 0)), d.get("auto_reset", True))
    return {"status": "ok"}


@app.delete("/api/ground-truth")
async def clear_gt():
    engine.clear_ground_truth()
    return {"status": "ok"}


@app.post("/api/reset-stats")
async def reset():
    engine.clear_errors()
    return {"status": "ok"}


@app.post("/api/distance")
async def add_distance(request: Request):
    """Manual distance input for testing without MQTT"""
    d = await request.json()
    aid = d.get("anchor") or d.get("anchor_id")
    dist = d.get("distance")
    if aid and dist is not None:
        engine.add_distance(aid, float(dist), float(d.get("rx_power", 0)))
        return {"status": "ok"}
    return {"status": "error"}


# =============================================================================
# STARTUP / SHUTDOWN
# =============================================================================

@app.on_event("startup")
async def startup():
    logger.info(f"Starting UWB IPS Dashboard (mode={config.mode})")
    logger.info(f"Anchors: {config.anchor_positions}")
    mqtt_handler.connect()
    logger.info(f"Dashboard: http://localhost:{config.server_port}")


@app.on_event("shutdown")
async def shutdown():
    mqtt_handler.disconnect()


if __name__ == "__main__":
    uvicorn.run("uwb_dashboard:app", host=config.server_host,
                port=config.server_port, reload=True, log_level="info")