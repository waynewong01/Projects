#!/usr/bin/env python3
"""
Indoor Positioning System - Real-Time Dashboard Server
=======================================================
A comprehensive FastAPI server with integrated web dashboard for BLE-based
indoor positioning. Compares WCL, Trilateration, BCCP, and Fused algorithms
with real-time visualization and ground truth error tracking.

Author: Wayne Wong
Project: EE4002D Indoor Tracking
"""

import asyncio
import json
import math
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import logging

# FastAPI and async components
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# MQTT client
import paho.mqtt.client as mqtt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION - Easily modifiable parameters
# =============================================================================

@dataclass
class Config:
    """System configuration - modify these values as needed"""
    
    # Room dimensions (meters)
    room_width: float = 8.2
    room_height: float = 8.0
    
    # Anchor positions (x, y) in meters
    anchor_positions: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "A1": (0.0, 0.0),
        "A2": (8.2, 8.0),
        "A3": (0.0, 8.0),
        "A4": (8.2, 0.0),
    })
    
    # Calibration values per anchor
    calibration: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "A1": {"rssi1m": -74.69, "n": 1.813},
        "A2": {"rssi1m": -72.86, "n": 2.334},
        "A3": {"rssi1m": -75.86, "n": 2.522},
        "A4": {"rssi1m": -73.55, "n": 2.478},
    })
    
    # Smoothing factors per algorithm (EMA alpha values)
    smoothing_factors: Dict[str, float] = field(default_factory=lambda: {
        "WCL": 0.35,
        "TRI": 0.20,
        "BCCP": 0.20,
        "FUSED": 0.25,
    })
    
    # MQTT configuration
    mqtt_broker: str = "localhost"
    mqtt_port: int = 1883
    mqtt_topic: str = "ips/rssi"
    
    # Server configuration
    server_host: str = "0.0.0.0"
    server_port: int = 8000
    
    # Data processing
    rssi_history_size: int = 50
    update_rate_hz: float = 10.0
    
    # WCL weight exponent
    wcl_exponent: float = 2.0
    
    # Distance constraints
    min_distance: float = 0.3
    max_distance: float = 15.0


# Global configuration instance
config = Config()


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class RSSIData:
    """RSSI measurement from an anchor"""
    anchor_id: str
    rssi: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class Position:
    """2D position estimate"""
    x: float
    y: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class AlgorithmResult:
    """Result from a positioning algorithm"""
    name: str
    position: Optional[Position]
    distances: Dict[str, float]
    error: Optional[float] = None


@dataclass
class GroundTruth:
    """Ground truth position for error calculation"""
    x: float
    y: float
    set_time: float = field(default_factory=time.time)


# =============================================================================
# POSITIONING ALGORITHMS
# =============================================================================

class PositioningAlgorithms:
    """Collection of indoor positioning algorithms"""
    
    @staticmethod
    def rssi_to_distance(rssi: float, rssi1m: float, n: float) -> float:
        """Convert RSSI to distance using log-distance path loss model"""
        if rssi >= rssi1m:
            return config.min_distance
        distance = 10 ** ((rssi1m - rssi) / (10 * n))
        return max(config.min_distance, min(config.max_distance, distance))
    
    @staticmethod
    def weighted_centroid(distances: Dict[str, float]) -> Optional[Position]:
        """
        Weighted Centroid Localization (WCL)
        Position = Σ(w_i * pos_i) / Σ(w_i) where w_i = 1/d_i^g
        """
        if len(distances) < 3:
            return None
        
        total_weight = 0.0
        weighted_x = 0.0
        weighted_y = 0.0
        
        for anchor_id, distance in distances.items():
            if anchor_id not in config.anchor_positions:
                continue
            
            # Weight is inverse distance raised to power g
            weight = 1.0 / (distance ** config.wcl_exponent)
            pos = config.anchor_positions[anchor_id]
            
            weighted_x += weight * pos[0]
            weighted_y += weight * pos[1]
            total_weight += weight
        
        if total_weight == 0:
            return None
        
        x = weighted_x / total_weight
        y = weighted_y / total_weight
        
        # Constrain to room bounds
        x = max(0, min(config.room_width, x))
        y = max(0, min(config.room_height, y))
        
        return Position(x=x, y=y)
    
    @staticmethod
    def least_squares_trilateration(distances: Dict[str, float]) -> Optional[Position]:
        """
        Least Squares Trilateration
        Linearizes the circle equations and solves using least squares
        """
        anchors = [(aid, config.anchor_positions[aid], d) 
                   for aid, d in distances.items() 
                   if aid in config.anchor_positions]
        
        if len(anchors) < 3:
            return None
        
        # Use the last anchor as reference for linearization
        ref_id, (x_n, y_n), r_n = anchors[-1]
        
        # Build A matrix and b vector
        A = []
        b = []
        
        for i in range(len(anchors) - 1):
            aid, (x_i, y_i), r_i = anchors[i]
            
            # 2(x_n - x_i) * x + 2(y_n - y_i) * y = r_i^2 - r_n^2 - x_i^2 - y_i^2 + x_n^2 + y_n^2
            A.append([2 * (x_n - x_i), 2 * (y_n - y_i)])
            b.append(r_i**2 - r_n**2 - x_i**2 - y_i**2 + x_n**2 + y_n**2)
        
        # Convert to matrices for least squares solution
        # x = (A^T A)^-1 A^T b
        try:
            # Manual matrix operations (avoiding numpy dependency)
            n_rows = len(A)
            
            # A^T * A (2x2 matrix)
            ata = [[0, 0], [0, 0]]
            for row in A:
                ata[0][0] += row[0] * row[0]
                ata[0][1] += row[0] * row[1]
                ata[1][0] += row[1] * row[0]
                ata[1][1] += row[1] * row[1]
            
            # A^T * b (2x1 vector)
            atb = [0, 0]
            for i, row in enumerate(A):
                atb[0] += row[0] * b[i]
                atb[1] += row[1] * b[i]
            
            # Inverse of 2x2 matrix
            det = ata[0][0] * ata[1][1] - ata[0][1] * ata[1][0]
            if abs(det) < 1e-10:
                return None
            
            inv_ata = [
                [ata[1][1] / det, -ata[0][1] / det],
                [-ata[1][0] / det, ata[0][0] / det]
            ]
            
            # Solution
            x = inv_ata[0][0] * atb[0] + inv_ata[0][1] * atb[1]
            y = inv_ata[1][0] * atb[0] + inv_ata[1][1] * atb[1]
            
            # Constrain to room bounds
            x = max(0, min(config.room_width, x))
            y = max(0, min(config.room_height, y))
            
            return Position(x=x, y=y)
            
        except Exception as e:
            logger.error(f"Trilateration error: {e}")
            return None
    
    @staticmethod
    def bccp(distances: Dict[str, float]) -> Optional[Position]:
        """
        Barycentric Coordinates using Closed Points (BCCP)
        Finds circle intersections and uses the centroid of closest points
        """
        anchors = [(aid, config.anchor_positions[aid], d) 
                   for aid, d in distances.items() 
                   if aid in config.anchor_positions]
        
        if len(anchors) < 3:
            return None
        
        def circle_intersections(c1, r1, c2, r2) -> List[Tuple[float, float]]:
            """Find intersection points of two circles"""
            x1, y1 = c1
            x2, y2 = c2
            
            d = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            
            # No intersection cases
            if d > r1 + r2 or d < abs(r1 - r2) or d == 0:
                # Return closest points on line between centers
                if d == 0:
                    return [(x1, y1)]
                ratio = r1 / (r1 + r2)
                mid_x = x1 + ratio * (x2 - x1)
                mid_y = y1 + ratio * (y2 - y1)
                return [(mid_x, mid_y)]
            
            a = (r1**2 - r2**2 + d**2) / (2 * d)
            h_sq = r1**2 - a**2
            if h_sq < 0:
                h_sq = 0
            h = math.sqrt(h_sq)
            
            # Point on line between centers
            px = x1 + a * (x2 - x1) / d
            py = y1 + a * (y2 - y1) / d
            
            # Intersection points
            if h == 0:
                return [(px, py)]
            
            dx = h * (y2 - y1) / d
            dy = h * (x2 - x1) / d
            
            return [
                (px + dx, py - dy),
                (px - dx, py + dy)
            ]
        
        # Find closed points for each pair of anchors
        closed_points = []
        
        for i in range(len(anchors)):
            for j in range(i + 1, len(anchors)):
                _, c1, r1 = anchors[i]
                _, c2, r2 = anchors[j]
                
                intersections = circle_intersections(c1, r1, c2, r2)
                
                if len(intersections) == 2:
                    # Choose the point closest to the centroid of remaining anchors
                    other_centers = [anchors[k][1] for k in range(len(anchors)) 
                                     if k != i and k != j]
                    if other_centers:
                        centroid = (
                            sum(c[0] for c in other_centers) / len(other_centers),
                            sum(c[1] for c in other_centers) / len(other_centers)
                        )
                        
                        dist1 = math.sqrt((intersections[0][0] - centroid[0])**2 + 
                                         (intersections[0][1] - centroid[1])**2)
                        dist2 = math.sqrt((intersections[1][0] - centroid[0])**2 + 
                                         (intersections[1][1] - centroid[1])**2)
                        
                        closed_points.append(intersections[0] if dist1 < dist2 else intersections[1])
                    else:
                        closed_points.append(intersections[0])
                elif intersections:
                    closed_points.append(intersections[0])
        
        if not closed_points:
            return None
        
        # Compute centroid of closed points
        x = sum(p[0] for p in closed_points) / len(closed_points)
        y = sum(p[1] for p in closed_points) / len(closed_points)
        
        # Constrain to room bounds
        x = max(0, min(config.room_width, x))
        y = max(0, min(config.room_height, y))
        
        return Position(x=x, y=y)
    
    @staticmethod
    def fused(wcl_pos: Optional[Position], 
              tri_pos: Optional[Position], 
              bccp_pos: Optional[Position],
              distances: Dict[str, float]) -> Optional[Position]:
        """
        Fused algorithm combining WCL, Trilateration, and BCCP
        Uses optimized weights based on simulation results
        """
        positions = []
        weights = []
        
        # Weights from simulation (Table 1 in report, σ=0.5)
        if wcl_pos:
            positions.append(wcl_pos)
            weights.append(0.01)  # ωWCL
        
        if tri_pos:
            positions.append(tri_pos)
            weights.append(0.79)  # ωLS
        
        if bccp_pos:
            positions.append(bccp_pos)
            weights.append(0.20)  # ωBCCP
        
        if not positions:
            return None
        
        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        # Weighted average
        x = sum(w * p.x for w, p in zip(weights, positions))
        y = sum(w * p.y for w, p in zip(weights, positions))
        
        return Position(x=x, y=y)


# =============================================================================
# POSITIONING ENGINE
# =============================================================================

class PositioningEngine:
    """Main positioning engine that processes RSSI and computes positions"""
    
    def __init__(self):
        self.rssi_history: Dict[str, deque] = {
            aid: deque(maxlen=config.rssi_history_size)
            for aid in config.anchor_positions.keys()
        }
        
        # Smoothed positions per algorithm
        self.smoothed_positions: Dict[str, Optional[Position]] = {
            "WCL": None,
            "TRI": None,
            "BCCP": None,
            "FUSED": None,
        }
        
        # Ground truth
        self.ground_truth: Optional[GroundTruth] = None
        
        # Error history for statistics
        self.error_history: Dict[str, List[float]] = {
            "WCL": [],
            "TRI": [],
            "BCCP": [],
            "FUSED": [],
        }
        
        # Current distances
        self.current_distances: Dict[str, float] = {}
        
        # Lock for thread safety
        self._lock = asyncio.Lock()
    
    def add_rssi(self, anchor_id: str, rssi: float):
        """Add RSSI measurement"""
        if anchor_id in self.rssi_history:
            self.rssi_history[anchor_id].append(RSSIData(anchor_id, rssi))
    
    def get_smoothed_rssi(self, anchor_id: str) -> Optional[float]:
        """Get smoothed RSSI using simple moving average"""
        if anchor_id not in self.rssi_history:
            return None
        
        history = self.rssi_history[anchor_id]
        if not history:
            return None
        
        # Use last 5 measurements for smoothing
        recent = list(history)[-5:]
        return sum(r.rssi for r in recent) / len(recent)
    
    def compute_distances(self) -> Dict[str, float]:
        """Compute distances from smoothed RSSI values"""
        distances = {}
        
        for anchor_id in config.anchor_positions.keys():
            rssi = self.get_smoothed_rssi(anchor_id)
            if rssi is not None:
                calib = config.calibration.get(anchor_id, {"rssi1m": -70, "n": 2.0})
                distance = PositioningAlgorithms.rssi_to_distance(
                    rssi, calib["rssi1m"], calib["n"]
                )
                distances[anchor_id] = distance
        
        self.current_distances = distances
        return distances
    
    def smooth_position(self, algo_name: str, new_pos: Optional[Position]) -> Optional[Position]:
        """Apply exponential moving average smoothing"""
        if new_pos is None:
            return self.smoothed_positions.get(algo_name)
        
        prev_pos = self.smoothed_positions.get(algo_name)
        alpha = config.smoothing_factors.get(algo_name, 0.3)
        
        if prev_pos is None:
            self.smoothed_positions[algo_name] = new_pos
            return new_pos
        
        smoothed = Position(
            x=alpha * new_pos.x + (1 - alpha) * prev_pos.x,
            y=alpha * new_pos.y + (1 - alpha) * prev_pos.y
        )
        self.smoothed_positions[algo_name] = smoothed
        return smoothed
    
    def calculate_error(self, position: Optional[Position]) -> Optional[float]:
        """Calculate Euclidean error from ground truth"""
        if position is None or self.ground_truth is None:
            return None
        
        return math.sqrt(
            (position.x - self.ground_truth.x)**2 +
            (position.y - self.ground_truth.y)**2
        )
    
    def compute_all_positions(self) -> Dict[str, AlgorithmResult]:
        """Compute positions using all algorithms"""
        distances = self.compute_distances()
        
        results = {}
        
        # WCL
        wcl_raw = PositioningAlgorithms.weighted_centroid(distances)
        wcl_smoothed = self.smooth_position("WCL", wcl_raw)
        wcl_error = self.calculate_error(wcl_smoothed)
        if wcl_error is not None:
            self.error_history["WCL"].append(wcl_error)
        results["WCL"] = AlgorithmResult("WCL", wcl_smoothed, distances.copy(), wcl_error)
        
        # Trilateration
        tri_raw = PositioningAlgorithms.least_squares_trilateration(distances)
        tri_smoothed = self.smooth_position("TRI", tri_raw)
        tri_error = self.calculate_error(tri_smoothed)
        if tri_error is not None:
            self.error_history["TRI"].append(tri_error)
        results["TRI"] = AlgorithmResult("TRI", tri_smoothed, distances.copy(), tri_error)
        
        # BCCP
        bccp_raw = PositioningAlgorithms.bccp(distances)
        bccp_smoothed = self.smooth_position("BCCP", bccp_raw)
        bccp_error = self.calculate_error(bccp_smoothed)
        if bccp_error is not None:
            self.error_history["BCCP"].append(bccp_error)
        results["BCCP"] = AlgorithmResult("BCCP", bccp_smoothed, distances.copy(), bccp_error)
        
        # Fused
        fused_raw = PositioningAlgorithms.fused(wcl_smoothed, tri_smoothed, bccp_smoothed, distances)
        fused_smoothed = self.smooth_position("FUSED", fused_raw)
        fused_error = self.calculate_error(fused_smoothed)
        if fused_error is not None:
            self.error_history["FUSED"].append(fused_error)
        results["FUSED"] = AlgorithmResult("FUSED", fused_smoothed, distances.copy(), fused_error)
        
        return results
    
    def set_ground_truth(self, x: float, y: float, auto_reset: bool = True):
        """Set ground truth position"""
        if auto_reset:
            self.clear_error_history()
        self.ground_truth = GroundTruth(x=x, y=y)
    
    def clear_ground_truth(self):
        """Clear ground truth"""
        self.ground_truth = None
    
    def clear_error_history(self):
        """Clear error history for all algorithms"""
        for key in self.error_history:
            self.error_history[key] = []
    
    def get_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get error statistics for all algorithms"""
        stats = {}
        
        for algo_name, errors in self.error_history.items():
            if errors:
                n = len(errors)
                mean_error = sum(errors) / n
                rmse = math.sqrt(sum(e**2 for e in errors) / n)
                min_error = min(errors)
                max_error = max(errors)
                
                stats[algo_name] = {
                    "count": n,
                    "mean": mean_error,
                    "rmse": rmse,
                    "min": min_error,
                    "max": max_error,
                }
            else:
                stats[algo_name] = {
                    "count": 0,
                    "mean": None,
                    "rmse": None,
                    "min": None,
                    "max": None,
                }
        
        return stats
    
    def update_config(self, new_config: dict):
        """Update configuration dynamically"""
        global config
        
        if "room_width" in new_config:
            config.room_width = float(new_config["room_width"])
        if "room_height" in new_config:
            config.room_height = float(new_config["room_height"])
        if "anchor_positions" in new_config:
            for aid, pos in new_config["anchor_positions"].items():
                config.anchor_positions[aid] = tuple(pos)
        if "calibration" in new_config:
            for aid, calib in new_config["calibration"].items():
                if aid not in config.calibration:
                    config.calibration[aid] = {}
                config.calibration[aid].update(calib)
        if "smoothing_factors" in new_config:
            config.smoothing_factors.update(new_config["smoothing_factors"])


# =============================================================================
# MQTT CLIENT
# =============================================================================

class MQTTHandler:
    """MQTT client for receiving RSSI data from ESP32 anchors"""
    
    def __init__(self, engine: PositioningEngine):
        self.engine = engine
        self.client = mqtt.Client()
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.connected = False
    
    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            logger.info(f"Connected to MQTT broker at {config.mqtt_broker}:{config.mqtt_port}")
            client.subscribe(config.mqtt_topic + "/#")
            self.connected = True
        else:
            logger.error(f"Failed to connect to MQTT broker: {rc}")
    
    def _on_message(self, client, userdata, msg):
        try:
            # Parse message - expected format: {"anchor": "A1", "rssi": -65}
            payload = json.loads(msg.payload.decode())
            
            anchor_id = payload.get("anchor") or payload.get("anchor_id")
            rssi = payload.get("rssi")
            
            if anchor_id and rssi is not None:
                self.engine.add_rssi(anchor_id, float(rssi))
                
        except json.JSONDecodeError:
            # Try simple format: topic ends with anchor ID
            try:
                topic_parts = msg.topic.split("/")
                if len(topic_parts) >= 2:
                    anchor_id = topic_parts[-1]
                    rssi = float(msg.payload.decode())
                    self.engine.add_rssi(anchor_id, rssi)
            except:
                pass
        except Exception as e:
            logger.error(f"Error processing MQTT message: {e}")
    
    def connect(self):
        """Connect to MQTT broker"""
        try:
            self.client.connect(config.mqtt_broker, config.mqtt_port, 60)
            self.client.loop_start()
        except Exception as e:
            logger.error(f"MQTT connection error: {e}")
    
    def disconnect(self):
        """Disconnect from MQTT broker"""
        self.client.loop_stop()
        self.client.disconnect()


# =============================================================================
# FASTAPI APPLICATION
# =============================================================================

app = FastAPI(title="IPS Dashboard", description="Indoor Positioning System Dashboard")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
engine = PositioningEngine()
mqtt_handler = MQTTHandler(engine)


# =============================================================================
# HTML DASHBOARD
# =============================================================================

DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>IPS Dashboard - Indoor Positioning System</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Space+Grotesk:wght@400;500;600;700&display=swap');
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        :root {
            --bg-dark: #0a0a0f;
            --bg-card: #12121a;
            --bg-card-hover: #1a1a25;
            --border-color: #2a2a3a;
            --text-primary: #f0f0f5;
            --text-secondary: #8888a0;
            --text-muted: #555566;
            
            --color-wcl: #3b82f6;
            --color-tri: #f97316;
            --color-bccp: #a855f7;
            --color-fused: #ef4444;
            --color-gt: #22c55e;
            
            --color-a1: #06b6d4;
            --color-a2: #8b5cf6;
            --color-a3: #ec4899;
            --color-a4: #eab308;
            
            --glow-wcl: rgba(59, 130, 246, 0.3);
            --glow-tri: rgba(249, 115, 22, 0.3);
            --glow-bccp: rgba(168, 85, 247, 0.3);
            --glow-fused: rgba(239, 68, 68, 0.3);
        }
        
        body {
            font-family: 'Space Grotesk', sans-serif;
            background: var(--bg-dark);
            color: var(--text-primary);
            min-height: 100vh;
            overflow-x: hidden;
        }
        
        .noise-overlay {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
            background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 400 400' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noiseFilter'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noiseFilter)'/%3E%3C/svg%3E");
            opacity: 0.03;
            z-index: 1000;
        }
        
        .container {
            max-width: 1800px;
            margin: 0 auto;
            padding: 20px;
        }
        
        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 20px 0;
            border-bottom: 1px solid var(--border-color);
            margin-bottom: 24px;
        }
        
        .logo {
            display: flex;
            align-items: center;
            gap: 12px;
        }
        
        .logo-icon {
            width: 40px;
            height: 40px;
            background: linear-gradient(135deg, var(--color-fused), var(--color-wcl));
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
            font-size: 18px;
        }
        
        .logo h1 {
            font-size: 24px;
            font-weight: 600;
            letter-spacing: -0.5px;
        }
        
        .logo span {
            color: var(--text-secondary);
            font-size: 14px;
            font-weight: 400;
        }
        
        .status-indicator {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 8px 16px;
            background: var(--bg-card);
            border-radius: 20px;
            border: 1px solid var(--border-color);
        }
        
        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--color-gt);
            animation: pulse 2s infinite;
        }
        
        .status-dot.disconnected {
            background: var(--color-fused);
            animation: none;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.5; transform: scale(1.2); }
        }
        
        .main-grid {
            display: grid;
            grid-template-columns: 1fr 380px;
            gap: 24px;
        }
        
        .card {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 20px;
            transition: border-color 0.3s ease;
        }
        
        .card:hover {
            border-color: #3a3a4a;
        }
        
        .card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 16px;
        }
        
        .card-title {
            font-size: 14px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: var(--text-secondary);
        }
        
        .position-plot {
            position: relative;
            width: 100%;
            aspect-ratio: 1.025;
            background: linear-gradient(135deg, #0d0d15 0%, #15151f 100%);
            border-radius: 12px;
            overflow: hidden;
            cursor: crosshair;
        }
        
        .plot-grid {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
        }
        
        .grid-line {
            stroke: var(--border-color);
            stroke-width: 0.5;
            opacity: 0.5;
        }
        
        .grid-line.major {
            stroke-width: 1;
            opacity: 0.8;
        }
        
        .axis-label {
            fill: var(--text-muted);
            font-size: 10px;
            font-family: 'JetBrains Mono', monospace;
        }
        
        .anchor-marker {
            transition: all 0.3s ease;
        }
        
        .anchor-marker:hover {
            filter: brightness(1.3);
        }
        
        .position-marker {
            transition: all 0.1s ease;
        }
        
        .distance-circle {
            fill: none;
            stroke-width: 1.5;
            opacity: 0.4;
            transition: all 0.3s ease;
        }
        
        .ground-truth-marker {
            cursor: pointer;
        }
        
        .legend {
            display: flex;
            flex-wrap: wrap;
            gap: 16px;
            margin-top: 16px;
        }
        
        .legend-item {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 13px;
        }
        
        .legend-color {
            width: 12px;
            height: 12px;
            border-radius: 3px;
        }
        
        .legend-color.wcl { background: var(--color-wcl); }
        .legend-color.tri { background: var(--color-tri); }
        .legend-color.bccp { background: var(--color-bccp); }
        .legend-color.fused { background: var(--color-fused); }
        .legend-color.gt { background: var(--color-gt); }
        
        .side-panel {
            display: flex;
            flex-direction: column;
            gap: 20px;
        }
        
        .algo-stats {
            display: grid;
            gap: 12px;
        }
        
        .algo-card {
            display: grid;
            grid-template-columns: auto 1fr auto;
            align-items: center;
            gap: 12px;
            padding: 12px;
            background: var(--bg-dark);
            border-radius: 10px;
            border: 1px solid transparent;
            transition: all 0.3s ease;
        }
        
        .algo-card:hover {
            background: var(--bg-card-hover);
        }
        
        .algo-card.wcl { border-left: 3px solid var(--color-wcl); }
        .algo-card.tri { border-left: 3px solid var(--color-tri); }
        .algo-card.bccp { border-left: 3px solid var(--color-bccp); }
        .algo-card.fused { border-left: 3px solid var(--color-fused); }
        
        .algo-icon {
            width: 36px;
            height: 36px;
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
            font-size: 12px;
        }
        
        .algo-icon.wcl { background: rgba(59, 130, 246, 0.2); color: var(--color-wcl); }
        .algo-icon.tri { background: rgba(249, 115, 22, 0.2); color: var(--color-tri); }
        .algo-icon.bccp { background: rgba(168, 85, 247, 0.2); color: var(--color-bccp); }
        .algo-icon.fused { background: rgba(239, 68, 68, 0.2); color: var(--color-fused); }
        
        .algo-info h4 {
            font-size: 14px;
            font-weight: 600;
            margin-bottom: 2px;
        }
        
        .algo-info .coords {
            font-family: 'JetBrains Mono', monospace;
            font-size: 11px;
            color: var(--text-secondary);
        }
        
        .algo-error {
            text-align: right;
        }
        
        .algo-error .value {
            font-family: 'JetBrains Mono', monospace;
            font-size: 16px;
            font-weight: 600;
        }
        
        .algo-error .label {
            font-size: 10px;
            color: var(--text-muted);
            text-transform: uppercase;
        }
        
        .ground-truth-panel {
            background: linear-gradient(135deg, rgba(34, 197, 94, 0.1), rgba(34, 197, 94, 0.05));
            border: 1px solid rgba(34, 197, 94, 0.3);
        }
        
        .gt-inputs {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 12px;
            margin-bottom: 12px;
        }
        
        .input-group {
            display: flex;
            flex-direction: column;
            gap: 4px;
        }
        
        .input-group label {
            font-size: 11px;
            text-transform: uppercase;
            color: var(--text-secondary);
            letter-spacing: 0.5px;
        }
        
        .input-group input {
            background: var(--bg-dark);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 10px 12px;
            color: var(--text-primary);
            font-family: 'JetBrains Mono', monospace;
            font-size: 14px;
            transition: border-color 0.3s ease;
        }
        
        .input-group input:focus {
            outline: none;
            border-color: var(--color-gt);
        }
        
        .gt-buttons {
            display: flex;
            gap: 8px;
        }
        
        .btn {
            flex: 1;
            padding: 10px 16px;
            border: none;
            border-radius: 8px;
            font-family: 'Space Grotesk', sans-serif;
            font-size: 13px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .btn-primary {
            background: var(--color-gt);
            color: var(--bg-dark);
        }
        
        .btn-primary:hover {
            background: #16a34a;
            transform: translateY(-1px);
        }
        
        .btn-secondary {
            background: var(--bg-dark);
            color: var(--text-primary);
            border: 1px solid var(--border-color);
        }
        
        .btn-secondary:hover {
            background: var(--bg-card-hover);
        }
        
        .stats-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 12px;
        }
        
        .stats-table th {
            text-align: left;
            padding: 8px;
            color: var(--text-muted);
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            border-bottom: 1px solid var(--border-color);
        }
        
        .stats-table td {
            padding: 8px;
            font-family: 'JetBrains Mono', monospace;
            border-bottom: 1px solid rgba(42, 42, 58, 0.5);
        }
        
        .rssi-charts {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 12px;
        }
        
        .rssi-chart {
            background: var(--bg-dark);
            border-radius: 10px;
            padding: 12px;
            min-height: 100px;
        }
        
        .rssi-chart-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
        }
        
        .rssi-chart-title {
            font-size: 12px;
            font-weight: 600;
        }
        
        .rssi-value {
            font-family: 'JetBrains Mono', monospace;
            font-size: 14px;
            font-weight: 600;
        }
        
        .rssi-distance {
            font-size: 10px;
            color: var(--text-secondary);
        }
        
        .rssi-bar-container {
            height: 6px;
            background: var(--bg-card);
            border-radius: 3px;
            overflow: hidden;
            margin-top: 8px;
        }
        
        .rssi-bar {
            height: 100%;
            border-radius: 3px;
            transition: width 0.3s ease;
        }
        
        .config-panel {
            margin-top: 20px;
        }
        
        .config-section {
            margin-bottom: 16px;
        }
        
        .config-section h4 {
            font-size: 12px;
            color: var(--text-secondary);
            margin-bottom: 8px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .config-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 8px 0;
            border-bottom: 1px solid rgba(42, 42, 58, 0.3);
        }
        
        .config-row label {
            font-size: 13px;
        }
        
        .config-row input {
            width: 80px;
            background: var(--bg-dark);
            border: 1px solid var(--border-color);
            border-radius: 6px;
            padding: 6px 10px;
            color: var(--text-primary);
            font-family: 'JetBrains Mono', monospace;
            font-size: 12px;
            text-align: right;
        }
        
        .config-row input:focus {
            outline: none;
            border-color: var(--color-wcl);
        }
        
        .tab-container {
            border-bottom: 1px solid var(--border-color);
            margin-bottom: 16px;
        }
        
        .tabs {
            display: flex;
            gap: 4px;
        }
        
        .tab {
            padding: 8px 16px;
            background: transparent;
            border: none;
            color: var(--text-secondary);
            font-family: 'Space Grotesk', sans-serif;
            font-size: 13px;
            cursor: pointer;
            border-bottom: 2px solid transparent;
            transition: all 0.3s ease;
        }
        
        .tab:hover {
            color: var(--text-primary);
        }
        
        .tab.active {
            color: var(--text-primary);
            border-bottom-color: var(--color-wcl);
        }
        
        .tab-content {
            display: none;
        }
        
        .tab-content.active {
            display: block;
        }
        
        .plot-tooltip {
            position: absolute;
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 8px 12px;
            font-size: 12px;
            pointer-events: none;
            z-index: 100;
            opacity: 0;
            transition: opacity 0.2s ease;
        }
        
        .plot-tooltip.visible {
            opacity: 1;
        }
        
        .click-hint {
            position: absolute;
            bottom: 10px;
            left: 50%;
            transform: translateX(-50%);
            font-size: 11px;
            color: var(--text-muted);
            background: rgba(10, 10, 15, 0.8);
            padding: 4px 12px;
            border-radius: 12px;
        }
        
        @media (max-width: 1200px) {
            .main-grid {
                grid-template-columns: 1fr;
            }
            
            .side-panel {
                display: grid;
                grid-template-columns: 1fr 1fr;
            }
        }
        
        @media (max-width: 768px) {
            .side-panel {
                grid-template-columns: 1fr;
            }
            
            .rssi-charts {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="noise-overlay"></div>
    
    <div class="container">
        <header>
            <div class="logo">
                <div class="logo-icon">IPS</div>
                <div>
                    <h1>Indoor Positioning System</h1>
                    <span>EE4002D Real-Time Dashboard</span>
                </div>
            </div>
            <div class="status-indicator">
                <div class="status-dot" id="statusDot"></div>
                <span id="statusText">Connecting...</span>
            </div>
        </header>
        
        <div class="main-grid">
            <div class="plot-section">
                <div class="card">
                    <div class="card-header">
                        <span class="card-title">Position Plot</span>
                        <span id="updateRate" style="font-size: 12px; color: var(--text-secondary);">0 Hz</span>
                    </div>
                    
                    <div class="position-plot" id="positionPlot">
                        <svg class="plot-grid" id="plotSvg" viewBox="0 0 820 800" preserveAspectRatio="xMidYMid meet">
                            <!-- Grid will be drawn by JS -->
                        </svg>
                        <div class="plot-tooltip" id="plotTooltip"></div>
                        <div class="click-hint">Click to set ground truth</div>
                    </div>
                    
                    <div class="legend">
                        <div class="legend-item">
                            <div class="legend-color wcl"></div>
                            <span>WCL</span>
                        </div>
                        <div class="legend-item">
                            <div class="legend-color tri"></div>
                            <span>Trilateration</span>
                        </div>
                        <div class="legend-item">
                            <div class="legend-color bccp"></div>
                            <span>BCCP</span>
                        </div>
                        <div class="legend-item">
                            <div class="legend-color fused"></div>
                            <span>Fused</span>
                        </div>
                        <div class="legend-item">
                            <div class="legend-color gt"></div>
                            <span>Ground Truth</span>
                        </div>
                    </div>
                </div>
                
                <div class="card" style="margin-top: 20px;">
                    <div class="card-header">
                        <span class="card-title">RSSI & Distance</span>
                    </div>
                    <div class="rssi-charts" id="rssiCharts">
                        <!-- RSSI charts will be generated by JS -->
                    </div>
                </div>
            </div>
            
            <div class="side-panel">
                <div class="card">
                    <div class="card-header">
                        <span class="card-title">Algorithm Positions</span>
                    </div>
                    <div class="algo-stats" id="algoStats">
                        <!-- Algorithm cards will be generated by JS -->
                    </div>
                </div>
                
                <div class="card ground-truth-panel">
                    <div class="card-header">
                        <span class="card-title">Ground Truth</span>
                    </div>
                    <div class="gt-inputs">
                        <div class="input-group">
                            <label>X Position (m)</label>
                            <input type="number" id="gtX" step="0.1" min="0" placeholder="0.00">
                        </div>
                        <div class="input-group">
                            <label>Y Position (m)</label>
                            <input type="number" id="gtY" step="0.1" min="0" placeholder="0.00">
                        </div>
                    </div>
                    <div class="gt-buttons">
                        <button class="btn btn-primary" id="setGtBtn">Set GT</button>
                        <button class="btn btn-secondary" id="clearGtBtn">Clear</button>
                        <button class="btn btn-secondary" id="resetStatsBtn">Reset Stats</button>
                    </div>
                </div>
                
                <div class="card">
                    <div class="tab-container">
                        <div class="tabs">
                            <button class="tab active" data-tab="stats">Statistics</button>
                            <button class="tab" data-tab="config">Config</button>
                        </div>
                    </div>
                    
                    <div class="tab-content active" id="statsTab">
                        <table class="stats-table">
                            <thead>
                                <tr>
                                    <th>Algorithm</th>
                                    <th>RMSE</th>
                                    <th>Mean</th>
                                    <th>Count</th>
                                </tr>
                            </thead>
                            <tbody id="statsTableBody">
                                <!-- Stats will be generated by JS -->
                            </tbody>
                        </table>
                    </div>
                    
                    <div class="tab-content" id="configTab">
                        <div class="config-section">
                            <h4>Room Dimensions</h4>
                            <div class="config-row">
                                <label>Width (m)</label>
                                <input type="number" id="configWidth" step="0.1" value="8.2">
                            </div>
                            <div class="config-row">
                                <label>Height (m)</label>
                                <input type="number" id="configHeight" step="0.1" value="8.0">
                            </div>
                        </div>
                        <div class="config-section">
                            <h4>Smoothing (α)</h4>
                            <div class="config-row">
                                <label>WCL</label>
                                <input type="number" id="smoothWcl" step="0.05" min="0" max="1" value="0.35">
                            </div>
                            <div class="config-row">
                                <label>Trilateration</label>
                                <input type="number" id="smoothTri" step="0.05" min="0" max="1" value="0.20">
                            </div>
                            <div class="config-row">
                                <label>BCCP</label>
                                <input type="number" id="smoothBccp" step="0.05" min="0" max="1" value="0.20">
                            </div>
                            <div class="config-row">
                                <label>Fused</label>
                                <input type="number" id="smoothFused" step="0.05" min="0" max="1" value="0.25">
                            </div>
                        </div>
                        <button class="btn btn-primary" id="applyConfigBtn" style="width: 100%; margin-top: 12px;">Apply Config</button>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        // Configuration
        let roomWidth = 8.2;
        let roomHeight = 8.0;
        const PADDING = 50;
        const PLOT_WIDTH = 820;
        const PLOT_HEIGHT = 800;
        
        const COLORS = {
            WCL: '#3b82f6',
            TRI: '#f97316',
            BCCP: '#a855f7',
            FUSED: '#ef4444',
            GT: '#22c55e',
            A1: '#06b6d4',
            A2: '#8b5cf6',
            A3: '#ec4899',
            A4: '#eab308'
        };
        
        let anchors = {
            A1: {x: 0, y: 0},
            A2: {x: 8.2, y: 8.0},
            A3: {x: 0, y: 8.0},
            A4: {x: 8.2, y: 0}
        };
        
        let groundTruth = null;
        let currentData = null;
        let updateCount = 0;
        let lastUpdateTime = Date.now();
        
        // Coordinate conversion functions
        function metersToPixels(x, y) {
            const scaleX = (PLOT_WIDTH - 2 * PADDING) / roomWidth;
            const scaleY = (PLOT_HEIGHT - 2 * PADDING) / roomHeight;
            return {
                x: PADDING + x * scaleX,
                y: PLOT_HEIGHT - PADDING - y * scaleY  // Flip Y axis
            };
        }
        
        function pixelsToMeters(px, py) {
            const scaleX = (PLOT_WIDTH - 2 * PADDING) / roomWidth;
            const scaleY = (PLOT_HEIGHT - 2 * PADDING) / roomHeight;
            return {
                x: (px - PADDING) / scaleX,
                y: (PLOT_HEIGHT - PADDING - py) / scaleY
            };
        }
        
        // Initialize SVG plot
        function initPlot() {
            const svg = document.getElementById('plotSvg');
            svg.innerHTML = '';
            
            // Create defs for gradients and filters
            const defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');
            
            // Glow filter
            const filter = document.createElementNS('http://www.w3.org/2000/svg', 'filter');
            filter.setAttribute('id', 'glow');
            filter.innerHTML = `
                <feGaussianBlur stdDeviation="3" result="coloredBlur"/>
                <feMerge>
                    <feMergeNode in="coloredBlur"/>
                    <feMergeNode in="SourceGraphic"/>
                </feMerge>
            `;
            defs.appendChild(filter);
            svg.appendChild(defs);
            
            // Draw grid
            const gridGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
            gridGroup.setAttribute('class', 'grid');
            
            // Vertical lines
            for (let x = 0; x <= roomWidth; x += 1) {
                const pos = metersToPixels(x, 0);
                const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
                line.setAttribute('x1', pos.x);
                line.setAttribute('y1', PADDING);
                line.setAttribute('x2', pos.x);
                line.setAttribute('y2', PLOT_HEIGHT - PADDING);
                line.setAttribute('class', x % 2 === 0 ? 'grid-line major' : 'grid-line');
                gridGroup.appendChild(line);
                
                // X axis labels
                if (x % 2 === 0) {
                    const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
                    label.setAttribute('x', pos.x);
                    label.setAttribute('y', PLOT_HEIGHT - PADDING + 20);
                    label.setAttribute('class', 'axis-label');
                    label.setAttribute('text-anchor', 'middle');
                    label.textContent = x + 'm';
                    gridGroup.appendChild(label);
                }
            }
            
            // Horizontal lines
            for (let y = 0; y <= roomHeight; y += 1) {
                const pos = metersToPixels(0, y);
                const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
                line.setAttribute('x1', PADDING);
                line.setAttribute('y1', pos.y);
                line.setAttribute('x2', PLOT_WIDTH - PADDING);
                line.setAttribute('y2', pos.y);
                line.setAttribute('class', y % 2 === 0 ? 'grid-line major' : 'grid-line');
                gridGroup.appendChild(line);
                
                // Y axis labels
                if (y % 2 === 0) {
                    const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
                    label.setAttribute('x', PADDING - 10);
                    label.setAttribute('y', pos.y + 4);
                    label.setAttribute('class', 'axis-label');
                    label.setAttribute('text-anchor', 'end');
                    label.textContent = y + 'm';
                    gridGroup.appendChild(label);
                }
            }
            
            svg.appendChild(gridGroup);
            
            // Create groups for different elements
            const circlesGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
            circlesGroup.setAttribute('id', 'distanceCircles');
            svg.appendChild(circlesGroup);
            
            const positionsGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
            positionsGroup.setAttribute('id', 'positions');
            svg.appendChild(positionsGroup);
            
            const anchorsGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
            anchorsGroup.setAttribute('id', 'anchors');
            svg.appendChild(anchorsGroup);
            
            // Draw anchor markers
            drawAnchors();
        }
        
        function drawAnchors() {
            const group = document.getElementById('anchors');
            group.innerHTML = '';
            
            Object.entries(anchors).forEach(([id, pos]) => {
                const pixel = metersToPixels(pos.x, pos.y);
                const color = COLORS[id];
                
                // Outer glow
                const glow = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
                glow.setAttribute('cx', pixel.x);
                glow.setAttribute('cy', pixel.y);
                glow.setAttribute('r', 20);
                glow.setAttribute('fill', color);
                glow.setAttribute('opacity', '0.2');
                group.appendChild(glow);
                
                // Inner circle
                const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
                circle.setAttribute('cx', pixel.x);
                circle.setAttribute('cy', pixel.y);
                circle.setAttribute('r', 12);
                circle.setAttribute('fill', color);
                circle.setAttribute('class', 'anchor-marker');
                group.appendChild(circle);
                
                // Label
                const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
                label.setAttribute('x', pixel.x);
                label.setAttribute('y', pixel.y + 4);
                label.setAttribute('text-anchor', 'middle');
                label.setAttribute('fill', '#fff');
                label.setAttribute('font-size', '10');
                label.setAttribute('font-weight', 'bold');
                label.textContent = id;
                group.appendChild(label);
            });
        }
        
        function drawDistanceCircles(distances) {
            const group = document.getElementById('distanceCircles');
            group.innerHTML = '';
            
            if (!distances) return;
            
            Object.entries(distances).forEach(([id, distance]) => {
                if (!anchors[id]) return;
                
                const pixel = metersToPixels(anchors[id].x, anchors[id].y);
                const scaleX = (PLOT_WIDTH - 2 * PADDING) / roomWidth;
                const radiusPixels = distance * scaleX;
                
                const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
                circle.setAttribute('cx', pixel.x);
                circle.setAttribute('cy', pixel.y);
                circle.setAttribute('r', radiusPixels);
                circle.setAttribute('class', 'distance-circle');
                circle.setAttribute('stroke', COLORS[id]);
                circle.setAttribute('stroke-dasharray', '5,5');
                group.appendChild(circle);
            });
        }
        
        function drawPositions(positions) {
            const group = document.getElementById('positions');
            group.innerHTML = '';
            
            // Draw ground truth first (so it's behind)
            if (groundTruth) {
                const gtPixel = metersToPixels(groundTruth.x, groundTruth.y);
                
                // Cross marker for ground truth
                const size = 15;
                const cross = document.createElementNS('http://www.w3.org/2000/svg', 'g');
                cross.setAttribute('class', 'ground-truth-marker');
                
                const line1 = document.createElementNS('http://www.w3.org/2000/svg', 'line');
                line1.setAttribute('x1', gtPixel.x - size);
                line1.setAttribute('y1', gtPixel.y - size);
                line1.setAttribute('x2', gtPixel.x + size);
                line1.setAttribute('y2', gtPixel.y + size);
                line1.setAttribute('stroke', COLORS.GT);
                line1.setAttribute('stroke-width', '3');
                cross.appendChild(line1);
                
                const line2 = document.createElementNS('http://www.w3.org/2000/svg', 'line');
                line2.setAttribute('x1', gtPixel.x + size);
                line2.setAttribute('y1', gtPixel.y - size);
                line2.setAttribute('x2', gtPixel.x - size);
                line2.setAttribute('y2', gtPixel.y + size);
                line2.setAttribute('stroke', COLORS.GT);
                line2.setAttribute('stroke-width', '3');
                cross.appendChild(line2);
                
                group.appendChild(cross);
            }
            
            // Draw algorithm positions
            if (!positions) return;
            
            const algoOrder = ['WCL', 'TRI', 'BCCP', 'FUSED'];
            
            algoOrder.forEach(algo => {
                const result = positions[algo];
                if (!result || !result.position) return;
                
                const pixel = metersToPixels(result.position.x, result.position.y);
                const color = COLORS[algo];
                
                // Glow effect
                const glow = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
                glow.setAttribute('cx', pixel.x);
                glow.setAttribute('cy', pixel.y);
                glow.setAttribute('r', 15);
                glow.setAttribute('fill', color);
                glow.setAttribute('opacity', '0.3');
                glow.setAttribute('filter', 'url(#glow)');
                group.appendChild(glow);
                
                // Position marker
                const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
                circle.setAttribute('cx', pixel.x);
                circle.setAttribute('cy', pixel.y);
                circle.setAttribute('r', algo === 'FUSED' ? 10 : 7);
                circle.setAttribute('fill', color);
                circle.setAttribute('class', 'position-marker');
                group.appendChild(circle);
            });
        }
        
        // Initialize RSSI charts
        function initRssiCharts() {
            const container = document.getElementById('rssiCharts');
            container.innerHTML = '';
            
            Object.keys(anchors).forEach(id => {
                const chart = document.createElement('div');
                chart.className = 'rssi-chart';
                chart.id = `rssi-${id}`;
                chart.innerHTML = `
                    <div class="rssi-chart-header">
                        <span class="rssi-chart-title" style="color: ${COLORS[id]}">${id}</span>
                        <div>
                            <span class="rssi-value" id="rssi-value-${id}">-- dBm</span>
                            <div class="rssi-distance" id="rssi-dist-${id}">-- m</div>
                        </div>
                    </div>
                    <div class="rssi-bar-container">
                        <div class="rssi-bar" id="rssi-bar-${id}" style="background: ${COLORS[id]}; width: 0%"></div>
                    </div>
                `;
                container.appendChild(chart);
            });
        }
        
        function updateRssiCharts(distances, rssiValues) {
            Object.keys(anchors).forEach(id => {
                const distance = distances ? distances[id] : null;
                const rssi = rssiValues ? rssiValues[id] : null;
                
                const valueEl = document.getElementById(`rssi-value-${id}`);
                const distEl = document.getElementById(`rssi-dist-${id}`);
                const barEl = document.getElementById(`rssi-bar-${id}`);
                
                if (rssi !== null && rssi !== undefined) {
                    valueEl.textContent = `${rssi.toFixed(0)} dBm`;
                    // RSSI typically ranges from -30 (strong) to -100 (weak)
                    const percentage = Math.max(0, Math.min(100, ((rssi + 100) / 70) * 100));
                    barEl.style.width = `${percentage}%`;
                } else {
                    valueEl.textContent = '-- dBm';
                    barEl.style.width = '0%';
                }
                
                if (distance !== null && distance !== undefined) {
                    distEl.textContent = `${distance.toFixed(2)} m`;
                } else {
                    distEl.textContent = '-- m';
                }
            });
        }
        
        // Initialize algorithm stats cards
        function initAlgoStats() {
            const container = document.getElementById('algoStats');
            container.innerHTML = '';
            
            const algorithms = [
                {id: 'WCL', name: 'Weighted Centroid'},
                {id: 'TRI', name: 'Trilateration'},
                {id: 'BCCP', name: 'Barycentric'},
                {id: 'FUSED', name: 'Fused'}
            ];
            
            algorithms.forEach(algo => {
                const card = document.createElement('div');
                card.className = `algo-card ${algo.id.toLowerCase()}`;
                card.id = `algo-card-${algo.id}`;
                card.innerHTML = `
                    <div class="algo-icon ${algo.id.toLowerCase()}">${algo.id.substring(0, 2)}</div>
                    <div class="algo-info">
                        <h4>${algo.name}</h4>
                        <div class="coords" id="coords-${algo.id}">X: --.-- | Y: --.--</div>
                    </div>
                    <div class="algo-error">
                        <div class="value" id="error-${algo.id}" style="color: ${COLORS[algo.id]}">--</div>
                        <div class="label">Error (m)</div>
                    </div>
                `;
                container.appendChild(card);
            });
        }
        
        function updateAlgoStats(positions) {
            if (!positions) return;
            
            Object.entries(positions).forEach(([algo, result]) => {
                const coordsEl = document.getElementById(`coords-${algo}`);
                const errorEl = document.getElementById(`error-${algo}`);
                
                if (result && result.position) {
                    coordsEl.textContent = `X: ${result.position.x.toFixed(2)} | Y: ${result.position.y.toFixed(2)}`;
                    
                    if (result.error !== null && result.error !== undefined) {
                        errorEl.textContent = result.error.toFixed(3);
                    } else {
                        errorEl.textContent = '--';
                    }
                } else {
                    coordsEl.textContent = 'X: --.-- | Y: --.--';
                    errorEl.textContent = '--';
                }
            });
        }
        
        function updateStatsTable(stats) {
            const tbody = document.getElementById('statsTableBody');
            tbody.innerHTML = '';
            
            const algorithms = ['WCL', 'TRI', 'BCCP', 'FUSED'];
            
            algorithms.forEach(algo => {
                const s = stats ? stats[algo] : null;
                const row = document.createElement('tr');
                
                if (s && s.count > 0) {
                    row.innerHTML = `
                        <td style="color: ${COLORS[algo]}">${algo}</td>
                        <td>${s.rmse ? s.rmse.toFixed(3) : '--'}</td>
                        <td>${s.mean ? s.mean.toFixed(3) : '--'}</td>
                        <td>${s.count}</td>
                    `;
                } else {
                    row.innerHTML = `
                        <td style="color: ${COLORS[algo]}">${algo}</td>
                        <td>--</td>
                        <td>--</td>
                        <td>0</td>
                    `;
                }
                
                tbody.appendChild(row);
            });
        }
        
        // Event Handlers
        function setupEventHandlers() {
            // Plot click handler
            const plot = document.getElementById('positionPlot');
            plot.addEventListener('click', (e) => {
                const rect = plot.getBoundingClientRect();
                const svg = document.getElementById('plotSvg');
                const svgRect = svg.getBoundingClientRect();
                
                const scaleX = PLOT_WIDTH / svgRect.width;
                const scaleY = PLOT_HEIGHT / svgRect.height;
                
                const px = (e.clientX - svgRect.left) * scaleX;
                const py = (e.clientY - svgRect.top) * scaleY;
                
                const meters = pixelsToMeters(px, py);
                
                // Clamp to room bounds
                meters.x = Math.max(0, Math.min(roomWidth, meters.x));
                meters.y = Math.max(0, Math.min(roomHeight, meters.y));
                
                // Update input fields
                document.getElementById('gtX').value = meters.x.toFixed(2);
                document.getElementById('gtY').value = meters.y.toFixed(2);
                
                // Set ground truth
                setGroundTruth(meters.x, meters.y);
            });
            
            // Ground truth buttons
            document.getElementById('setGtBtn').addEventListener('click', () => {
                const x = parseFloat(document.getElementById('gtX').value);
                const y = parseFloat(document.getElementById('gtY').value);
                
                if (!isNaN(x) && !isNaN(y)) {
                    setGroundTruth(x, y);
                }
            });
            
            document.getElementById('clearGtBtn').addEventListener('click', () => {
                groundTruth = null;
                document.getElementById('gtX').value = '';
                document.getElementById('gtY').value = '';
                fetch('/api/ground-truth', {method: 'DELETE'});
                updatePlot();
            });
            
            document.getElementById('resetStatsBtn').addEventListener('click', () => {
                fetch('/api/reset-stats', {method: 'POST'});
            });
            
            // Tab switching
            document.querySelectorAll('.tab').forEach(tab => {
                tab.addEventListener('click', () => {
                    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
                    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
                    
                    tab.classList.add('active');
                    document.getElementById(tab.dataset.tab + 'Tab').classList.add('active');
                });
            });
            
            // Config apply
            document.getElementById('applyConfigBtn').addEventListener('click', applyConfig);
            
            // Plot hover for tooltip
            const svg = document.getElementById('plotSvg');
            const tooltip = document.getElementById('plotTooltip');
            
            svg.addEventListener('mousemove', (e) => {
                const rect = svg.getBoundingClientRect();
                const scaleX = PLOT_WIDTH / rect.width;
                const scaleY = PLOT_HEIGHT / rect.height;
                
                const px = (e.clientX - rect.left) * scaleX;
                const py = (e.clientY - rect.top) * scaleY;
                
                const meters = pixelsToMeters(px, py);
                
                if (meters.x >= 0 && meters.x <= roomWidth && meters.y >= 0 && meters.y <= roomHeight) {
                    tooltip.textContent = `X: ${meters.x.toFixed(2)}m, Y: ${meters.y.toFixed(2)}m`;
                    tooltip.style.left = (e.clientX - plot.getBoundingClientRect().left + 10) + 'px';
                    tooltip.style.top = (e.clientY - plot.getBoundingClientRect().top + 10) + 'px';
                    tooltip.classList.add('visible');
                } else {
                    tooltip.classList.remove('visible');
                }
            });
            
            svg.addEventListener('mouseleave', () => {
                tooltip.classList.remove('visible');
            });
        }
        
        async function setGroundTruth(x, y) {
            groundTruth = {x, y};
            
            try {
                await fetch('/api/ground-truth', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({x, y, auto_reset: true})
                });
            } catch (e) {
                console.error('Failed to set ground truth:', e);
            }
            
            updatePlot();
        }
        
        async function applyConfig() {
            const config = {
                room_width: parseFloat(document.getElementById('configWidth').value),
                room_height: parseFloat(document.getElementById('configHeight').value),
                smoothing_factors: {
                    WCL: parseFloat(document.getElementById('smoothWcl').value),
                    TRI: parseFloat(document.getElementById('smoothTri').value),
                    BCCP: parseFloat(document.getElementById('smoothBccp').value),
                    FUSED: parseFloat(document.getElementById('smoothFused').value)
                }
            };
            
            try {
                await fetch('/api/config', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(config)
                });
                
                roomWidth = config.room_width;
                roomHeight = config.room_height;
                initPlot();
                updatePlot();
            } catch (e) {
                console.error('Failed to apply config:', e);
            }
        }
        
        function updatePlot() {
            if (currentData) {
                const positions = currentData.positions;
                const distances = positions && positions.WCL ? positions.WCL.distances : null;
                
                drawDistanceCircles(distances);
                drawPositions(positions);
                updateAlgoStats(positions);
                updateRssiCharts(distances, currentData.rssi_values);
                updateStatsTable(currentData.statistics);
            } else {
                drawPositions(null);
            }
        }
        
        // SSE Connection
        function connectSSE() {
            const statusDot = document.getElementById('statusDot');
            const statusText = document.getElementById('statusText');
            
            const eventSource = new EventSource('/api/stream');
            
            eventSource.onopen = () => {
                statusDot.classList.remove('disconnected');
                statusText.textContent = 'Connected';
            };
            
            eventSource.onmessage = (event) => {
                try {
                    currentData = JSON.parse(event.data);
                    updatePlot();
                    
                    // Update rate calculation
                    updateCount++;
                    const now = Date.now();
                    if (now - lastUpdateTime >= 1000) {
                        document.getElementById('updateRate').textContent = `${updateCount} Hz`;
                        updateCount = 0;
                        lastUpdateTime = now;
                    }
                } catch (e) {
                    console.error('Failed to parse SSE data:', e);
                }
            };
            
            eventSource.onerror = () => {
                statusDot.classList.add('disconnected');
                statusText.textContent = 'Disconnected';
                
                // Reconnect after 2 seconds
                setTimeout(() => {
                    eventSource.close();
                    connectSSE();
                }, 2000);
            };
        }
        
        // Load initial config
        async function loadConfig() {
            try {
                const response = await fetch('/api/config');
                const config = await response.json();
                
                roomWidth = config.room_width || 8.2;
                roomHeight = config.room_height || 8.0;
                
                if (config.anchor_positions) {
                    anchors = {};
                    Object.entries(config.anchor_positions).forEach(([id, pos]) => {
                        anchors[id] = {x: pos[0], y: pos[1]};
                    });
                }
                
                // Update config inputs
                document.getElementById('configWidth').value = roomWidth;
                document.getElementById('configHeight').value = roomHeight;
                
                if (config.smoothing_factors) {
                    document.getElementById('smoothWcl').value = config.smoothing_factors.WCL || 0.35;
                    document.getElementById('smoothTri').value = config.smoothing_factors.TRI || 0.20;
                    document.getElementById('smoothBccp').value = config.smoothing_factors.BCCP || 0.20;
                    document.getElementById('smoothFused').value = config.smoothing_factors.FUSED || 0.25;
                }
                
                // Load ground truth if exists
                if (config.ground_truth) {
                    groundTruth = config.ground_truth;
                    document.getElementById('gtX').value = groundTruth.x.toFixed(2);
                    document.getElementById('gtY').value = groundTruth.y.toFixed(2);
                }
                
            } catch (e) {
                console.error('Failed to load config:', e);
            }
        }
        
        // Initialize everything
        async function init() {
            await loadConfig();
            initPlot();
            initRssiCharts();
            initAlgoStats();
            setupEventHandlers();
            updateStatsTable(null);
            connectSSE();
        }
        
        // Start when DOM is ready
        document.addEventListener('DOMContentLoaded', init);
    </script>
</body>
</html>
"""


# =============================================================================
# API ROUTES
# =============================================================================

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Serve the main dashboard"""
    return DASHBOARD_HTML


@app.get("/api/stream")
async def stream_data():
    """Server-Sent Events stream for real-time data"""
    async def generate():
        while True:
            try:
                # Compute positions
                results = engine.compute_all_positions()
                
                # Get RSSI values
                rssi_values = {}
                for anchor_id in config.anchor_positions.keys():
                    rssi = engine.get_smoothed_rssi(anchor_id)
                    if rssi is not None:
                        rssi_values[anchor_id] = rssi
                
                # Prepare response
                data = {
                    "timestamp": time.time(),
                    "positions": {
                        name: {
                            "position": {
                                "x": result.position.x,
                                "y": result.position.y
                            } if result.position else None,
                            "distances": result.distances,
                            "error": result.error
                        }
                        for name, result in results.items()
                    },
                    "rssi_values": rssi_values,
                    "statistics": engine.get_statistics(),
                    "ground_truth": {
                        "x": engine.ground_truth.x,
                        "y": engine.ground_truth.y
                    } if engine.ground_truth else None
                }
                
                yield f"data: {json.dumps(data)}\n\n"
                
            except Exception as e:
                logger.error(f"Stream error: {e}")
            
            await asyncio.sleep(1.0 / config.update_rate_hz)
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@app.get("/api/config")
async def get_config():
    """Get current configuration"""
    return {
        "room_width": config.room_width,
        "room_height": config.room_height,
        "anchor_positions": config.anchor_positions,
        "calibration": config.calibration,
        "smoothing_factors": config.smoothing_factors,
        "ground_truth": {
            "x": engine.ground_truth.x,
            "y": engine.ground_truth.y
        } if engine.ground_truth else None
    }


@app.post("/api/config")
async def update_config(request: Request):
    """Update configuration"""
    data = await request.json()
    engine.update_config(data)
    return {"status": "ok"}


@app.post("/api/ground-truth")
async def set_ground_truth(request: Request):
    """Set ground truth position"""
    data = await request.json()
    x = float(data.get("x", 0))
    y = float(data.get("y", 0))
    auto_reset = data.get("auto_reset", True)
    
    engine.set_ground_truth(x, y, auto_reset)
    return {"status": "ok", "ground_truth": {"x": x, "y": y}}


@app.delete("/api/ground-truth")
async def clear_ground_truth():
    """Clear ground truth"""
    engine.clear_ground_truth()
    return {"status": "ok"}


@app.post("/api/reset-stats")
async def reset_stats():
    """Reset error statistics"""
    engine.clear_error_history()
    return {"status": "ok"}


@app.get("/api/positions")
async def get_positions():
    """Get current position estimates"""
    results = engine.compute_all_positions()
    return {
        name: {
            "position": {
                "x": result.position.x,
                "y": result.position.y
            } if result.position else None,
            "distances": result.distances,
            "error": result.error
        }
        for name, result in results.items()
    }


@app.get("/api/statistics")
async def get_statistics():
    """Get error statistics"""
    return engine.get_statistics()


@app.post("/api/rssi")
async def add_rssi(request: Request):
    """Add RSSI measurement (for testing without MQTT)"""
    data = await request.json()
    anchor_id = data.get("anchor_id") or data.get("anchor")
    rssi = data.get("rssi")
    
    if anchor_id and rssi is not None:
        engine.add_rssi(anchor_id, float(rssi))
        return {"status": "ok"}
    
    return {"status": "error", "message": "Invalid data"}


@app.post("/api/rssi/bulk")
async def add_rssi_bulk(request: Request):
    """Add multiple RSSI measurements"""
    data = await request.json()
    
    for item in data:
        anchor_id = item.get("anchor_id") or item.get("anchor")
        rssi = item.get("rssi")
        
        if anchor_id and rssi is not None:
            engine.add_rssi(anchor_id, float(rssi))
    
    return {"status": "ok", "count": len(data)}


# =============================================================================
# STARTUP AND SHUTDOWN
# =============================================================================

@app.on_event("startup")
async def startup():
    """Initialize on startup"""
    logger.info("Starting IPS Dashboard Server...")
    
    # Connect to MQTT broker
    mqtt_handler.connect()
    
    # Add some demo data for testing (remove in production)
    # This simulates RSSI values for a device at approximately (4, 4)
    import random
    
    async def demo_data():
        """Generate demo RSSI data for testing"""
        while True:
            # Simulate device at position (4, 4)
            target_x, target_y = 4.0, 4.0
            
            for anchor_id, pos in config.anchor_positions.items():
                # Calculate true distance
                true_dist = math.sqrt((target_x - pos[0])**2 + (target_y - pos[1])**2)
                
                # Convert to RSSI with noise
                calib = config.calibration[anchor_id]
                rssi = calib["rssi1m"] - 10 * calib["n"] * math.log10(max(0.1, true_dist))
                rssi += random.gauss(0, 2)  # Add noise
                
                engine.add_rssi(anchor_id, rssi)
            
            await asyncio.sleep(0.1)
    
    # Start demo data generation (comment out when using real MQTT)
    # asyncio.create_task(demo_data())
    
    logger.info(f"Server running at http://{config.server_host}:{config.server_port}")


@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown"""
    logger.info("Shutting down IPS Dashboard Server...")
    mqtt_handler.disconnect()


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "ips_dashboard_server:app",
        host=config.server_host,
        port=config.server_port,
        reload=True,
        log_level="info"
    )
