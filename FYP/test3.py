"""
2D Indoor Positioning Tracker - IMPROVED VERSION
================================================
Improvements over original:
1. Position-level Exponential Moving Average (EMA) smoothing
2. Median-based RSSI filtering with outlier rejection
3. Room bounds clamping to prevent wild estimates
4. Weighted fusion of all three methods
5. Configurable smoothing parameters
6. Better error handling and edge cases
7. Velocity-based outlier rejection

Requires 3+ anchors for 2D positioning.
"""

import json
import time
from collections import deque
from queue import Queue, Empty
from itertools import combinations
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, List

import numpy as np
import paho.mqtt.client as mqtt
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# =============================================================================
# CONFIGURATION
# =============================================================================

# ----- MQTT Settings -----
BROKER_HOST = "127.0.0.1"
BROKER_PORT = 1883
TOPIC = "ips/rssi"

# ----- Anchor Positions (x, y) in meters -----
ANCHOR_POS = {
    "A1": (0, 0),
    "A2": (8.2,8 ),
    "A3": (0, 8),
    "A4": (8.2, 0),
}

# ----- Calibration per anchor -----
CALIB = {
    "A1": {"rssi1m": -74.69, "n": 1.813},
    "A2": {"rssi1m": -72.86, "n": 2.334},
    "A3": {"rssi1m": -75.86, "n": 2.522},
    "A4": {"rssi1m": -73.55, "n": 2.478},
}

# ----- RSSI Processing Settings -----
RSSI_WINDOW_SIZE = 10          # Number of RSSI samples to keep (increased from 5)
RSSI_TRIM_PERCENT = 0.2        # Trim top/bottom 20% for outlier rejection
D_MIN, D_MAX = 0.1, 12.0       # Distance clamp range (meters)
MAX_AGE = 2.0                  # Maximum age of anchor data (seconds)
MIN_ANCHORS = 3                # Minimum anchors needed for positioning

# ----- Position Smoothing Settings -----
POS_ALPHA_WCL = 0.35           # WCL is already stable, less smoothing needed
POS_ALPHA_TRI = 0.20           # Trilateration needs more smoothing
POS_ALPHA_BCCP = 0.20          # BCCP needs more smoothing
POS_ALPHA_FUSED = 0.25         # Fused position smoothing

# ----- Fusion Weights -----
FUSION_WEIGHTS = {
    "wcl": 0.25,
    "tri": 0.40,
    "bccp": 0.35,
}

# ----- Room Bounds -----
ROOM_MARGIN = 0.3  # Allow positions slightly outside anchor bounds

# ----- Outlier Rejection -----
MAX_JUMP_DISTANCE = 2.0        # Maximum allowed position jump per update (meters)
MAX_VELOCITY = 3.0             # Maximum realistic velocity (m/s)

# ----- Update Rate -----
MIN_UPDATE_INTERVAL = 0.08


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class AnchorData:
    """Stores data for a single anchor."""
    rssi_hist: deque = field(default_factory=lambda: deque(maxlen=RSSI_WINDOW_SIZE))
    last_update: float = 0.0
    distance: Optional[float] = None
    last_mac: Optional[str] = None
    raw_rssi: Optional[float] = None
    filtered_rssi: Optional[float] = None


class PositionSmoother:
    """
    Exponential Moving Average smoother for position coordinates.
    Includes velocity-based outlier rejection.
    """
    
    def __init__(self, alpha: float = 0.3, max_jump: float = MAX_JUMP_DISTANCE):
        self.alpha = alpha
        self.max_jump = max_jump
        self.x: Optional[float] = None
        self.y: Optional[float] = None
        self.last_time: float = 0
        self._initialized = False
    
    def update(self, x_new: float, y_new: float, timestamp: float = None) -> Tuple[float, float]:
        """Update smoother with new position and return smoothed position."""
        if timestamp is None:
            timestamp = time.time()
        
        # Handle invalid inputs
        if not np.isfinite(x_new) or not np.isfinite(y_new):
            if self._initialized:
                return self.x, self.y
            return np.nan, np.nan
        
        if not self._initialized:
            self.x, self.y = x_new, y_new
            self.last_time = timestamp
            self._initialized = True
        else:
            # Calculate jump distance
            jump = np.hypot(x_new - self.x, y_new - self.y)
            dt = max(timestamp - self.last_time, 0.01)
            velocity = jump / dt
            
            # Reject outliers based on velocity
            if velocity > MAX_VELOCITY and jump > self.max_jump:
                # Large jump - reduce alpha significantly (trust history more)
                effective_alpha = self.alpha * 0.1
            elif jump > self.max_jump * 0.5:
                # Medium jump - reduce alpha moderately
                effective_alpha = self.alpha * 0.5
            else:
                effective_alpha = self.alpha
            
            self.x = effective_alpha * x_new + (1 - effective_alpha) * self.x
            self.y = effective_alpha * y_new + (1 - effective_alpha) * self.y
            self.last_time = timestamp
        
        return self.x, self.y
    
    def reset(self):
        """Reset the smoother state."""
        self.x = None
        self.y = None
        self._initialized = False
        self.last_time = 0
    
    def get_current(self) -> Tuple[Optional[float], Optional[float]]:
        """Get current smoothed position without updating."""
        return self.x, self.y


# =============================================================================
# GLOBAL STATE
# =============================================================================

anchors = {name: AnchorData() for name in ANCHOR_POS}

# Position smoothers for each method
smoother_wcl = PositionSmoother(alpha=POS_ALPHA_WCL)
smoother_tri = PositionSmoother(alpha=POS_ALPHA_TRI)
smoother_bccp = PositionSmoother(alpha=POS_ALPHA_BCCP)
smoother_fused = PositionSmoother(alpha=POS_ALPHA_FUSED)

frame_count = 0
last_plot_time = 0
position_queue = Queue(maxsize=1)


# =============================================================================
# RSSI PROCESSING FUNCTIONS
# =============================================================================

def filter_rssi_trimmed_mean(rssi_hist: deque) -> float:
    """
    Calculate trimmed mean of RSSI values.
    Removes top and bottom percentile to reject outliers.
    """
    if len(rssi_hist) == 0:
        return np.nan
    
    if len(rssi_hist) < 3:
        return float(np.mean(rssi_hist))
    
    sorted_rssi = sorted(rssi_hist)
    n = len(sorted_rssi)
    trim_count = max(1, int(n * RSSI_TRIM_PERCENT))
    
    if 2 * trim_count >= n:
        return float(np.median(rssi_hist))
    
    trimmed = sorted_rssi[trim_count:-trim_count]
    return float(np.mean(trimmed))


def rssi_to_distance(rssi: float, rssi1m: float, n: float) -> float:
    """Convert RSSI to distance using log-distance path loss model."""
    if not np.isfinite(rssi):
        return D_MAX
    
    try:
        d = 10 ** ((rssi1m - rssi) / (10.0 * n))
        return float(np.clip(d, D_MIN, D_MAX))
    except (ValueError, OverflowError):
        return D_MAX


def clamp_to_room(x: float, y: float, margin: float = ROOM_MARGIN) -> Tuple[float, float]:
    """Clamp position to room bounds with margin."""
    if not np.isfinite(x) or not np.isfinite(y):
        return x, y
    
    xs = [p[0] for p in ANCHOR_POS.values()]
    ys = [p[1] for p in ANCHOR_POS.values()]
    
    x = float(np.clip(x, min(xs) - margin, max(xs) + margin))
    y = float(np.clip(y, min(ys) - margin, max(ys) + margin))
    
    return x, y


# =============================================================================
# POSITIONING ALGORITHMS
# =============================================================================

def wcl_2d(distances: Dict[str, float], p: float = 2.0) -> Tuple[float, float]:
    """Weighted Centroid Localization in 2D."""
    num_x, num_y, den = 0.0, 0.0, 0.0
    
    for anchor, dist in distances.items():
        if anchor not in ANCHOR_POS:
            continue
        if not np.isfinite(dist) or dist <= 0:
            continue
            
        x, y = ANCHOR_POS[anchor]
        w = 1.0 / (dist ** p + 1e-9)
        num_x += w * x
        num_y += w * y
        den += w
    
    if den <= 0:
        return np.nan, np.nan
    
    return num_x / den, num_y / den


def trilateration_2d(distances: Dict[str, float]) -> Tuple[float, float]:
    """
    Trilateration using linearized least squares without a reference anchor.
    
    Introduces c = x² + y² as an extra variable to keep all equations.
    Solves the system: -2x_i x - 2y_i y + c = d_i² - x_i² - y_i² for all i.
    """
    anchor_list = [a for a in distances if a in ANCHOR_POS and np.isfinite(distances[a])]
    
    if len(anchor_list) < 3:
        return np.nan, np.nan
    
    # Build the linear system Ax = b where x = [x, y, c]
    A = []
    b = []
    
    for anchor in anchor_list:
        xi, yi = ANCHOR_POS[anchor]
        di = distances[anchor]
        A.append([-2 * xi, -2 * yi, 1])
        b.append(di**2 - xi**2 - yi**2)
    
    A = np.array(A)
    b = np.array(b)
    
    try:
        result, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        x, y, c = result
        return float(x), float(y)
    except (np.linalg.LinAlgError, ValueError):
        return np.nan, np.nan

# =============================================================================
# BCCP FUNCTIONS
# =============================================================================

def _circ_ints(C1, r1, C2, r2, tol=1e-8):
    """Return up to two circle intersection points."""
    C1 = np.asarray(C1, float)
    C2 = np.asarray(C2, float)
    d = np.linalg.norm(C2 - C1)
    
    if not np.isfinite(d) or d < 1e-12:
        return np.array([np.nan, np.nan]), np.array([np.nan, np.nan])
    if d > r1 + r2 + tol or d < abs(r1 - r2) - tol:
        return np.array([np.nan, np.nan]), np.array([np.nan, np.nan])

    ex = (C2 - C1) / d
    a = (r1*r1 - r2*r2 + d*d) / (2*d)
    h2 = r1*r1 - a*a
    h = np.sqrt(max(h2, 0.0))

    P0 = C1 + a * ex
    ey = np.array([-ex[1], ex[0]])
    P1 = P0 + h * ey
    P2 = P0 - h * ey
    return P1, P2


def _pick_cp(Pa, Pb, Ck, rk):
    """Pick the intersection point that best satisfies circle k."""
    Pa = np.asarray(Pa, float)
    Pb = np.asarray(Pb, float)
    okA = np.all(np.isfinite(Pa))
    okB = np.all(np.isfinite(Pb))
    
    if not okA and not okB:
        return np.array([np.nan, np.nan])
    if okA and not okB:
        return Pa
    if okB and not okA:
        return Pb

    ea = abs(np.linalg.norm(Pa - Ck) - rk)
    eb = abs(np.linalg.norm(Pb - Ck) - rk)
    return Pa if ea <= eb else Pb


def _project_radii_to_intersect(AP, D_in, eps=1e-6, iters=15):
    """Make the 3 circles intersect/overlap by adjusting radii minimally."""
    AP = np.asarray(AP, float)
    D = np.asarray(D_in, float).copy()
    pairs = [(0, 1), (0, 2), (1, 2)]

    for _ in range(iters):
        changed = False
        for i, j in pairs:
            dij = np.linalg.norm(AP[i] - AP[j])

            if D[i] + D[j] < dij - eps:
                delta = (dij - eps) - (D[i] + D[j])
                D[i] += 0.5 * delta
                D[j] += 0.5 * delta
                changed = True

            if abs(D[i] - D[j]) > dij + eps:
                S = D[i] + D[j]
                if D[i] >= D[j]:
                    D[i] = (S + (dij + eps)) / 2.0
                    D[j] = S - D[i]
                else:
                    D[j] = (S + (dij + eps)) / 2.0
                    D[i] = S - D[j]
                changed = True

        if not changed:
            break
    return D


def _inflate_for_overlap(AP, D0, alpha=0.20, eps=1e-6, max_iter=12):
    """Inflate radii so all pairs overlap robustly."""
    AP = np.asarray(AP, float)
    D0 = np.asarray(D0, float)
    pairs = [(0, 1), (0, 2), (1, 2)]

    s = 1.0
    for i, j in pairs:
        dij = np.linalg.norm(AP[i] - AP[j])
        den = D0[i] + D0[j]
        if den > 0:
            s = max(s, ((1.0 + alpha) * dij) / den)

    D = _project_radii_to_intersect(AP, D0 * s, eps=eps, iters=20)
    for _ in range(max_iter):
        ok = True
        for i, j in pairs:
            dij = np.linalg.norm(AP[i] - AP[j])
            if D[i] + D[j] < (1.0 + alpha) * dij - eps:
                ok = False
                break
        if ok:
            return D
        s *= 1.03
        D = _project_radii_to_intersect(AP, D0 * s, eps=eps, iters=20)
    return D


def _bccp_once_3anchors(anchor_names, anchor_pos, distances):
    """One-shot BCCP with exactly 3 anchors."""
    AP = np.array([anchor_pos[a] for a in anchor_names], float)
    D0 = np.array([max(float(distances[a]), 1e-9) for a in anchor_names], float)

    D = _project_radii_to_intersect(AP, D0)
    D = _inflate_for_overlap(AP, D, alpha=0.20)

    E, F = _circ_ints(AP[0], D[0], AP[1], D[1])
    G, H = _circ_ints(AP[0], D[0], AP[2], D[2])
    I, J = _circ_ints(AP[1], D[1], AP[2], D[2])

    CP12 = _pick_cp(E, F, AP[2], D[2])
    CP13 = _pick_cp(G, H, AP[1], D[1])
    CP23 = _pick_cp(I, J, AP[0], D[0])

    CPs = np.vstack([CP12, CP13, CP23])
    CPs = CPs[np.all(np.isfinite(CPs), axis=1)]
    
    if CPs.shape[0] < 3:
        return float("nan"), float("nan")

    p = np.mean(CPs, axis=0)
    return float(p[0]), float(p[1])


def _residual_sum(p, anchor_list, anchor_pos, distances):
    """Sum of squared range residuals."""
    x, y = p
    s = 0.0
    for a in anchor_list:
        ax, ay = anchor_pos[a]
        d_meas = float(distances[a])
        d_act = np.hypot(x - ax, y - ay)
        s += (d_act - d_meas) ** 2
    return s


def bccp_2d(distances: dict, clamp_bounds=None):
    """BCCP for 3 or 4 anchors."""
    anchors_list = [a for a in distances if a in ANCHOR_POS and np.isfinite(distances[a])]
    
    if len(anchors_list) < 3:
        return float("nan"), float("nan")

    best = None
    for trip in combinations(anchors_list, 3):
        x, y = _bccp_once_3anchors(trip, ANCHOR_POS, distances)
        if not np.isfinite(x) or not np.isfinite(y):
            continue

        score = _residual_sum((x, y), anchors_list, ANCHOR_POS, distances)

        if best is None or score < best[0]:
            best = (score, (x, y), trip)

    if best is None:
        return float("nan"), float("nan")

    x, y = best[1]
    if clamp_bounds is not None:
        (xmin, ymin), (xmax, ymax) = clamp_bounds
        x = min(max(x, xmin), xmax)
        y = min(max(y, ymin), ymax)

    return float(x), float(y)


# =============================================================================
# FUSION ALGORITHM
# =============================================================================

def fused_position(wcl: Tuple[float, float], 
                   tri: Tuple[float, float], 
                   bccp: Tuple[float, float],
                   distances: Dict[str, float] = None) -> Tuple[float, float]:
    """
    Fuse multiple position estimates using weighted average.
    """
    positions = []
    weights = []
    
    if np.isfinite(wcl[0]) and np.isfinite(wcl[1]):
        positions.append(wcl)
        weights.append(FUSION_WEIGHTS["wcl"])
    
    if np.isfinite(tri[0]) and np.isfinite(tri[1]):
        positions.append(tri)
        weights.append(FUSION_WEIGHTS["tri"])
    
    if np.isfinite(bccp[0]) and np.isfinite(bccp[1]):
        positions.append(bccp)
        weights.append(FUSION_WEIGHTS["bccp"])
    
    if not positions:
        return np.nan, np.nan
    
    if len(positions) == 1:
        return positions[0]
    
    weights = np.array(weights)
    weights = weights / weights.sum()
    positions = np.array(positions)
    
    x = float(np.sum(weights * positions[:, 0]))
    y = float(np.sum(weights * positions[:, 1]))
    
    return x, y


def adaptive_fused_position(wcl: Tuple[float, float], 
                            tri: Tuple[float, float], 
                            bccp: Tuple[float, float],
                            distances: Dict[str, float]) -> Tuple[float, float]:
    """
    Adaptive fusion that adjusts weights based on position agreement.
    
    If methods agree, use standard weights.
    If methods disagree, favor the more stable method (WCL).
    """
    positions = []
    methods = []
    
    if np.isfinite(wcl[0]) and np.isfinite(wcl[1]):
        positions.append(wcl)
        methods.append("wcl")
    
    if np.isfinite(tri[0]) and np.isfinite(tri[1]):
        positions.append(tri)
        methods.append("tri")
    
    if np.isfinite(bccp[0]) and np.isfinite(bccp[1]):
        positions.append(bccp)
        methods.append("bccp")
    
    if len(positions) == 0:
        return np.nan, np.nan
    
    if len(positions) == 1:
        return positions[0]
    
    positions = np.array(positions)
    
    # Calculate pairwise distances between estimates
    spread = 0.0
    count = 0
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            spread += np.linalg.norm(positions[i] - positions[j])
            count += 1
    
    avg_spread = spread / count if count > 0 else 0
    
    # Adjust weights based on spread
    if avg_spread > 1.5:  # Methods disagree significantly
        adjusted_weights = {"wcl": 0.5, "tri": 0.25, "bccp": 0.25}
    elif avg_spread > 0.8:  # Moderate disagreement
        adjusted_weights = {"wcl": 0.35, "tri": 0.35, "bccp": 0.30}
    else:  # Good agreement
        adjusted_weights = FUSION_WEIGHTS
    
    weights = np.array([adjusted_weights[m] for m in methods])
    weights = weights / weights.sum()
    
    x = float(np.sum(weights * positions[:, 0]))
    y = float(np.sum(weights * positions[:, 1]))
    
    return x, y


# =============================================================================
# MQTT HANDLERS
# =============================================================================

def on_connect(client, userdata, flags, rc):
    print(f"[MQTT] Connected (rc={rc})")
    client.subscribe(TOPIC)
    print(f"[MQTT] Subscribed to {TOPIC}")


def on_message(client, userdata, msg):
    global frame_count, last_plot_time
    
    try:
        data = json.loads(msg.payload.decode())
    except:
        return
    
    anchor = data.get("anchor")
    mac = data.get("mac")
    rssi = data.get("rssi")
    
    if anchor not in anchors or not isinstance(rssi, (int, float)):
        return
    
    ts = time.time()
    
    # Update anchor data
    a = anchors[anchor]
    a.rssi_hist.append(rssi)
    a.last_update = ts
    a.last_mac = mac
    a.raw_rssi = rssi
    
    # Apply trimmed mean filter to RSSI
    rssi_filtered = filter_rssi_trimmed_mean(a.rssi_hist)
    a.filtered_rssi = rssi_filtered
    
    # Convert to distance
    c = CALIB[anchor]
    a.distance = rssi_to_distance(rssi_filtered, c["rssi1m"], c["n"])
    
    # Collect valid anchor distances
    valid_anchors = {}
    for name, anchor_data in anchors.items():
        if anchor_data.distance is None:
            continue
        age = ts - anchor_data.last_update
        if age <= MAX_AGE:
            valid_anchors[name] = anchor_data.distance
    
    if len(valid_anchors) < MIN_ANCHORS:
        return
    
    # Rate limiting
    if ts - last_plot_time < MIN_UPDATE_INTERVAL:
        return
    
    last_plot_time = ts
    frame_count += 1
    
    # Calculate RAW positions using all methods
    x_wcl_raw, y_wcl_raw = wcl_2d(valid_anchors, p=2.0)
    x_tri_raw, y_tri_raw = trilateration_2d(valid_anchors)
    x_bccp_raw, y_bccp_raw = bccp_2d(valid_anchors)
    
    # Clamp to room bounds
    x_wcl_raw, y_wcl_raw = clamp_to_room(x_wcl_raw, y_wcl_raw)
    x_tri_raw, y_tri_raw = clamp_to_room(x_tri_raw, y_tri_raw)
    x_bccp_raw, y_bccp_raw = clamp_to_room(x_bccp_raw, y_bccp_raw)
    
    # Apply position smoothing
    x_wcl, y_wcl = smoother_wcl.update(x_wcl_raw, y_wcl_raw, ts)
    x_tri, y_tri = smoother_tri.update(x_tri_raw, y_tri_raw, ts)
    x_bccp, y_bccp = smoother_bccp.update(x_bccp_raw, y_bccp_raw, ts)
    
    # Calculate fused position (using smoothed values)
    x_fused_raw, y_fused_raw = adaptive_fused_position(
        (x_wcl, y_wcl), (x_tri, y_tri), (x_bccp, y_bccp), valid_anchors
    )
    x_fused, y_fused = smoother_fused.update(x_fused_raw, y_fused_raw, ts)
    x_fused, y_fused = clamp_to_room(x_fused, y_fused)
    
    # Log output
    print(f"[{frame_count:04d}] Anchors: {len(valid_anchors)} | "
          f"WCL=({x_wcl:.2f},{y_wcl:.2f}) "
          f"TRI=({x_tri:.2f},{y_tri:.2f}) "
          f"BCCP=({x_bccp:.2f},{y_bccp:.2f}) "
          f"FUSED=({x_fused:.2f},{y_fused:.2f})")
    
    # Send to plot queue
    try:
        position_queue.put_nowait({
            "distances": valid_anchors,
            "wcl": (x_wcl, y_wcl),
            "tri": (x_tri, y_tri),
            "bccp": (x_bccp, y_bccp),
            "fused": (x_fused, y_fused),
            "raw": {
                "wcl": (x_wcl_raw, y_wcl_raw),
                "tri": (x_tri_raw, y_tri_raw),
                "bccp": (x_bccp_raw, y_bccp_raw),
            }
        })
    except:
        pass


# =============================================================================
# MAIN VISUALIZATION
# =============================================================================

def main():
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(BROKER_HOST, BROKER_PORT)
    client.loop_start()
    
    plt.ion()
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Set up plot bounds
    xs = [p[0] for p in ANCHOR_POS.values()]
    ys = [p[1] for p in ANCHOR_POS.values()]
    margin = 1.0
    ax.set_xlim(min(xs) - margin, max(xs) + margin)
    ax.set_ylim(min(ys) - margin, max(ys) + margin)
    ax.set_aspect('equal')
    ax.set_xlabel("X (meters)", fontsize=12)
    ax.set_ylabel("Y (meters)", fontsize=12)
    ax.set_title("2D Indoor Positioning - Improved with Fusion & Smoothing", fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Plot anchors
    for name, (x, y) in ANCHOR_POS.items():
        ax.plot(x, y, 's', color='black', markersize=15, zorder=10)
        ax.annotate(name, (x, y), textcoords="offset points",
                   xytext=(0, 12), ha='center', fontweight='bold', fontsize=11)
    
    # Distance circles
    circles = {}
    for name, (x, y) in ANCHOR_POS.items():
        circle = plt.Circle((x, y), 0.1, fill=False, linestyle='--', alpha=0.3, color='gray')
        ax.add_patch(circle)
        circles[name] = circle
    
    # Position markers
    dot_wcl, = ax.plot([np.nan], [np.nan], 'o', color='blue', 
                       markersize=12, label='WCL', alpha=0.7)
    dot_tri, = ax.plot([np.nan], [np.nan], '^', color='orange', 
                       markersize=12, label='Trilateration', alpha=0.7)
    dot_bccp, = ax.plot([np.nan], [np.nan], 'p', color='purple', 
                        markersize=12, label='BCCP', alpha=0.7)
    dot_fused, = ax.plot([np.nan], [np.nan], '*', color='red', 
                         markersize=18, label='FUSED (Best)', alpha=0.9, 
                         markeredgecolor='darkred', markeredgewidth=1)
    
    ax.legend(handles=[dot_wcl, dot_tri, dot_bccp, dot_fused], 
              loc='upper right', title='Methods', fontsize=10)
    
    # Info text box
    info = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=9,
                   va='top', family='monospace',
                   bbox=dict(facecolor='wheat', alpha=0.85, edgecolor='gray'))
    
    fig.tight_layout()
    
    print("=" * 60)
    print("[MAIN] Indoor Positioning System - IMPROVED VERSION")
    print("=" * 60)
    print(f"[MAIN] Using {len(ANCHOR_POS)} anchors: {list(ANCHOR_POS.keys())}")
    print(f"[MAIN] RSSI Window: {RSSI_WINDOW_SIZE} samples")
    print(f"[MAIN] Smoothing alphas - WCL: {POS_ALPHA_WCL}, TRI: {POS_ALPHA_TRI}, "
          f"BCCP: {POS_ALPHA_BCCP}, FUSED: {POS_ALPHA_FUSED}")
    print(f"[MAIN] Fusion weights: {FUSION_WEIGHTS}")
    print("=" * 60)
    print("[MAIN] Running... Close window to stop")
    
    running = True
    
    def on_close(event):
        nonlocal running
        running = False
    
    fig.canvas.mpl_connect('close_event', on_close)
    
    try:
        while running and plt.fignum_exists(fig.number):
            try:
                result = None
                while True:
                    try:
                        result = position_queue.get_nowait()
                    except Empty:
                        break
                
                if result:
                    dot_wcl.set_xdata([result["wcl"][0]])
                    dot_wcl.set_ydata([result["wcl"][1]])
                    
                    dot_tri.set_xdata([result["tri"][0]])
                    dot_tri.set_ydata([result["tri"][1]])
                    
                    dot_bccp.set_xdata([result["bccp"][0]])
                    dot_bccp.set_ydata([result["bccp"][1]])
                    
                    dot_fused.set_xdata([result["fused"][0]])
                    dot_fused.set_ydata([result["fused"][1]])
                    
                    for name, dist in result["distances"].items():
                        if name in circles:
                            circles[name].set_radius(dist)
                    
                    lines = ["━" * 22]
                    lines.append("  DISTANCES (meters)")
                    lines.append("━" * 22)
                    for name, dist in sorted(result["distances"].items()):
                        rssi_info = ""
                        if anchors[name].filtered_rssi:
                            rssi_info = f" ({anchors[name].filtered_rssi:.0f}dBm)"
                        lines.append(f"  {name}: {dist:.2f}m{rssi_info}")
                    
                    lines.append("━" * 22)
                    lines.append("  POSITIONS (smoothed)")
                    lines.append("━" * 22)
                    lines.append(f"  WCL:   ({result['wcl'][0]:.2f}, {result['wcl'][1]:.2f})")
                    lines.append(f"  TRI:   ({result['tri'][0]:.2f}, {result['tri'][1]:.2f})")
                    lines.append(f"  BCCP:  ({result['bccp'][0]:.2f}, {result['bccp'][1]:.2f})")
                    lines.append("━" * 22)
                    lines.append(f"  ★ FUSED: ({result['fused'][0]:.2f}, {result['fused'][1]:.2f})")
                    lines.append("━" * 22)
                    
                    info.set_text('\n'.join(lines))
                    
                    fig.canvas.draw_idle()
                    fig.canvas.flush_events()
                    
            except Exception as e:
                print(f"[ERROR] Plot error: {e}")
            
            try:
                plt.pause(0.05)
            except:
                break
            
    except KeyboardInterrupt:
        print("\n[MAIN] Interrupted by user")
    finally:
        client.loop_stop()
        client.disconnect()
        plt.close('all')
        print("[MAIN] Done")


if __name__ == "__main__":
    main()