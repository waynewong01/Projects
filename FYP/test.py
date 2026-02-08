"""
2D Indoor Positioning Tracker - FAIR ALGORITHM COMPARISON VERSION
==================================================================

This version is designed for FAIR comparison of three localization algorithms:
1. Weighted Centroid Localization (WCL)
2. Least Squares Trilateration
3. Barycentric Coordinates using Closed Points (BCCP)

Key Principles:
- Same RSSI pre-processing for all algorithms
- Same smoothing (or no smoothing) applied equally to all
- No algorithm-specific bias corrections
- Pure algorithm outputs for accurate comparison
- Separate statistics for center vs boundary regions

For EE4002D Indoor Tracking Project
"""

import json
import time
import math
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

# ----- Room Dimensions (meters) -----
ROOM_WIDTH = 4.2
ROOM_HEIGHT = 4.17

# ----- Anchor Positions (x, y) in meters -----
# Modify these to match your actual anchor positions
ANCHOR_POS = {
    "A1": (0, 0),
    "A2": (0, ROOM_HEIGHT),
    "A3": (ROOM_WIDTH, 0),
    "A4": (ROOM_WIDTH, ROOM_HEIGHT),
}

# ----- Calibration per anchor -----
# Modify rssi1m and n based on your calibration
CALIB = {
    "A1": {"rssi1m": -74.69, "n": 1.813},
    "A2": {"rssi1m": -72.86, "n": 2.334},
    "A3": {"rssi1m": -75.86, "n": 2.522},
    "A4": {"rssi1m": -73.55, "n": 2.478},
}

# ----- RSSI Processing Settings -----
RSSI_WINDOW_SIZE = 10          # Number of RSSI samples to keep
RSSI_TRIM_PERCENT = 0.2        # Trim top/bottom 20% for outlier rejection
D_MIN, D_MAX = 0.1, 15.0       # Distance clamp range (meters)
MAX_AGE = 2.0                  # Maximum age of anchor data (seconds)
MIN_ANCHORS = 3                # Minimum anchors needed for positioning

# ----- Position Smoothing Settings -----
# Set to 1.0 for NO smoothing (pure algorithm comparison)
# Set to 0.3 for moderate smoothing (better for demo visualization)
ENABLE_SMOOTHING = True        # Set to False for pure comparison
SMOOTHING_ALPHA = 0.3          # SAME alpha for ALL algorithms (fair comparison)

# ----- Room Bounds -----
ROOM_MARGIN = 0.3              # Allow positions slightly outside anchor bounds

# ----- Boundary Analysis -----
BOUNDARY_MARGIN = 1.0          # Distance from edge to be considered "boundary"

# ----- Update Rate -----
MIN_UPDATE_INTERVAL = 0.1

# ----- Fusion Weights (for combined estimate only) -----
# These do NOT affect individual algorithm comparisons
FUSION_WEIGHTS = {
    "wcl": 0.25,
    "tri": 0.40,
    "bccp": 0.35,
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class AnchorData:
    """Stores data for a single anchor."""
    rssi_hist: deque = field(default_factory=lambda: deque(maxlen=RSSI_WINDOW_SIZE))
    last_update: float = 0.0
    distance: Optional[float] = None
    filtered_rssi: Optional[float] = None


class PositionSmoother:
    """
    Simple Exponential Moving Average smoother for position coordinates.
    Applied EQUALLY to all algorithms for fair comparison.
    """
    
    def __init__(self, alpha: float = 0.3):
        self.alpha = alpha
        self.x: Optional[float] = None
        self.y: Optional[float] = None
        self._initialized = False
    
    def update(self, x_new: float, y_new: float) -> Tuple[float, float]:
        """Update smoother with new position and return smoothed position."""
        # Handle invalid inputs
        if not np.isfinite(x_new) or not np.isfinite(y_new):
            if self._initialized:
                return self.x, self.y
            return np.nan, np.nan
        
        if not self._initialized or not ENABLE_SMOOTHING:
            self.x, self.y = x_new, y_new
            self._initialized = True
        else:
            # Simple EMA - same for all algorithms
            self.x = self.alpha * x_new + (1 - self.alpha) * self.x
            self.y = self.alpha * y_new + (1 - self.alpha) * self.y
        
        return self.x, self.y
    
    def reset(self):
        """Reset the smoother state."""
        self.x = None
        self.y = None
        self._initialized = False


@dataclass
class ErrorStatistics:
    """Stores error statistics for analysis."""
    center_errors: List[float] = field(default_factory=list)
    boundary_errors: List[float] = field(default_factory=list)
    all_errors: List[float] = field(default_factory=list)
    
    def add_error(self, error: float, is_boundary: bool):
        self.all_errors.append(error)
        if is_boundary:
            self.boundary_errors.append(error)
        else:
            self.center_errors.append(error)
    
    def get_rmse(self, region: str = 'all') -> float:
        if region == 'center':
            errors = self.center_errors
        elif region == 'boundary':
            errors = self.boundary_errors
        else:
            errors = self.all_errors
        
        if len(errors) == 0:
            return 0.0
        return np.sqrt(np.mean(np.array(errors) ** 2))
    
    def get_mean(self, region: str = 'all') -> float:
        if region == 'center':
            errors = self.center_errors
        elif region == 'boundary':
            errors = self.boundary_errors
        else:
            errors = self.all_errors
        
        if len(errors) == 0:
            return 0.0
        return np.mean(errors)


# =============================================================================
# GLOBAL STATE
# =============================================================================

anchors = {name: AnchorData() for name in ANCHOR_POS}

# Position smoothers - SAME alpha for all (fair comparison)
smoother_wcl = PositionSmoother(alpha=SMOOTHING_ALPHA)
smoother_tri = PositionSmoother(alpha=SMOOTHING_ALPHA)
smoother_bccp = PositionSmoother(alpha=SMOOTHING_ALPHA)
smoother_fused = PositionSmoother(alpha=SMOOTHING_ALPHA)

# Error statistics for each algorithm
stats_wcl = ErrorStatistics()
stats_tri = ErrorStatistics()
stats_bccp = ErrorStatistics()
stats_fused = ErrorStatistics()

# Ground truth position (set manually for error calculation)
# Set to None if not measuring error
GROUND_TRUTH_POS: Optional[Tuple[float, float]] = None

frame_count = 0
last_plot_time = 0
position_queue = Queue(maxsize=1)


# =============================================================================
# RSSI PROCESSING FUNCTIONS (Same for all algorithms)
# =============================================================================

def filter_rssi_trimmed_mean(rssi_hist: deque) -> float:
    """
    Calculate trimmed mean of RSSI values.
    Removes top and bottom percentile to reject outliers.
    This is applied EQUALLY to all algorithms.
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
    """
    Convert RSSI to distance using log-distance path loss model.
    d = 10 ^ ((rssi1m - rssi) / (10 * n))
    """
    if not np.isfinite(rssi):
        return D_MAX
    
    try:
        d = 10 ** ((rssi1m - rssi) / (10.0 * n))
        return float(np.clip(d, D_MIN, D_MAX))
    except (ValueError, OverflowError):
        return D_MAX


def clamp_to_room(x: float, y: float) -> Tuple[float, float]:
    """
    Clamp position to room bounds.
    This is a PHYSICAL constraint, applied equally to all algorithms.
    """
    if not np.isfinite(x) or not np.isfinite(y):
        return x, y
    
    x_min = -ROOM_MARGIN
    x_max = ROOM_WIDTH + ROOM_MARGIN
    y_min = -ROOM_MARGIN
    y_max = ROOM_HEIGHT + ROOM_MARGIN
    
    x = float(np.clip(x, x_min, x_max))
    y = float(np.clip(y, y_min, y_max))
    
    return x, y


def is_boundary_position(x: float, y: float) -> bool:
    """Check if position is near room boundary."""
    return (x < BOUNDARY_MARGIN or 
            x > ROOM_WIDTH - BOUNDARY_MARGIN or
            y < BOUNDARY_MARGIN or 
            y > ROOM_HEIGHT - BOUNDARY_MARGIN)


# =============================================================================
# ALGORITHM 1: WEIGHTED CENTROID LOCALIZATION (WCL)
# =============================================================================

def wcl_2d(distances: Dict[str, float], p: float = 2.0) -> Tuple[float, float]:
    """
    Weighted Centroid Localization.
    
    Position = Σ(w_i * pos_i) / Σ(w_i)
    where w_i = 1 / (distance_i ^ p)
    
    Args:
        distances: Dict of anchor_name -> distance
        p: Power factor for weighting (default 2.0)
    
    Returns:
        (x, y) estimated position
    """
    if len(distances) < MIN_ANCHORS:
        return np.nan, np.nan
    
    total_weight = 0.0
    x_weighted = 0.0
    y_weighted = 0.0
    
    for anchor_name, dist in distances.items():
        if anchor_name not in ANCHOR_POS:
            continue
        
        # Avoid division by zero
        dist = max(dist, 0.01)
        
        # Weight inversely proportional to distance^p
        weight = 1.0 / (dist ** p)
        
        ax, ay = ANCHOR_POS[anchor_name]
        x_weighted += weight * ax
        y_weighted += weight * ay
        total_weight += weight
    
    if total_weight < 1e-10:
        return np.nan, np.nan
    
    x = x_weighted / total_weight
    y = y_weighted / total_weight
    
    return float(x), float(y)


# =============================================================================
# ALGORITHM 2: LEAST SQUARES TRILATERATION
# =============================================================================

def trilateration_2d(distances: Dict[str, float]) -> Tuple[float, float]:
    """
    Least Squares Trilateration.
    
    Solves the system: A * [x, y]^T = b
    where the equations come from linearizing circle equations.
    
    Uses the last anchor as reference to eliminate quadratic terms.
    
    Args:
        distances: Dict of anchor_name -> distance
    
    Returns:
        (x, y) estimated position
    """
    if len(distances) < MIN_ANCHORS:
        return np.nan, np.nan
    
    # Get anchor names and ensure consistent ordering
    anchor_names = [name for name in distances.keys() if name in ANCHOR_POS]
    if len(anchor_names) < MIN_ANCHORS:
        return np.nan, np.nan
    
    # Use last anchor as reference
    n = len(anchor_names)
    ref_name = anchor_names[-1]
    x_n, y_n = ANCHOR_POS[ref_name]
    r_n = distances[ref_name]
    
    # Build A matrix and b vector
    A = []
    b = []
    
    for i in range(n - 1):
        name = anchor_names[i]
        x_i, y_i = ANCHOR_POS[name]
        r_i = distances[name]
        
        # Row of A: [2*(x_n - x_i), 2*(y_n - y_i)]
        A.append([2 * (x_n - x_i), 2 * (y_n - y_i)])
        
        # Element of b: r_i^2 - r_n^2 - x_i^2 - y_i^2 + x_n^2 + y_n^2
        b_i = (r_i**2 - r_n**2 - x_i**2 - y_i**2 + x_n**2 + y_n**2)
        b.append(b_i)
    
    A = np.array(A)
    b = np.array(b)
    
    try:
        # Solve using least squares: x = (A^T * A)^-1 * A^T * b
        ATA = A.T @ A
        ATb = A.T @ b
        
        # Check for singular matrix
        if np.linalg.det(ATA) < 1e-10:
            # Use pseudoinverse for ill-conditioned systems
            pos = np.linalg.lstsq(A, b, rcond=None)[0]
        else:
            pos = np.linalg.solve(ATA, ATb)
        
        return float(pos[0]), float(pos[1])
    
    except (np.linalg.LinAlgError, ValueError):
        return np.nan, np.nan


# =============================================================================
# ALGORITHM 3: BARYCENTRIC COORDINATES USING CLOSED POINTS (BCCP)
# =============================================================================

def circle_intersections(x1: float, y1: float, r1: float,
                         x2: float, y2: float, r2: float) -> List[Tuple[float, float]]:
    """
    Find intersection points of two circles.
    
    Circle 1: center (x1, y1), radius r1
    Circle 2: center (x2, y2), radius r2
    
    Returns:
        List of (x, y) intersection points (0, 1, or 2 points)
    """
    # Distance between centers
    d = math.hypot(x2 - x1, y2 - y1)
    
    # Check for no intersection or infinite intersections
    if d < 1e-10:  # Circles are concentric
        return []
    if d > r1 + r2:  # Circles too far apart
        return []
    if d < abs(r1 - r2):  # One circle inside the other
        return []
    
    # Calculate intersection points using geometric method
    a = (r1**2 - r2**2 + d**2) / (2 * d)
    
    # Check if a is valid
    if r1**2 - a**2 < 0:
        return []
    
    h = math.sqrt(r1**2 - a**2)
    
    # Point on line between centers at distance a from center 1
    px = x1 + a * (x2 - x1) / d
    py = y1 + a * (y2 - y1) / d
    
    # Two intersection points
    p1 = (px + h * (y2 - y1) / d, py - h * (x2 - x1) / d)
    p2 = (px - h * (y2 - y1) / d, py + h * (x2 - x1) / d)
    
    # If h is very small, circles are tangent (one intersection)
    if h < 1e-10:
        return [p1]
    
    return [p1, p2]


def find_closed_point(points1: List[Tuple[float, float]], 
                      points2: List[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
    """
    Find the closed point from two sets of intersection points.
    The closed point is the one from points1 that is closest to any point in points2.
    """
    if not points1 or not points2:
        return None
    
    min_dist = float('inf')
    best_point = None
    
    for p1 in points1:
        for p2 in points2:
            dist = math.hypot(p1[0] - p2[0], p1[1] - p2[1])
            if dist < min_dist:
                min_dist = dist
                best_point = p1
    
    return best_point


def bccp_2d(distances: Dict[str, float]) -> Tuple[float, float]:
    """
    Barycentric Coordinates using Closed Points (BCCP).
    
    Algorithm:
    1. For each pair of anchors, find circle intersection points
    2. Select the "closed point" from each pair (nearest to other intersections)
    3. Calculate centroid of closed points as final position
    
    Args:
        distances: Dict of anchor_name -> distance
    
    Returns:
        (x, y) estimated position
    """
    if len(distances) < MIN_ANCHORS:
        return np.nan, np.nan
    
    # Get anchor data
    anchor_names = [name for name in distances.keys() if name in ANCHOR_POS]
    if len(anchor_names) < MIN_ANCHORS:
        return np.nan, np.nan
    
    # Use only first 3-4 anchors for BCCP (as per original algorithm)
    anchor_names = anchor_names[:4]
    
    # Calculate all circle intersections
    all_intersections = {}
    for name1, name2 in combinations(anchor_names, 2):
        x1, y1 = ANCHOR_POS[name1]
        x2, y2 = ANCHOR_POS[name2]
        r1 = distances[name1]
        r2 = distances[name2]
        
        intersections = circle_intersections(x1, y1, r1, x2, y2, r2)
        if intersections:
            all_intersections[(name1, name2)] = intersections
    
    if len(all_intersections) < 2:
        # Fall back to WCL if not enough intersections
        return wcl_2d(distances)
    
    # Find closed points
    closed_points = []
    pairs = list(all_intersections.keys())
    
    for i, pair1 in enumerate(pairs):
        points1 = all_intersections[pair1]
        
        # Find the closest point in points1 to any point in other pairs
        min_total_dist = float('inf')
        best_point = None
        
        for p1 in points1:
            total_dist = 0
            count = 0
            for j, pair2 in enumerate(pairs):
                if i != j:
                    points2 = all_intersections[pair2]
                    # Find minimum distance to this pair
                    min_dist_to_pair = min(
                        math.hypot(p1[0] - p2[0], p1[1] - p2[1]) 
                        for p2 in points2
                    )
                    total_dist += min_dist_to_pair
                    count += 1
            
            if count > 0:
                avg_dist = total_dist / count
                if avg_dist < min_total_dist:
                    min_total_dist = avg_dist
                    best_point = p1
        
        if best_point:
            closed_points.append(best_point)
    
    if not closed_points:
        return wcl_2d(distances)
    
    # Calculate centroid of closed points (barycentric coordinate)
    x_sum = sum(p[0] for p in closed_points)
    y_sum = sum(p[1] for p in closed_points)
    n = len(closed_points)
    
    return float(x_sum / n), float(y_sum / n)


# =============================================================================
# FUSION (For reference only - not used in algorithm comparison)
# =============================================================================

def fused_position(pos_wcl: Tuple[float, float],
                   pos_tri: Tuple[float, float],
                   pos_bccp: Tuple[float, float]) -> Tuple[float, float]:
    """
    Calculate weighted fusion of three position estimates.
    Uses fixed weights - NOT adaptive (for fair comparison).
    """
    positions = [pos_wcl, pos_tri, pos_bccp]
    weights_list = [FUSION_WEIGHTS['wcl'], FUSION_WEIGHTS['tri'], FUSION_WEIGHTS['bccp']]
    
    x_sum = 0.0
    y_sum = 0.0
    w_sum = 0.0
    
    for pos, w in zip(positions, weights_list):
        if np.isfinite(pos[0]) and np.isfinite(pos[1]):
            x_sum += w * pos[0]
            y_sum += w * pos[1]
            w_sum += w
    
    if w_sum < 1e-10:
        return np.nan, np.nan
    
    return x_sum / w_sum, y_sum / w_sum


# =============================================================================
# ERROR CALCULATION
# =============================================================================

def calculate_error(estimated: Tuple[float, float], 
                    true_pos: Tuple[float, float]) -> float:
    """Calculate Euclidean distance error."""
    if not np.isfinite(estimated[0]) or not np.isfinite(estimated[1]):
        return np.nan
    return math.hypot(estimated[0] - true_pos[0], estimated[1] - true_pos[1])


# =============================================================================
# MQTT CALLBACKS
# =============================================================================

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print(f"[MQTT] Connected to broker at {BROKER_HOST}:{BROKER_PORT}")
        client.subscribe(TOPIC)
        print(f"[MQTT] Subscribed to topic: {TOPIC}")
    else:
        print(f"[MQTT] Connection failed with code {rc}")


def on_message(client, userdata, msg):
    """Process incoming MQTT message with RSSI data."""
    global frame_count, last_plot_time
    
    try:
        data = json.loads(msg.payload.decode())
    except json.JSONDecodeError:
        return
    
    anchor = data.get("anchor")
    if anchor not in anchors:
        return
    
    # Get RSSI value (prefer trimmed mean from ESP32 if available)
    rssi_raw = data.get("rssi_trim") or data.get("rssi")
    if rssi_raw is None:
        return
    
    ts = time.time()
    a = anchors[anchor]
    
    # Add to RSSI history
    a.rssi_hist.append(rssi_raw)
    a.last_update = ts
    
    # Filter RSSI using trimmed mean
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
    
    # =================================================================
    # CALCULATE RAW POSITIONS - NO ALGORITHM-SPECIFIC MODIFICATIONS
    # =================================================================
    
    # Algorithm 1: WCL
    x_wcl_raw, y_wcl_raw = wcl_2d(valid_anchors, p=2.0)
    x_wcl_raw, y_wcl_raw = clamp_to_room(x_wcl_raw, y_wcl_raw)
    
    # Algorithm 2: Trilateration
    x_tri_raw, y_tri_raw = trilateration_2d(valid_anchors)
    x_tri_raw, y_tri_raw = clamp_to_room(x_tri_raw, y_tri_raw)
    
    # Algorithm 3: BCCP
    x_bccp_raw, y_bccp_raw = bccp_2d(valid_anchors)
    x_bccp_raw, y_bccp_raw = clamp_to_room(x_bccp_raw, y_bccp_raw)
    
    # =================================================================
    # APPLY SAME SMOOTHING TO ALL (or no smoothing if disabled)
    # =================================================================
    
    x_wcl, y_wcl = smoother_wcl.update(x_wcl_raw, y_wcl_raw)
    x_tri, y_tri = smoother_tri.update(x_tri_raw, y_tri_raw)
    x_bccp, y_bccp = smoother_bccp.update(x_bccp_raw, y_bccp_raw)
    
    # Fused position (for reference)
    x_fused_raw, y_fused_raw = fused_position(
        (x_wcl, y_wcl), (x_tri, y_tri), (x_bccp, y_bccp)
    )
    x_fused, y_fused = smoother_fused.update(x_fused_raw, y_fused_raw)
    x_fused, y_fused = clamp_to_room(x_fused, y_fused)
    
    # =================================================================
    # COLLECT ERROR STATISTICS (if ground truth is set)
    # =================================================================
    
    if GROUND_TRUTH_POS is not None:
        is_boundary = is_boundary_position(GROUND_TRUTH_POS[0], GROUND_TRUTH_POS[1])
        
        err_wcl = calculate_error((x_wcl, y_wcl), GROUND_TRUTH_POS)
        err_tri = calculate_error((x_tri, y_tri), GROUND_TRUTH_POS)
        err_bccp = calculate_error((x_bccp, y_bccp), GROUND_TRUTH_POS)
        err_fused = calculate_error((x_fused, y_fused), GROUND_TRUTH_POS)
        
        if np.isfinite(err_wcl):
            stats_wcl.add_error(err_wcl, is_boundary)
        if np.isfinite(err_tri):
            stats_tri.add_error(err_tri, is_boundary)
        if np.isfinite(err_bccp):
            stats_bccp.add_error(err_bccp, is_boundary)
        if np.isfinite(err_fused):
            stats_fused.add_error(err_fused, is_boundary)
    
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
# STATISTICS PRINTING
# =============================================================================

def print_statistics():
    """Print error statistics for algorithm comparison."""
    print("\n" + "=" * 70)
    print("ALGORITHM COMPARISON STATISTICS")
    print("=" * 70)
    
    if GROUND_TRUTH_POS is None:
        print("Ground truth not set. Set GROUND_TRUTH_POS to enable error tracking.")
        return
    
    print(f"\nGround Truth Position: ({GROUND_TRUTH_POS[0]:.2f}, {GROUND_TRUTH_POS[1]:.2f})")
    is_boundary = is_boundary_position(GROUND_TRUTH_POS[0], GROUND_TRUTH_POS[1])
    print(f"Position Type: {'BOUNDARY' if is_boundary else 'CENTER'}")
    print(f"Total Samples: {len(stats_wcl.all_errors)}")
    
    print("\n" + "-" * 70)
    print(f"{'Algorithm':<15} {'RMSE (m)':<12} {'Mean Error (m)':<15} {'Samples':<10}")
    print("-" * 70)
    
    algorithms = [
        ("WCL", stats_wcl),
        ("Trilateration", stats_tri),
        ("BCCP", stats_bccp),
        ("FUSED", stats_fused),
    ]
    
    for name, stats in algorithms:
        rmse = stats.get_rmse('all')
        mean = stats.get_mean('all')
        samples = len(stats.all_errors)
        print(f"{name:<15} {rmse:<12.3f} {mean:<15.3f} {samples:<10}")
    
    print("-" * 70)
    
    # Boundary vs Center comparison
    if len(stats_wcl.center_errors) > 0 or len(stats_wcl.boundary_errors) > 0:
        print("\nREGION BREAKDOWN:")
        print("-" * 70)
        print(f"{'Algorithm':<15} {'Center RMSE':<15} {'Boundary RMSE':<15} {'Degradation':<12}")
        print("-" * 70)
        
        for name, stats in algorithms:
            center_rmse = stats.get_rmse('center')
            boundary_rmse = stats.get_rmse('boundary')
            if center_rmse > 0:
                degradation = ((boundary_rmse - center_rmse) / center_rmse) * 100
                print(f"{name:<15} {center_rmse:<15.3f} {boundary_rmse:<15.3f} {degradation:+.1f}%")
            else:
                print(f"{name:<15} {center_rmse:<15.3f} {boundary_rmse:<15.3f} {'N/A':<12}")
        
        print("-" * 70)
    
    print("=" * 70)


# =============================================================================
# MAIN VISUALIZATION
# =============================================================================

def main():
    """Main function with real-time visualization."""
    
    # Connect to MQTT
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    
    try:
        client.connect(BROKER_HOST, BROKER_PORT)
    except Exception as e:
        print(f"[ERROR] Cannot connect to MQTT broker: {e}")
        print(f"[ERROR] Make sure broker is running at {BROKER_HOST}:{BROKER_PORT}")
        return
    
    client.loop_start()
    
    # Setup plot
    plt.ion()
    fig, ax = plt.subplots(figsize=(12, 9))
    
    # Set up plot bounds
    margin = 1.0
    ax.set_xlim(-margin, ROOM_WIDTH + margin)
    ax.set_ylim(-margin, ROOM_HEIGHT + margin)
    ax.set_aspect('equal')
    ax.set_xlabel("X (meters)", fontsize=12)
    ax.set_ylabel("Y (meters)", fontsize=12)
    ax.set_title("2D Indoor Positioning - Fair Algorithm Comparison", fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Draw room boundary
    room_rect = plt.Rectangle((0, 0), ROOM_WIDTH, ROOM_HEIGHT, 
                               fill=False, edgecolor='gray', linewidth=2, linestyle='-')
    ax.add_patch(room_rect)
    
    # Draw boundary region (for reference)
    boundary_rect = plt.Rectangle((BOUNDARY_MARGIN, BOUNDARY_MARGIN), 
                                   ROOM_WIDTH - 2*BOUNDARY_MARGIN, 
                                   ROOM_HEIGHT - 2*BOUNDARY_MARGIN,
                                   fill=False, edgecolor='lightgreen', 
                                   linewidth=1, linestyle='--', alpha=0.5)
    ax.add_patch(boundary_rect)
    ax.text(ROOM_WIDTH/2, ROOM_HEIGHT/2, 'CENTER\nREGION', 
            ha='center', va='center', fontsize=8, alpha=0.3, color='green')
    
    # Plot anchors
    for name, (x, y) in ANCHOR_POS.items():
        ax.plot(x, y, 's', color='black', markersize=15, zorder=10)
        ax.annotate(name, (x, y), textcoords="offset points",
                   xytext=(0, 12), ha='center', fontweight='bold', fontsize=11)
    
    # Plot ground truth if set
    if GROUND_TRUTH_POS is not None:
        ax.plot(GROUND_TRUTH_POS[0], GROUND_TRUTH_POS[1], 'X', 
                color='green', markersize=20, label='Ground Truth', 
                markeredgecolor='darkgreen', markeredgewidth=2, zorder=15)
    
    # Distance circles
    circles = {}
    for name, (x, y) in ANCHOR_POS.items():
        circle = plt.Circle((x, y), 0.1, fill=False, linestyle='--', alpha=0.3, color='gray')
        ax.add_patch(circle)
        circles[name] = circle
    
    # Position markers - same size for fair visual comparison
    dot_wcl, = ax.plot([np.nan], [np.nan], 'o', color='blue', 
                       markersize=12, label='WCL', alpha=0.8)
    dot_tri, = ax.plot([np.nan], [np.nan], '^', color='orange', 
                       markersize=12, label='Trilateration', alpha=0.8)
    dot_bccp, = ax.plot([np.nan], [np.nan], 'D', color='purple', 
                        markersize=12, label='BCCP', alpha=0.8)
    dot_fused, = ax.plot([np.nan], [np.nan], '*', color='red', 
                         markersize=16, label='Fused', alpha=0.9)
    
    # Legend
    handles = [dot_wcl, dot_tri, dot_bccp, dot_fused]
    if GROUND_TRUTH_POS is not None:
        gt_marker = plt.Line2D([0], [0], marker='X', color='w', 
                               markerfacecolor='green', markersize=12, label='Ground Truth')
        handles.append(gt_marker)
    ax.legend(handles=handles, loc='upper right', title='Methods', fontsize=10)
    
    # Info text box
    info = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=9,
                   va='top', family='monospace',
                   bbox=dict(facecolor='wheat', alpha=0.85, edgecolor='gray'))
    
    # Settings info
    settings_text = f"Smoothing: {'ON (α={})'.format(SMOOTHING_ALPHA) if ENABLE_SMOOTHING else 'OFF'}"
    ax.text(0.98, 0.02, settings_text, transform=ax.transAxes, fontsize=8,
            ha='right', va='bottom', alpha=0.7)
    
    fig.tight_layout()
    
    # Print startup info
    print("\n" + "=" * 70)
    print("INDOOR POSITIONING SYSTEM - FAIR ALGORITHM COMPARISON")
    print("=" * 70)
    print(f"Anchors: {list(ANCHOR_POS.keys())}")
    print(f"RSSI Window: {RSSI_WINDOW_SIZE} samples")
    print(f"Smoothing: {'ENABLED (α={})'.format(SMOOTHING_ALPHA) if ENABLE_SMOOTHING else 'DISABLED'}")
    print(f"Ground Truth: {GROUND_TRUTH_POS if GROUND_TRUTH_POS else 'Not set'}")
    print("=" * 70)
    print("Press Ctrl+C to stop and print statistics")
    print("=" * 70 + "\n")
    
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
                    # Update position markers
                    dot_wcl.set_xdata([result["wcl"][0]])
                    dot_wcl.set_ydata([result["wcl"][1]])
                    
                    dot_tri.set_xdata([result["tri"][0]])
                    dot_tri.set_ydata([result["tri"][1]])
                    
                    dot_bccp.set_xdata([result["bccp"][0]])
                    dot_bccp.set_ydata([result["bccp"][1]])
                    
                    dot_fused.set_xdata([result["fused"][0]])
                    dot_fused.set_ydata([result["fused"][1]])
                    
                    # Update distance circles
                    for name, dist in result["distances"].items():
                        if name in circles:
                            circles[name].set_radius(dist)
                    
                    # Build info text
                    lines = ["━" * 24]
                    lines.append("    DISTANCES")
                    lines.append("━" * 24)
                    for name, dist in sorted(result["distances"].items()):
                        rssi = anchors[name].filtered_rssi
                        rssi_str = f"({rssi:.0f}dBm)" if rssi else ""
                        lines.append(f"  {name}: {dist:.2f}m {rssi_str}")
                    
                    lines.append("━" * 24)
                    lines.append("    POSITIONS")
                    lines.append("━" * 24)
                    lines.append(f"  WCL:   ({result['wcl'][0]:.2f}, {result['wcl'][1]:.2f})")
                    lines.append(f"  TRI:   ({result['tri'][0]:.2f}, {result['tri'][1]:.2f})")
                    lines.append(f"  BCCP:  ({result['bccp'][0]:.2f}, {result['bccp'][1]:.2f})")
                    lines.append(f"  FUSED: ({result['fused'][0]:.2f}, {result['fused'][1]:.2f})")
                    
                    if GROUND_TRUTH_POS is not None:
                        lines.append("━" * 24)
                        lines.append("    ERRORS (m)")
                        lines.append("━" * 24)
                        err_wcl = calculate_error(result['wcl'], GROUND_TRUTH_POS)
                        err_tri = calculate_error(result['tri'], GROUND_TRUTH_POS)
                        err_bccp = calculate_error(result['bccp'], GROUND_TRUTH_POS)
                        err_fused = calculate_error(result['fused'], GROUND_TRUTH_POS)
                        lines.append(f"  WCL:   {err_wcl:.3f}")
                        lines.append(f"  TRI:   {err_tri:.3f}")
                        lines.append(f"  BCCP:  {err_bccp:.3f}")
                        lines.append(f"  FUSED: {err_fused:.3f}")
                    
                    lines.append("━" * 24)
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
        # Print final statistics
        print_statistics()
        
        client.loop_stop()
        client.disconnect()
        plt.close('all')
        print("[MAIN] Done")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # =================================================================
    # SET GROUND TRUTH HERE FOR ERROR MEASUREMENT
    # =================================================================
    # Uncomment and set to your actual test position:
    # GROUND_TRUTH_POS = (2.1, 2.0)  # Example: center of room
    # GROUND_TRUTH_POS = (0.5, 0.5)  # Example: near corner (boundary)
    
    main()