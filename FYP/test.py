"""
2D Indoor Positioning Tracker
Requires 3+ anchors for 2D positioning.
"""
import json
import time
from collections import deque
from queue import Queue, Empty
from itertools import combinations

import numpy as np
from scipy.optimize import minimize
import paho.mqtt.client as mqtt
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
BROKER_HOST = "127.0.0.1"
BROKER_PORT = 1883
TOPIC = "ips/rssi"

# Anchor positions (x, y) in meters
ANCHOR_POS = {
    "A1": (0, 0),
    "A2": (5, 0),
    "A3": (5, 5),
    "A4": (0, 5),
}

# Calibration per anchor
CALIB = {
    "A1": {"rssi1m": -74.69, "n": 1.813},
    "A2": {"rssi1m": -72.86, "n": 2.334},
    "A3": {"rssi1m": -75.86, "n": 2.522},
    "A4": {"rssi1m": -73.55, "n": 2.478},
}

RSSI_SMOOTH_N = 5
D_MIN, D_MAX = 0.1, 20.0
MAX_AGE = 2.0
MIN_ANCHORS = 3

# =========================
# STABILITY CONFIG
# =========================
POS_ALPHA = 0.2  # EMA smoothing factor (lower = smoother, 0.1-0.3 recommended)
MAX_JUMP_DISTANCE = 1.5  # Max allowed jump in meters per update (outlier rejection)
USE_KALMAN = True  # Use Kalman filter for extra smoothing
KALMAN_PROCESS_NOISE = 0.05  # How much we expect position to change
KALMAN_MEASUREMENT_NOISE = 0.5  # How noisy measurements are

# =========================
# ROOM BOUNDARY CONFIG
# =========================
ENABLE_BOUNDARY_CLAMP = True  # Toggle boundary clamping on/off
ROOM_BOUNDS = {
    "x_min": -0.5,
    "x_max": 5.5,
    "y_min": -0.5,
    "y_max": 5.5,
}

# =========================
# STATE - Per anchor
# =========================
class AnchorData:
    def __init__(self):
        self.rssi_hist = deque(maxlen=RSSI_SMOOTH_N)
        self.last_update = 0
        self.distance = None
        self.last_mac = None

anchors = {name: AnchorData() for name in ANCHOR_POS}

frame_count = 0
last_plot_time = 0
position_queue = Queue(maxsize=1)

# =========================
# POSITION SMOOTHER CLASS
# =========================
class PositionSmoother:
    """Smooths position estimates using EMA and outlier rejection."""
    
    def __init__(self, alpha=0.2, max_jump=1.5):
        self.alpha = alpha
        self.max_jump = max_jump
        self.last_pos = None
        self.history = deque(maxlen=10)  # Keep history for median filtering
    
    def update(self, x, y):
        """Update with new position, returns smoothed position."""
        if not np.isfinite(x) or not np.isfinite(y):
            return self.last_pos if self.last_pos else (np.nan, np.nan)
        
        new_pos = np.array([x, y])
        
        # First position - just accept it
        if self.last_pos is None:
            self.last_pos = new_pos
            self.history.append(new_pos)
            return (x, y)
        
        # Outlier rejection - if jump is too large, use median of recent history
        distance = np.linalg.norm(new_pos - self.last_pos)
        if distance > self.max_jump and len(self.history) >= 3:
            # Use median of history instead of the outlier
            hist_array = np.array(self.history)
            median_pos = np.median(hist_array, axis=0)
            # Blend slightly toward new position in case it's a real movement
            new_pos = median_pos + 0.1 * (new_pos - median_pos)
        
        # EMA smoothing
        smoothed = self.alpha * new_pos + (1 - self.alpha) * self.last_pos
        
        self.last_pos = smoothed
        self.history.append(smoothed)
        
        return (float(smoothed[0]), float(smoothed[1]))
    
    def reset(self):
        """Reset smoother state."""
        self.last_pos = None
        self.history.clear()


class KalmanFilter2D:
    """Simple 2D Kalman filter for position tracking."""
    
    def __init__(self, process_noise=0.05, measurement_noise=0.5):
        # State: [x, y, vx, vy]
        self.state = None
        self.P = None  # Covariance matrix
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.last_time = None
        
    def predict(self, dt):
        """Predict next state based on velocity."""
        if self.state is None:
            return
        
        # State transition matrix (constant velocity model)
        F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # Process noise
        Q = self.process_noise * np.array([
            [dt**4/4, 0, dt**3/2, 0],
            [0, dt**4/4, 0, dt**3/2],
            [dt**3/2, 0, dt**2, 0],
            [0, dt**3/2, 0, dt**2]
        ])
        
        self.state = F @ self.state
        self.P = F @ self.P @ F.T + Q
    
    def update(self, x, y):
        """Update state with measurement."""
        if not np.isfinite(x) or not np.isfinite(y):
            if self.state is not None:
                return (float(self.state[0]), float(self.state[1]))
            return (np.nan, np.nan)
        
        current_time = time.time()
        
        # Initialize on first measurement
        if self.state is None:
            self.state = np.array([x, y, 0.0, 0.0])
            self.P = np.eye(4) * 1.0
            self.last_time = current_time
            return (x, y)
        
        # Predict step
        dt = current_time - self.last_time
        dt = max(0.01, min(dt, 1.0))  # Clamp dt to reasonable range
        self.predict(dt)
        self.last_time = current_time
        
        # Measurement matrix (we only measure position, not velocity)
        H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        # Measurement noise
        R = np.eye(2) * self.measurement_noise
        
        # Kalman gain
        z = np.array([x, y])
        y_residual = z - H @ self.state
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # Update state
        self.state = self.state + K @ y_residual
        self.P = (np.eye(4) - K @ H) @ self.P
        
        return (float(self.state[0]), float(self.state[1]))
    
    def reset(self):
        """Reset filter state."""
        self.state = None
        self.P = None
        self.last_time = None


# Create smoothers for each method
smoothers = {
    "wcl": PositionSmoother(alpha=POS_ALPHA, max_jump=MAX_JUMP_DISTANCE),
    "tri": PositionSmoother(alpha=POS_ALPHA, max_jump=MAX_JUMP_DISTANCE),
    "bccp": PositionSmoother(alpha=POS_ALPHA, max_jump=MAX_JUMP_DISTANCE),
}

# Create Kalman filters for each method (optional)
kalman_filters = {
    "wcl": KalmanFilter2D(KALMAN_PROCESS_NOISE, KALMAN_MEASUREMENT_NOISE),
    "tri": KalmanFilter2D(KALMAN_PROCESS_NOISE, KALMAN_MEASUREMENT_NOISE),
    "bccp": KalmanFilter2D(KALMAN_PROCESS_NOISE, KALMAN_MEASUREMENT_NOISE),
}

# =========================
# BOUNDARY FUNCTIONS
# =========================
def clamp_to_room(x, y):
    """Clamp coordinates to room boundaries if enabled."""
    if not ENABLE_BOUNDARY_CLAMP:
        return x, y
    
    if not np.isfinite(x) or not np.isfinite(y):
        return x, y
    
    x_clamped = np.clip(x, ROOM_BOUNDS["x_min"], ROOM_BOUNDS["x_max"])
    y_clamped = np.clip(y, ROOM_BOUNDS["y_min"], ROOM_BOUNDS["y_max"])
    
    return float(x_clamped), float(y_clamped)


def set_room_bounds(x_min, y_min, x_max, y_max):
    """Update room boundaries."""
    global ROOM_BOUNDS
    ROOM_BOUNDS = {
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
    }


def toggle_boundary_clamp(enabled=None):
    """Toggle or set boundary clamping."""
    global ENABLE_BOUNDARY_CLAMP
    if enabled is None:
        ENABLE_BOUNDARY_CLAMP = not ENABLE_BOUNDARY_CLAMP
    else:
        ENABLE_BOUNDARY_CLAMP = enabled
    return ENABLE_BOUNDARY_CLAMP


# =========================
# POSITIONING FUNCTIONS
# =========================
def rssi_to_distance(rssi, rssi1m, n):
    """Convert RSSI to distance using log-distance path loss model."""
    d = 10 ** ((rssi1m - rssi) / (10.0 * n))
    return float(np.clip(d, D_MIN, D_MAX))


def wcl_2d(distances: dict, p: float = 2.0):
    """Weighted Centroid Localization in 2D."""
    num_x, num_y, den = 0.0, 0.0, 0.0
    
    for anchor, dist in distances.items():
        if anchor not in ANCHOR_POS:
            continue
        x, y = ANCHOR_POS[anchor]
        w = 1.0 / (dist ** p + 1e-6)
        num_x += w * x
        num_y += w * y
        den += w
    
    if den <= 0:
        return np.nan, np.nan
    
    return num_x / den, num_y / den


def trilateration_2d(distances: dict):
    """Trilateration using linearized least squares."""
    anchor_list = [a for a in distances if a in ANCHOR_POS]
    if len(anchor_list) < 3:
        return np.nan, np.nan
    
    ref = anchor_list[0]
    x0, y0 = ANCHOR_POS[ref]
    d0 = distances[ref]
    
    A = []
    b = []
    
    for anchor in anchor_list[1:]:
        xi, yi = ANCHOR_POS[anchor]
        di = distances[anchor]
        A.append([2 * (xi - x0), 2 * (yi - y0)])
        b.append(d0**2 - di**2 + xi**2 + yi**2 - x0**2 - y0**2)
    
    A = np.array(A)
    b = np.array(b)
    
    try:
        result, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        return float(result[0]), float(result[1])
    except np.linalg.LinAlgError:
        return np.nan, np.nan


# =========================
# BCCP FUNCTIONS
# =========================
def _circ_ints(C1, r1, C2, r2, tol=1e-8):
    """Return up to two circle intersection points. Non-intersection -> NaNs."""
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


def _inflate_for_overlap(AP, D0, alpha=0.05, eps=1e-6, max_iter=12):
    """Inflate radii so all pairs overlap robustly."""
    AP = np.asarray(AP, float)
    D0 = np.asarray(D0, float)
    pairs = [(0, 1), (0, 2), (1, 2)]

    s = 1.0
    for i, j in pairs:
        dij = np.linalg.norm(AP[i] - AP[j])
        den = D0[i] + D0[j]
        if den > 0:
            needed = ((1.0 + alpha) * dij) / den
            s = max(s, min(needed, 1.5))  # Cap scaling at 1.5x

    D = _project_radii_to_intersect(AP, D0 * s, eps=eps, iters=20)
    return D  # Simplified - don't iterate further


def _bccp_once_3anchors(anchor_names, anchor_pos, distances):
    """One-shot BCCP with exactly 3 anchors. Returns (x,y) or (nan,nan)."""
    AP = np.array([anchor_pos[a] for a in anchor_names], float)
    D0 = np.array([max(float(distances[a]), 1e-9) for a in anchor_names], float)

    # Use smaller inflation to preserve geometry
    D = _project_radii_to_intersect(AP, D0, eps=1e-6, iters=20)
    D = _inflate_for_overlap(AP, D, alpha=0.05)  # Reduced from 0.20 to 0.05

    E, F = _circ_ints(AP[0], D[0], AP[1], D[1])
    G, H = _circ_ints(AP[0], D[0], AP[2], D[2])
    I, J = _circ_ints(AP[1], D[1], AP[2], D[2])

    CP12 = _pick_cp(E, F, AP[2], D[2])
    CP13 = _pick_cp(G, H, AP[1], D[1])
    CP23 = _pick_cp(I, J, AP[0], D[0])

    CPs = np.vstack([CP12, CP13, CP23])
    CPs = CPs[np.all(np.isfinite(CPs), axis=1)]
    if CPs.shape[0] < 2:  # Allow 2 points minimum
        return float("nan"), float("nan")

    p = np.mean(CPs, axis=0)
    return float(p[0]), float(p[1])


def _residual_sum(p, anchor_list, anchor_pos, distances):
    """Sum of squared range residuals against provided anchors."""
    x, y = p
    s = 0.0
    for a in anchor_list:
        ax, ay = anchor_pos[a]
        d_meas = float(distances[a])
        d_act = np.hypot(x - ax, y - ay)
        s += (d_act - d_meas) ** 2
    return s


def bccp_2d(distances: dict, clamp_bounds=None):
    """BCCP (Best Closed Point) for 3 or 4 anchors with improved robustness."""
    anchors_list = [a for a in distances if a in ANCHOR_POS and np.isfinite(distances[a])]
    if len(anchors_list) < 3:
        return float("nan"), float("nan")

    # Get anchor bounding box for sanity checking
    ax_coords = [ANCHOR_POS[a][0] for a in anchors_list]
    ay_coords = [ANCHOR_POS[a][1] for a in anchors_list]
    margin = 3.0  # Allow some margin outside anchors
    x_min, x_max = min(ax_coords) - margin, max(ax_coords) + margin
    y_min, y_max = min(ay_coords) - margin, max(ay_coords) + margin

    candidates = []
    for trip in combinations(anchors_list, 3):
        x, y = _bccp_once_3anchors(trip, ANCHOR_POS, distances)
        if not np.isfinite(x) or not np.isfinite(y):
            continue
        
        # Reject points far outside anchor region
        if x < x_min or x > x_max or y < y_min or y > y_max:
            continue

        score = _residual_sum((x, y), anchors_list, ANCHOR_POS, distances)
        candidates.append((score, (x, y), trip))

    if not candidates:
        return float("nan"), float("nan")

    # If we have multiple candidates, use weighted average of best ones
    candidates.sort(key=lambda c: c[0])
    
    if len(candidates) == 1:
        best = candidates[0]
    else:
        # Use top 2 candidates weighted by inverse score
        top_n = min(2, len(candidates))
        weights = [1.0 / (c[0] + 1e-6) for c in candidates[:top_n]]
        total_w = sum(weights)
        
        x = sum(w * c[1][0] for w, c in zip(weights, candidates[:top_n])) / total_w
        y = sum(w * c[1][1] for w, c in zip(weights, candidates[:top_n])) / total_w
        best = (0, (x, y), None)

    x, y = best[1]
    
    # Apply room bounds if specified
    if clamp_bounds is not None:
        (xmin, ymin), (xmax, ymax) = clamp_bounds
        x = min(max(x, xmin), xmax)
        y = min(max(y, ymin), ymax)

    return float(x), float(y)


def smooth_position(method, x, y):
    """Apply smoothing and boundary clamping to a position estimate."""
    # First apply EMA smoothing with outlier rejection
    x_smooth, y_smooth = smoothers[method].update(x, y)
    
    # Optionally apply Kalman filter for extra smoothing
    if USE_KALMAN:
        x_smooth, y_smooth = kalman_filters[method].update(x_smooth, y_smooth)
    
    # Apply room boundary clamping
    x_final, y_final = clamp_to_room(x_smooth, y_smooth)
    
    return x_final, y_final


# =========================
# MQTT
# =========================
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
    
    a = anchors[anchor]
    a.rssi_hist.append(rssi)
    a.last_update = ts
    a.last_mac = mac
    
    rssi_avg = np.mean(a.rssi_hist)
    c = CALIB[anchor]
    a.distance = rssi_to_distance(rssi_avg, c["rssi1m"], c["n"])
    
    valid_anchors = {}
    for name, anchor_data in anchors.items():
        if anchor_data.distance is None:
            continue
        age = ts - anchor_data.last_update
        if age <= MAX_AGE:
            valid_anchors[name] = anchor_data.distance
    
    if len(valid_anchors) < MIN_ANCHORS:
        return
    
    if ts - last_plot_time < 0.1:
        return
    
    last_plot_time = ts
    frame_count += 1
    
    # Calculate raw positions
    x_wcl_raw, y_wcl_raw = wcl_2d(valid_anchors, p=2.0)
    x_tri_raw, y_tri_raw = trilateration_2d(valid_anchors)
    x_bccp_raw, y_bccp_raw = bccp_2d(valid_anchors)
    
    # Apply smoothing and boundary clamping
    x_wcl, y_wcl = smooth_position("wcl", x_wcl_raw, y_wcl_raw)
    x_tri, y_tri = smooth_position("tri", x_tri_raw, y_tri_raw)
    x_bccp, y_bccp = smooth_position("bccp", x_bccp_raw, y_bccp_raw)
    
    print(f"[{frame_count:04d}] Anchors: {len(valid_anchors)} | "
          f"WCL=({x_wcl:.2f},{y_wcl:.2f}) "
          f"TRI=({x_tri:.2f},{y_tri:.2f}) "
          f"BCCP=({x_bccp:.2f},{y_bccp:.2f}) "
          f"[Bounds: {'ON' if ENABLE_BOUNDARY_CLAMP else 'OFF'}]")
    
    position_queue.put({
        "distances": valid_anchors,
        "wcl": (x_wcl, y_wcl),
        "tri": (x_tri, y_tri),
        "bccp": (x_bccp, y_bccp),
        "wcl_raw": (x_wcl_raw, y_wcl_raw),
        "tri_raw": (x_tri_raw, y_tri_raw),
        "bccp_raw": (x_bccp_raw, y_bccp_raw),
    })


# =========================
# MAIN
# =========================
def main():
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(BROKER_HOST, BROKER_PORT)
    client.loop_start()
    
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 8))
    
    xs = [p[0] for p in ANCHOR_POS.values()]
    ys = [p[1] for p in ANCHOR_POS.values()]
    margin = 1.0
    ax.set_xlim(min(xs) - margin, max(xs) + margin)
    ax.set_ylim(min(ys) - margin, max(ys) + margin)
    ax.set_aspect('equal')
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    ax.set_title("2D Indoor Positioning (Stabilized with BCCP)")
    ax.grid(True, alpha=0.3)
    
    # Draw room boundary if enabled
    if ENABLE_BOUNDARY_CLAMP:
        boundary_rect = plt.Rectangle(
            (ROOM_BOUNDS["x_min"], ROOM_BOUNDS["y_min"]),
            ROOM_BOUNDS["x_max"] - ROOM_BOUNDS["x_min"],
            ROOM_BOUNDS["y_max"] - ROOM_BOUNDS["y_min"],
            fill=False, edgecolor='red', linestyle='-', linewidth=2, alpha=0.7
        )
        ax.add_patch(boundary_rect)
    
    for name, (x, y) in ANCHOR_POS.items():
        ax.plot(x, y, 's', color='black', markersize=15, zorder=10)
        ax.annotate(name, (x, y), textcoords="offset points",
                   xytext=(0, 12), ha='center', fontweight='bold', fontsize=11)
    
    circles = {}
    for name, (x, y) in ANCHOR_POS.items():
        circle = plt.Circle((x, y), 0.1, fill=False, linestyle='--', alpha=0.4, color='gray')
        ax.add_patch(circle)
        circles[name] = circle
    
    # Position markers for each method
    dot_wcl, = ax.plot([np.nan], [np.nan], 'o', color='blue', 
                       markersize=14, label='WCL (p=2)', alpha=0.8)
    dot_tri, = ax.plot([np.nan], [np.nan], '^', color='orange', 
                       markersize=12, label='Trilateration', alpha=0.8)
    dot_bccp, = ax.plot([np.nan], [np.nan], 'p', color='purple', 
                        markersize=14, label='BCCP', alpha=0.8)
    
    ax.legend(handles=[dot_wcl, dot_tri, dot_bccp], 
              loc='upper right', title='Methods')
    
    info = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=9,
                   va='top', family='monospace',
                   bbox=dict(facecolor='wheat', alpha=0.8))
    
    fig.tight_layout()
    
    print("[MAIN] Running... Close window to stop")
    print(f"[MAIN] Using {len(ANCHOR_POS)} anchors: {list(ANCHOR_POS.keys())}")
    print(f"[MAIN] Boundary clamping: {'ENABLED' if ENABLE_BOUNDARY_CLAMP else 'DISABLED'}")
    print(f"[MAIN] Kalman filter: {'ENABLED' if USE_KALMAN else 'DISABLED'}")
    print("[MAIN] Press 'b' to toggle boundary clamping")
    
    running = True
    
    def on_close(event):
        nonlocal running
        running = False
    
    def on_key(event):
        if event.key == 'b':
            state = toggle_boundary_clamp()
            print(f"[MAIN] Boundary clamping: {'ENABLED' if state else 'DISABLED'}")
    
    fig.canvas.mpl_connect('close_event', on_close)
    fig.canvas.mpl_connect('key_press_event', on_key)
    
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
                    
                    for name, dist in result["distances"].items():
                        if name in circles:
                            circles[name].set_radius(dist)
                    
                    lines = ["Distances:"]
                    for name, dist in sorted(result["distances"].items()):
                        lines.append(f"  {name}: {dist:.2f}m")
                    lines.append("─" * 20)
                    lines.append("Smoothed Positions:")
                    lines.append(f"  WCL:  ({result['wcl'][0]:.2f}, {result['wcl'][1]:.2f})")
                    lines.append(f"  TRI:  ({result['tri'][0]:.2f}, {result['tri'][1]:.2f})")
                    lines.append(f"  BCCP: ({result['bccp'][0]:.2f}, {result['bccp'][1]:.2f})")
                    lines.append("─" * 20)
                    lines.append(f"Bounds: {'ON' if ENABLE_BOUNDARY_CLAMP else 'OFF'}")
                    lines.append(f"Kalman: {'ON' if USE_KALMAN else 'OFF'}")
                    
                    info.set_text('\n'.join(lines))
                    
                    fig.canvas.draw_idle()
                    fig.canvas.flush_events()
                    
            except Exception as e:
                print(f"Plot error: {e}")
            
            try:
                plt.pause(0.05)
            except:
                break
            
    except KeyboardInterrupt:
        pass
    finally:
        client.loop_stop()
        client.disconnect()
        plt.close('all')
        print("[MAIN] Done")


if __name__ == "__main__":
    main()