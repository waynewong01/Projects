"""
2D Indoor Positioning Tracker
Requires 3+ anchors for 2D positioning.
"""
import json
import time
from collections import deque
from queue import Queue, Empty

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
# Example: 3 anchors in a triangle
ANCHOR_POS = {
    "A1": (0, 0),
    "A2": (0, 4.17),
    "A3": (4.2, 0),
    "A4": (4.2, 4.17),
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
MAX_AGE = 2.0  # Max age of reading in seconds
MIN_ANCHORS = 3  # Minimum anchors needed for 2D position

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
# POSITIONING FUNCTIONS
# =========================
def rssi_to_distance(rssi, rssi1m, n):
    """Convert RSSI to distance using log-distance path loss model."""
    d = 10 ** ((rssi1m - rssi) / (10.0 * n))
    return float(np.clip(d, D_MIN, D_MAX))


def wcl_2d(distances: dict, p: float = 2.0):
    """
    Weighted Centroid Localization in 2D.
    
    Formula: position = Σ(w_i * pos_i) / Σ(w_i)
    where w_i = 1 / (distance_i ^ p)
    """
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


def ls_2d_grid(distances: dict, step: float = 0.05):
    """
    Least Squares 2D via grid search.
    
    Finds position that minimizes Σ(|measured_dist - actual_dist|²)
    """
    # Get bounds from anchor positions
    xs_anchors = [p[0] for p in ANCHOR_POS.values()]
    ys_anchors = [p[1] for p in ANCHOR_POS.values()]
    
    x_min, x_max = min(xs_anchors) - 0.5, max(xs_anchors) + 0.5
    y_min, y_max = min(ys_anchors) - 0.5, max(ys_anchors) + 0.5
    
    xs = np.arange(x_min, x_max + step, step)
    ys = np.arange(y_min, y_max + step, step)
    X, Y = np.meshgrid(xs, ys)
    
    error = np.zeros_like(X)
    
    for anchor, dist in distances.items():
        if anchor not in ANCHOR_POS:
            continue
        ax, ay = ANCHOR_POS[anchor]
        actual_dist = np.sqrt((X - ax)**2 + (Y - ay)**2)
        error += (actual_dist - dist) ** 2
    
    min_idx = np.unravel_index(np.argmin(error), error.shape)
    return float(X[min_idx]), float(Y[min_idx])


def trilateration_2d(distances: dict):
    """
    Trilateration using linearized least squares.
    
    For 3+ anchors, solves the system of equations:
    (x - x_i)² + (y - y_i)² = d_i²
    """
    anchor_list = [a for a in distances if a in ANCHOR_POS]
    if len(anchor_list) < 3:
        return np.nan, np.nan
    
    # Use first anchor as reference
    ref = anchor_list[0]
    x0, y0 = ANCHOR_POS[ref]
    d0 = distances[ref]
    
    # Build linear system: Ax = b
    A = []
    b = []
    
    for anchor in anchor_list[1:]:
        xi, yi = ANCHOR_POS[anchor]
        di = distances[anchor]
        
        # Linearized equation
        A.append([2 * (xi - x0), 2 * (yi - y0)])
        b.append(d0**2 - di**2 + xi**2 + yi**2 - x0**2 - y0**2)
    
    A = np.array(A)
    b = np.array(b)
    
    try:
        result, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        return float(result[0]), float(result[1])
    except np.linalg.LinAlgError:
        return np.nan, np.nan


def multilateration_nlls(distances: dict):
    """
    Non-Linear Least Squares multilateration.
    
    Most accurate method - uses optimization to find best fit.
    """
    anchor_list = [a for a in distances if a in ANCHOR_POS]
    if len(anchor_list) < 3:
        return np.nan, np.nan
    
    # Initial guess from WCL
    x0, y0 = wcl_2d(distances, p=2.0)
    if np.isnan(x0):
        positions = [ANCHOR_POS[a] for a in anchor_list]
        x0 = np.mean([p[0] for p in positions])
        y0 = np.mean([p[1] for p in positions])
    
    def objective(pos):
        x, y = pos
        error = 0.0
        for anchor in anchor_list:
            ax, ay = ANCHOR_POS[anchor]
            actual = np.sqrt((x - ax)**2 + (y - ay)**2)
            measured = distances[anchor]
            error += (actual - measured) ** 2
        return error
    
    result = minimize(objective, [x0, y0], method='L-BFGS-B')
    return float(result.x[0]), float(result.x[1])

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
    
    # Update anchor data
    a = anchors[anchor]
    a.rssi_hist.append(rssi)
    a.last_update = ts
    a.last_mac = mac
    
    rssi_avg = np.mean(a.rssi_hist)
    c = CALIB[anchor]
    a.distance = rssi_to_distance(rssi_avg, c["rssi1m"], c["n"])
    
    # Check how many anchors have recent data
    valid_anchors = {}
    for name, anchor_data in anchors.items():
        if anchor_data.distance is None:
            continue
        age = ts - anchor_data.last_update
        if age <= MAX_AGE:
            valid_anchors[name] = anchor_data.distance
    
    if len(valid_anchors) < MIN_ANCHORS:
        return
    
    # Rate limit output
    if ts - last_plot_time < 0.1:
        return
    
    last_plot_time = ts
    frame_count += 1
    
    # Calculate positions using all methods
    x_wcl, y_wcl = wcl_2d(valid_anchors, p=2.0)
    x_ls, y_ls = ls_2d_grid(valid_anchors, step=0.05)
    x_tri, y_tri = trilateration_2d(valid_anchors)
    x_nlls, y_nlls = multilateration_nlls(valid_anchors)
    
    print(f"[{frame_count:04d}] Anchors: {len(valid_anchors)} | "
          f"WCL=({x_wcl:.2f},{y_wcl:.2f}) "
          f"LS=({x_ls:.2f},{y_ls:.2f}) "
          f"TRI=({x_tri:.2f},{y_tri:.2f}) "
          f"NLLS=({x_nlls:.2f},{y_nlls:.2f})")
    
    # Put result in queue for plotting
    position_queue.put({
        "distances": valid_anchors,
        "wcl": (x_wcl, y_wcl),
        "ls": (x_ls, y_ls),
        "tri": (x_tri, y_tri),
        "nlls": (x_nlls, y_nlls),
    })

# =========================
# MAIN
# =========================
def main():
    # MQTT setup
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(BROKER_HOST, BROKER_PORT)
    client.loop_start()
    
    # Plot setup
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Calculate bounds
    xs = [p[0] for p in ANCHOR_POS.values()]
    ys = [p[1] for p in ANCHOR_POS.values()]
    margin = 1.0
    ax.set_xlim(min(xs) - margin, max(xs) + margin)
    ax.set_ylim(min(ys) - margin, max(ys) + margin)
    ax.set_aspect('equal')
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    ax.set_title("2D Indoor Positioning")
    ax.grid(True, alpha=0.3)
    
    # Draw anchors
    for name, (x, y) in ANCHOR_POS.items():
        ax.plot(x, y, 's', color='black', markersize=15, zorder=10)
        ax.annotate(name, (x, y), textcoords="offset points",
                   xytext=(0, 12), ha='center', fontweight='bold', fontsize=11)
    
    # Distance circles (will be updated)
    circles = {}
    for name, (x, y) in ANCHOR_POS.items():
        circle = plt.Circle((x, y), 0.1, fill=False, linestyle='--', alpha=0.4, color='gray')
        ax.add_patch(circle)
        circles[name] = circle
    
    # Position markers for each method
    dot_wcl, = ax.plot([np.nan], [np.nan], 'o', color='blue', 
                       markersize=14, label='WCL (p=2)', alpha=0.8)
    dot_ls, = ax.plot([np.nan], [np.nan], 's', color='green', 
                      markersize=12, label='Least Squares (Grid)', alpha=0.8)
    dot_tri, = ax.plot([np.nan], [np.nan], '^', color='orange', 
                       markersize=12, label='Trilateration', alpha=0.8)
    dot_nlls, = ax.plot([np.nan], [np.nan], 'D', color='red', 
                        markersize=12, label='NLLS', alpha=0.8)
    
    # Legend with all markers
    ax.legend(handles=[dot_wcl, dot_ls, dot_tri, dot_nlls], 
              loc='upper right', title='Methods')
    
    # Info text
    info = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=9,
                   va='top', family='monospace',
                   bbox=dict(facecolor='wheat', alpha=0.8))
    
    fig.tight_layout()
    
    print("[MAIN] Running... Close window to stop")
    print(f"[MAIN] Using {len(ANCHOR_POS)} anchors: {list(ANCHOR_POS.keys())}")
    
    running = True
    
    def on_close(event):
        nonlocal running
        running = False
    
    fig.canvas.mpl_connect('close_event', on_close)
    
    try:
        while running and plt.fignum_exists(fig.number):
            try:
                # Get latest position (discard old ones)
                result = None
                while True:
                    try:
                        result = position_queue.get_nowait()
                    except Empty:
                        break
                
                if result:
                    # Update position markers
                    x, y = result["wcl"]
                    dot_wcl.set_xdata([x])
                    dot_wcl.set_ydata([y])
                    
                    x, y = result["ls"]
                    dot_ls.set_xdata([x])
                    dot_ls.set_ydata([y])
                    
                    x, y = result["tri"]
                    dot_tri.set_xdata([x])
                    dot_tri.set_ydata([y])
                    
                    x, y = result["nlls"]
                    dot_nlls.set_xdata([x])
                    dot_nlls.set_ydata([y])
                    
                    # Update distance circles
                    for name, dist in result["distances"].items():
                        if name in circles:
                            circles[name].set_radius(dist)
                    
                    # Update info text
                    lines = ["Distances:"]
                    for name, dist in sorted(result["distances"].items()):
                        lines.append(f"  {name}: {dist:.2f}m")
                    lines.append("─" * 15)
                    lines.append("Positions:")
                    lines.append(f"  WCL:  ({result['wcl'][0]:.2f}, {result['wcl'][1]:.2f})")
                    lines.append(f"  LS:   ({result['ls'][0]:.2f}, {result['ls'][1]:.2f})")
                    lines.append(f"  TRI:  ({result['tri'][0]:.2f}, {result['tri'][1]:.2f})")
                    lines.append(f"  NLLS: ({result['nlls'][0]:.2f}, {result['nlls'][1]:.2f})")
                    
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