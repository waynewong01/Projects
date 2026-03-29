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
MAX_AGE = 2.0
MIN_ANCHORS = 3

# Position smoothing
POS_ALPHA = 0.3  # Lower = smoother but slower response (0.1-0.5 recommended)

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


# def ls_2d_grid(distances: dict, step: float = 0.05):
#     """Least Squares 2D via grid search."""
#     xs_anchors = [p[0] for p in ANCHOR_POS.values()]
#     ys_anchors = [p[1] for p in ANCHOR_POS.values()]
    
#     x_min, x_max = min(xs_anchors) - 0.5, max(xs_anchors) + 0.5
#     y_min, y_max = min(ys_anchors) - 0.5, max(ys_anchors) + 0.5
    
#     xs = np.arange(x_min, x_max + step, step)
#     ys = np.arange(y_min, y_max + step, step)
#     X, Y = np.meshgrid(xs, ys)
    
#     error = np.zeros_like(X)
    
#     for anchor, dist in distances.items():
#         if anchor not in ANCHOR_POS:
#             continue
#         ax, ay = ANCHOR_POS[anchor]
#         actual_dist = np.sqrt((X - ax)**2 + (Y - ay)**2)
#         error += (actual_dist - dist) ** 2
    
#     min_idx = np.unravel_index(np.argmin(error), error.shape)
#     return float(X[min_idx]), float(Y[min_idx])


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

            # ensure overlap: ri + rj >= dij - eps
            if D[i] + D[j] < dij - eps:
                delta = (dij - eps) - (D[i] + D[j])
                D[i] += 0.5 * delta
                D[j] += 0.5 * delta
                changed = True

            # ensure |ri - rj| <= dij + eps
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
    """One-shot BCCP with exactly 3 anchors. Returns (x,y) or (nan,nan)."""
    AP = np.array([anchor_pos[a] for a in anchor_names], float)
    D0 = np.array([max(float(distances[a]), 1e-9) for a in anchor_names], float)

    D = _project_radii_to_intersect(AP, D0)
    D = _inflate_for_overlap(AP, D, alpha=0.20)

    # intersections
    E, F = _circ_ints(AP[0], D[0], AP[1], D[1])
    G, H = _circ_ints(AP[0], D[0], AP[2], D[2])
    I, J = _circ_ints(AP[1], D[1], AP[2], D[2])

    # closed points (each picked using the 3rd circle)
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
    """
    BCCP (Best Closed Point) for 3 or 4 anchors.
    
    - If 3 anchors: run BCCP once
    - If 4+ anchors: try all triplets and pick estimate with lowest residual
    
    Returns: (x, y)
    """
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
    
    # Calculate positions using all methods (including BCCP)
    x_wcl, y_wcl = wcl_2d(valid_anchors, p=2.0)
    # x_ls, y_ls = ls_2d_grid(valid_anchors, step=0.05)
    x_tri, y_tri = trilateration_2d(valid_anchors)

    x_bccp, y_bccp = bccp_2d(valid_anchors)
    
    print(f"[{frame_count:04d}] Anchors: {len(valid_anchors)} | "
          f"WCL=({x_wcl:.2f},{y_wcl:.2f}) "
        #   f"LS=({x_ls:.2f},{y_ls:.2f}) "
          f"TRI=({x_tri:.2f},{y_tri:.2f}) "
          f"BCCP=({x_bccp:.2f},{y_bccp:.2f})")
    
    position_queue.put({
        "distances": valid_anchors,
        "wcl": (x_wcl, y_wcl),
        # "ls": (x_ls, y_ls),
        "tri": (x_tri, y_tri),
        "bccp": (x_bccp, y_bccp),
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
    ax.set_title("2D Indoor Positioning (with BCCP)")
    ax.grid(True, alpha=0.3)
    
    for name, (x, y) in ANCHOR_POS.items():
        ax.plot(x, y, 's', color='black', markersize=15, zorder=10)
        ax.annotate(name, (x, y), textcoords="offset points",
                   xytext=(0, 12), ha='center', fontweight='bold', fontsize=11)
    
    circles = {}
    for name, (x, y) in ANCHOR_POS.items():
        circle = plt.Circle((x, y), 0.1, fill=False, linestyle='--', alpha=0.4, color='gray')
        ax.add_patch(circle)
        circles[name] = circle
    
    # Position markers for each method (added BCCP)
    dot_wcl, = ax.plot([np.nan], [np.nan], 'o', color='blue', 
                       markersize=14, label='WCL (p=2)', alpha=0.8)
    # dot_ls, = ax.plot([np.nan], [np.nan], 's', color='green', 
    #                   markersize=12, label='Least Squares (Grid)', alpha=0.8)
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
                    
                    # dot_ls.set_xdata([result["ls"][0]])
                    # dot_ls.set_ydata([result["ls"][1]])
                    
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
                    lines.append("─" * 18)
                    lines.append("Positions:")
                    lines.append(f"  WCL:  ({result['wcl'][0]:.2f}, {result['wcl'][1]:.2f})")
                    # lines.append(f"  LS:   ({result['ls'][0]:.2f}, {result['ls'][1]:.2f})")
                    lines.append(f"  TRI:  ({result['tri'][0]:.2f}, {result['tri'][1]:.2f})")
                    lines.append(f"  BCCP: ({result['bccp'][0]:.2f}, {result['bccp'][1]:.2f})")
                    
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