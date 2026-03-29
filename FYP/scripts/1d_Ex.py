"""
Simplified tracker that ignores MAC addresses.
Tracks the strongest signal at each anchor.
"""
import json
import time
from collections import deque

import numpy as np
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
#5.2
ANCHOR_POS = {"A1": 0.0, "A2": 1}
CALIB = {
    "A1": {"rssi1m": -65.07, "n": 1.435},
    "A2": {"rssi1m": -67.47, "n": 1.810},
}

RSSI_SMOOTH_N = 5
D_MIN, D_MAX = 0.1, 20.0
MAX_AGE = 2.0  # Max age of reading in seconds


# =========================
# STATE - Per anchor, ignoring MAC
# =========================
class AnchorData:
    def __init__(self):
        self.rssi_hist = deque(maxlen=RSSI_SMOOTH_N)
        self.last_update = 0
        self.distance = None
        self.last_mac = None

anchors = {
    "A1": AnchorData(),
    "A2": AnchorData(),
}

frame_count = 0
last_plot_time = 0

# =========================
# FUNCTIONS
# =========================
def rssi_to_distance(rssi, rssi1m, n):
    d = 10 ** ((rssi1m - rssi) / (10.0 * n))
    return float(np.clip(d, D_MIN, D_MAX))

def wcl_p1(d1, d2):
    """Weighted Centroid with p=1 (inverse distance)."""
    x = (d2 * ANCHOR_POS["A1"] + d1 * ANCHOR_POS["A2"]) / (d1 + d2 + 0.001)
    return np.clip(x, 0, max(ANCHOR_POS.values()))

def wcl_p2(d1, d2):
    """Weighted Centroid with p=2 (inverse distance squared)."""
    w1 = 1 / (d1**2 + 0.001)
    w2 = 1 / (d2**2 + 0.001)
    x = (w1 * ANCHOR_POS["A1"] + w2 * ANCHOR_POS["A2"]) / (w1 + w2)
    return np.clip(x, 0, max(ANCHOR_POS.values()))

def least_squares(d1, d2):
    """Least Squares - find x that minimizes distance error."""
    x_min = min(ANCHOR_POS.values())
    x_max = max(ANCHOR_POS.values())
    xs = np.linspace(x_min, x_max, 500)
    
    # Error = (|x - x_A1| - d1)² + (|x - x_A2| - d2)²
    err = (np.abs(xs - ANCHOR_POS["A1"]) - d1)**2 + \
          (np.abs(xs - ANCHOR_POS["A2"]) - d2)**2
    
    return float(xs[np.argmin(err)])
# =========================
# MQTT
# =========================
def on_connect(client, userdata, flags, rc):
    print(f"[MQTT] Connected (rc={rc})")
    client.subscribe(TOPIC)

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
    
    # Update anchor data (ignore MAC - just use latest)
    a = anchors[anchor]
    a.rssi_hist.append(rssi)
    a.last_update = ts
    a.last_mac = mac
    
    rssi_avg = np.mean(a.rssi_hist)
    c = CALIB[anchor]
    a.distance = rssi_to_distance(rssi_avg, c["rssi1m"], c["n"])
    
    # Check if both anchors have recent data
    a1, a2 = anchors["A1"], anchors["A2"]
    
    if a1.distance is None or a2.distance is None:
        return
    
    age1 = ts - a1.last_update
    age2 = ts - a2.last_update
    
    if age1 > MAX_AGE or age2 > MAX_AGE:
        return
    
    # Rate limit output
    if ts - last_plot_time < 0.1:
        return
    
    last_plot_time = ts
    frame_count += 1
    
    # Calculate position
    d1, d2 = a1.distance, a2.distance
    

    x_wcl = wcl_p2(d1, d2)
    x_ls = least_squares(d1, d2)
    
    print(f"[{frame_count:04d}] d1={d1:.2f}m d2={d2:.2f}m | "
          f"WCL2={x_wcl:.2f}m LS={x_ls:.2f}m")
    
    # Put result in queue for plotting
    position_queue.put({
        "d1": d1, 
        "d2": d2, 
        "x_wcl": x_wcl,
        "x_ls": x_ls
    })
# =========================
# MAIN
# =========================
from queue import Queue, Empty
position_queue = Queue(maxsize=1)

def main():
    # MQTT setup
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(BROKER_HOST, BROKER_PORT)
    client.loop_start()
    
    # Plot setup
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 3))
    
    ax.set_xlim(-0.5, max(ANCHOR_POS.values()) + 0.5)
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    ax.set_xlabel("Position (m)")
    ax.set_title("1D Tracker")
    ax.grid(True, axis='x', alpha=0.3)
    
    for name, x in ANCHOR_POS.items():
        ax.axvline(x, color='gray', linestyle='--')
        ax.plot(x, 0.3, 's', color='black', markersize=12)
        ax.text(x, 0.4, name, ha='center', fontweight='bold')
    
    
    dot_wcl, = ax.plot([np.nan], [0.0], 'o', color='blue', markersize=16, label='WCL (g=2)')
    dot_ls, = ax.plot([np.nan], [0.2], 'o', color='red', markersize=16, label='Least Squares')
    ax.legend(handles=[dot_wcl, dot_ls], loc='upper right')
    info = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=10,
                   va='top', family='monospace',
                   bbox=dict(facecolor='wheat', alpha=0.8))
    
    fig.tight_layout()
    
    print("[MAIN] Running... Close window to stop")
    
    try:
        while plt.fignum_exists(fig.number):
            try:
                # Get latest position (discard old ones)
                result = None
                while True:
                    try:
                        result = position_queue.get_nowait()
                    except Empty:
                        break
                
                if result:
                    # Update all three dots with correct keys

                    dot_wcl.set_xdata([result["x_wcl"]])
                    dot_ls.set_xdata([result["x_ls"]])
                    
                    # Update info text
                    info.set_text(
                        f"Distances:\n"
                        f"  A1: {result['d1']:.2f}m\n"
                        f"  A2: {result['d2']:.2f}m\n"
                        f"─────────────\n"
                        f"Positions:\n"

                        f"  WCL(p=2): {result['x_wcl']:.2f}m\n"
                        f"  LS:       {result['x_ls']:.2f}m"
                    )
                    fig.canvas.draw()
                    fig.canvas.flush_events()
                    
            except Exception as e:
                print(f"Plot error: {e}")
            
            plt.pause(0.05)
            
    except KeyboardInterrupt:
        pass
    finally:
        client.loop_stop()
        plt.close('all')
        print("[MAIN] Done")

if __name__ == "__main__":
    main()