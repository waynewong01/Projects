import json
import time
from collections import defaultdict, deque
from queue import Queue, Empty

import numpy as np
import paho.mqtt.client as mqtt
import matplotlib
matplotlib.use('TkAgg')  # Force TkAgg backend
import matplotlib.pyplot as plt

# =========================
# USER CONFIG
# =========================
BROKER_HOST = "127.0.0.1"
BROKER_PORT = 1883
TOPIC = "ips/rssi"

ANCHOR_POS = {"A1": 0.0, "A2": 5.2}
CALIB = {
    "A1": {"rssi1m": -65.07, "n": 1.435},
    "A2": {"rssi1m": -67.47, "n": 1.810},
}

TARGET_MAC = None
FRAME_WINDOW_S = 1.0
RSSI_SMOOTH_N = 8
D_MIN, D_MAX = 0.1, 20.0

# =========================
# STATE
# =========================
rssi_hist = defaultdict(lambda: deque(maxlen=RSSI_SMOOTH_N))
frame = defaultdict(dict)
last_emit = defaultdict(float)
position_queue = Queue(maxsize=1)

# =========================
# CORE FUNCTIONS
# =========================
def rssi_to_distance(rssi, rssi1m, n):
    d = 10 ** ((rssi1m - rssi) / (10.0 * n))
    return float(np.clip(d, D_MIN, D_MAX))

def wcl_1d(dist_by_anchor):
    num, den = 0.0, 0.0
    for a, d in dist_by_anchor.items():
        w = 1.0 / (d ** 2 + 1e-6)
        num += w * ANCHOR_POS[a]
        den += w
    return num / den if den > 0 else np.nan

def ls_1d(dist_by_anchor):
    xs = np.linspace(0, max(ANCHOR_POS.values()), 500)
    err = sum((np.abs(xs - ANCHOR_POS[a]) - d) ** 2 for a, d in dist_by_anchor.items())
    return float(xs[np.argmin(err)])

# =========================
# MQTT
# =========================
def on_connect(client, userdata, flags, rc):
    print(f"[MQTT] Connected (rc={rc})")
    client.subscribe(TOPIC)

def on_message(client, userdata, msg):
    global TARGET_MAC
    
    try:
        data = json.loads(msg.payload.decode())
    except:
        return
    
    anchor, mac, rssi = data.get("anchor"), data.get("mac"), data.get("rssi")
    
    if anchor not in ANCHOR_POS or not isinstance(rssi, (int, float)):
        return
    
    if TARGET_MAC is None:
        TARGET_MAC = mac
        print(f"[TRACK] Target: {mac}")
    
    if mac != TARGET_MAC:
        return
    
    ts = time.time()
    
    # Smooth and convert
    rssi_hist[(anchor, mac)].append(rssi)
    rssi_avg = np.mean(rssi_hist[(anchor, mac)])
    c = CALIB[anchor]
    dist = rssi_to_distance(rssi_avg, c["rssi1m"], c["n"])
    
    # Update frame
    frame[mac][anchor] = (ts, dist)
    
    # Check complete
    if "A1" in frame[mac] and "A2" in frame[mac]:
        t1, d1 = frame[mac]["A1"]
        t2, d2 = frame[mac]["A2"]
        
        if abs(t1 - t2) <= FRAME_WINDOW_S and ts - last_emit[mac] > 0.1:
            last_emit[mac] = ts
            
            dists = {"A1": d1, "A2": d2}
            x_wcl = wcl_1d(dists)
            x_ls = ls_1d(dists)
            
            result = {"x_wcl": x_wcl, "x_ls": x_ls, "d1": d1, "d2": d2}
            
            # Clear and put
            try:
                position_queue.get_nowait()
            except Empty:
                pass
            position_queue.put(result)
            
            print(f"[FRAME] d1={d1:.2f}m d2={d2:.2f}m | WCL={x_wcl:.2f}m LS={x_ls:.2f}m")

# =========================
# MAIN
# =========================
def main():
    # Setup MQTT
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(BROKER_HOST, BROKER_PORT)
    client.loop_start()
    
    # Setup plot
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 4))
    
    x_max = max(ANCHOR_POS.values())
    ax.set_xlim(-0.5, x_max + 0.5)
    ax.set_ylim(-1, 1.5)
    ax.set_yticks([])
    ax.set_xlabel("Position (m)")
    ax.set_title("1D Tracking - Close window to stop")
    ax.grid(True, axis='x', alpha=0.3)
    
    # Draw anchors
    for name, x in ANCHOR_POS.items():
        ax.axvline(x, color='gray', linestyle='--', alpha=0.5)
        ax.plot(x, 1.0, 's', color='black', markersize=15)
        ax.text(x, 1.2, name, ha='center', fontweight='bold')
    
    # Position markers
    wcl_dot, = ax.plot([np.nan], [0.2], 'o', color='green', markersize=15, label='WCL')
    ls_dot, = ax.plot([np.nan], [-0.2], 'o', color='blue', markersize=15, label='LS')
    
    # Info text
    info_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=10,
                        verticalalignment='top', fontfamily='monospace',
                        bbox=dict(facecolor='wheat', alpha=0.8))
    
    ax.legend(loc='upper right')
    fig.tight_layout()
    
    print("[MAIN] Starting... Close plot window to stop.")
    
    try:
        while plt.fignum_exists(fig.number):
            # Check for new data
            try:
                result = position_queue.get_nowait()
                
                # Update markers
                wcl_dot.set_xdata([result["x_wcl"]])
                ls_dot.set_xdata([result["x_ls"]])
                
                # Update text
                info_text.set_text(
                    f"d(A1): {result['d1']:.2f}m\n"
                    f"d(A2): {result['d2']:.2f}m\n"
                    f"WCL:  {result['x_wcl']:.2f}m\n"
                    f"LS:   {result['x_ls']:.2f}m"
                )
                
                # Redraw
                fig.canvas.draw()
                fig.canvas.flush_events()
                
            except Empty:
                pass
            
            plt.pause(0.05)  # 20 FPS max, allows GUI to process
            
    except KeyboardInterrupt:
        print("\n[MAIN] Interrupted")
    finally:
        client.loop_stop()
        client.disconnect()
        plt.close('all')
        print("[MAIN] Done")

if __name__ == "__main__":
    main()