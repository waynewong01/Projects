import json
import time
import csv
from pathlib import Path
import paho.mqtt.client as mqtt

BROKER_HOST = "127.0.0.1"   # use "192.168.1.113" if not running on same PC
BROKER_PORT = 1883
TOPIC = "ips/rssi"

ANCHOR_FILTER = "A1"
TAG_MAC_FILTER = None       # keep None for phone tags (MAC may randomize)
DISTANCE_M = .5
OUT_CSV = Path("calibration1.csv")

def ensure_header():
    if not OUT_CSV.exists() or OUT_CSV.stat().st_size == 0:
        with OUT_CSV.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["ts", "anchor", "mac", "distance_m", "rssi"])

def on_connect(client, userdata, flags, rc):
    print(f"[MQTT] connected rc={rc}")
    client.subscribe(TOPIC)
    print(f"[MQTT] subscribed {TOPIC}")
    print(f"[LOG] anchor={ANCHOR_FILTER} distance={DISTANCE_M}m output={OUT_CSV}")

def on_message(client, userdata, msg):
    try:
        data = json.loads(msg.payload.decode("utf-8"))
    except Exception:
        return

    anchor = data.get("anchor")
    mac = data.get("mac")
    rssi = data.get("rssi")

    if anchor != ANCHOR_FILTER:
        return
    if TAG_MAC_FILTER and (not mac or mac.lower() != TAG_MAC_FILTER.lower()):
        return
    if not isinstance(rssi, (int, float)):
        return

    ts = time.time()

    with OUT_CSV.open("a", newline="") as f:
        w = csv.writer(f)
        w.writerow([ts, anchor, mac, DISTANCE_M, int(rssi)])

    print(f"{time.strftime('%H:%M:%S')}  {anchor}  {mac}  d={DISTANCE_M:.2f}m  rssi={rssi}")

def main():
    ensure_header()
    client = mqtt.Client(client_id=f"calib-logger-{ANCHOR_FILTER}")
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(BROKER_HOST, BROKER_PORT, keepalive=60)
    client.loop_forever()

if __name__ == "__main__":
    main()
