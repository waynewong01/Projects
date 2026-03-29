import json
import time
import csv
from pathlib import Path
import paho.mqtt.client as mqtt

BROKER_HOST = "127.0.0.1"
BROKER_PORT = 1883
TOPIC = "ips/rssi"

ANCHOR_FILTER = "A8"
TAG_MAC_FILTER = None
DISTANCE_M = 6   # Change this for each distance
MAX_READINGS = 20        # Readings per distance

# Single file for all distances
DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "calibration"
OUT_CSV = DATA_DIR / f"calibration_{ANCHOR_FILTER}.csv"

# Counter for readings
reading_count = 0
session_count = 0

def count_existing_for_distance():
    """Count readings already in file for current distance."""
    if not OUT_CSV.exists():
        return 0
    try:
        with OUT_CSV.open("r", newline="") as f:
            reader = csv.reader(f)
            next(reader, None)  # Skip header
            return sum(1 for row in reader if len(row) >= 4 and row[3] == str(DISTANCE_M))
    except PermissionError:
        return -1

def ensure_header():
    """Create file with header if it doesn't exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not OUT_CSV.exists():
        try:
            with OUT_CSV.open("w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["ts", "anchor", "mac", "distance_m", "rssi"])
            print(f"[NEW] Created {OUT_CSV}")
        except PermissionError:
            print(f"[ERROR] Cannot create {OUT_CSV}")
            raise
    else:
        # Test if file is accessible
        try:
            with OUT_CSV.open("a", newline="") as f:
                pass
        except PermissionError:
            print(f"[ERROR] Cannot write to {OUT_CSV} - file is open in Excel!")
            print("[TIP] Close Excel and try again")
            raise

def on_connect(client, userdata, flags, rc):
    global reading_count
    print(f"[MQTT] connected rc={rc}")
    client.subscribe(TOPIC)
    print(f"[MQTT] subscribed {TOPIC}")
    print(f"[LOG] anchor={ANCHOR_FILTER} distance={DISTANCE_M}m")
    print(f"[LOG] output={OUT_CSV}")
    print(f"[LOG] Existing readings for {DISTANCE_M}m: {reading_count}")
    print(f"[LOG] Will collect {MAX_READINGS - reading_count} more readings")
    print("-" * 60)

def on_message(client, userdata, msg):
    global reading_count, session_count
    
    # Stop if we've reached the limit for this distance
    if reading_count >= MAX_READINGS:
        return
    
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

    try:
        with OUT_CSV.open("a", newline="") as f:
            w = csv.writer(f)
            w.writerow([ts, anchor, mac, DISTANCE_M, int(rssi)])
    except PermissionError:
        print(f"[ERROR] Cannot write - file locked. Close Excel!")
        return

    reading_count += 1
    session_count += 1
    remaining = MAX_READINGS - reading_count
    
    print(f"[{reading_count:03d}/{MAX_READINGS}] {time.strftime('%H:%M:%S')}  "
          f"{anchor}  {DISTANCE_M}m  rssi={rssi:4d}  (remaining: {remaining})")
    
    # Auto-stop when done
    if reading_count >= MAX_READINGS:
        print("-" * 60)
        print(f"[DONE] Collected {MAX_READINGS} readings for {DISTANCE_M}m!")
        print(f"[DONE] This session: {session_count} readings")
        print(f"[DONE] Saved to: {OUT_CSV.absolute()}")
        print(f"[NEXT] Change DISTANCE_M and run again for next distance")
        client.disconnect()

def main():
    global reading_count
    
    # Check existing readings for this distance
    existing = count_existing_for_distance()
    
    if existing == -1:
        print(f"[ERROR] {OUT_CSV} is locked - close Excel and try again!")
        return
    
    reading_count = existing
    
    if reading_count >= MAX_READINGS:
        print(f"[INFO] Already have {reading_count} readings for {DISTANCE_M}m")
        print(f"[INFO] Change DISTANCE_M to collect for another distance")
        return
    
    ensure_header()
    
    client = mqtt.Client(client_id=f"calib-logger-{ANCHOR_FILTER}")
    client.on_connect = on_connect
    client.on_message = on_message
    
    try:
        client.connect(BROKER_HOST, BROKER_PORT, keepalive=60)
        client.loop_forever()
    except KeyboardInterrupt:
        print(f"\n[STOPPED] This session: {session_count} readings")
        print(f"[STOPPED] Total for {DISTANCE_M}m: {reading_count} readings")
        print(f"[STOPPED] Saved to: {OUT_CSV.absolute()}")
    
    print("[EXIT] Done")

if __name__ == "__main__":
    main()
