from unittest import result
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Deque, Dict, Any, Optional
from collections import deque
import threading
import time
import json

import numpy as np
import paho.mqtt.client as mqtt

# ----------------------------
# Config
# ----------------------------
MQTT_BROKER = "127.0.0.1"
MQTT_PORT = 1883
MQTT_TOPIC = "ips/rssi"

HISTORY_SECONDS_DEFAULT = 30
MAX_HISTORY_POINTS = 5000

# Anchor positions and calibration (from 1d_Ex.py)
ANCHOR_POS = {"A1": 0.0, "A2": 1.0}
CALIB = {
    "A1": {"rssi1m": -65.07, "n": 1.435},
    "A2": {"rssi1m": -67.47, "n": 1.810},
}

RSSI_SMOOTH_N = 5
D_MIN, D_MAX = 0.1, 20.0

# ----------------------------
# Shared state
# ----------------------------
lock = threading.Lock()

# Per-anchor RSSI history for smoothing
anchor_rssi_history: Dict[str, deque] = {
    "A1": deque(maxlen=RSSI_SMOOTH_N),
    "A2": deque(maxlen=RSSI_SMOOTH_N),
}

latest: Dict[str, Any] = {
    "ts": None,
    "frame": "1d",
    "position": {"x_wcl": None, "x_ls": None},
    "distances": {"d1": None, "d2": None},
    "method": "wcl_p2",
    "quality": {
        "latency_ms": None,
        "anchors_seen": 0,
        "staleness_ms": None,
    },
    "rssi": {},
    "rssi_smooth" : {"A1": None, "A2": None}
}

history: Deque[Dict[str, Any]] = deque(maxlen=MAX_HISTORY_POINTS)
event_id = 0

# ----------------------------
# Positioning algorithms from 1d_Ex.py
# ----------------------------
def rssi_to_distance(rssi: float, rssi1m: float, n: float) -> float:
    """Convert RSSI to distance using log-distance path loss model."""
    d = 10 ** ((rssi1m - rssi) / (10.0 * n))
    return float(np.clip(d, D_MIN, D_MAX))

def wcl_p2(d1: float, d2: float) -> float:
    """Weighted Centroid Localization with p=2 (inverse distance squared)."""
    w1 = 1 / (d1**2 + 0.001)
    w2 = 1 / (d2**2 + 0.001)
    x = (w1 * ANCHOR_POS["A1"] + w2 * ANCHOR_POS["A2"]) / (w1 + w2)
    return float(np.clip(x, 0, max(ANCHOR_POS.values())))

def least_squares(d1: float, d2: float) -> float:
    """Least Squares - find x that minimizes distance error."""
    x_min = min(ANCHOR_POS.values())
    x_max = max(ANCHOR_POS.values())
    xs = np.linspace(x_min, x_max, 500)
    
    err = (np.abs(xs - ANCHOR_POS["A1"]) - d1)**2 + \
          (np.abs(xs - ANCHOR_POS["A2"]) - d2)**2
    
    return float(xs[np.argmin(err)])

def estimate_position(rssi_map: Dict[str, float]) -> Optional[Dict[str, float]]:
    """Estimate 1D position using smoothed RSSI values."""
    if "A1" not in rssi_map or "A2" not in rssi_map:
        return None
    
    # Get smoothed RSSI
    if len(anchor_rssi_history["A1"]) == 0 or len(anchor_rssi_history["A2"]) == 0:
        return None
    
    rssi1_avg = np.mean(anchor_rssi_history["A1"])
    rssi2_avg = np.mean(anchor_rssi_history["A2"])
    
    # Convert to distances
    d1 = rssi_to_distance(rssi1_avg, CALIB["A1"]["rssi1m"], CALIB["A1"]["n"])
    d2 = rssi_to_distance(rssi2_avg, CALIB["A2"]["rssi1m"], CALIB["A2"]["n"])
    
    # Calculate positions
    x_wcl = wcl_p2(d1, d2)
    x_ls = least_squares(d1, d2)
    
    return {
  "x_wcl": x_wcl, "x_ls": x_ls, "d1": d1, "d2": d2,
  "rssi1_avg": float(rssi1_avg), "rssi2_avg": float(rssi2_avg)
}


def update_state(anchor_id: str, rssi_val: float, msg_ts: Optional[float] = None) -> None:
    global event_id

    now = time.time()
    if msg_ts is None:
        msg_ts = now

    with lock:
        # Update RSSI and history
        latest["rssi"][anchor_id] = float(rssi_val)
        
        if anchor_id in anchor_rssi_history:
            anchor_rssi_history[anchor_id].append(rssi_val)

        result = estimate_position(latest["rssi"])

        latest["ts"] = msg_ts
        
        if result:
            latest["position"]["x_wcl"] = result["x_wcl"]
            latest["position"]["x_ls"] = result["x_ls"]
            latest["distances"]["d1"] = result["d1"]
            latest["distances"]["d2"] = result["d2"]
            latest["rssi_smooth"]["A1"] = result["rssi1_avg"]
            latest["rssi_smooth"]["A2"] = result["rssi2_avg"]


        anchors_seen = len(latest["rssi"])
        latest["quality"]["anchors_seen"] = anchors_seen
        latest["quality"]["staleness_ms"] = int((now - msg_ts) * 1000)

        if result:
            point = {
                "ts": msg_ts,
                "x_wcl": result["x_wcl"],
                "x_ls": result["x_ls"],
                "d1": result["d1"],
                "d2": result["d2"],
                "rssi": dict(latest["rssi"])
            }
            history.append(point)

        event_id += 1

# ----------------------------
# MQTT client
# ----------------------------
def on_connect(client, userdata, flags, rc):
    print("MQTT connected:", rc)
    client.subscribe(MQTT_TOPIC)

def on_message(client, userdata, msg):
    payload = msg.payload.decode("utf-8", errors="ignore").strip()
    
    anchor_id = None
    rssi_val = None
    msg_ts = None

    try:
        obj = json.loads(payload)
        if isinstance(obj, dict):
            anchor_id = obj.get("anchor")
            rssi_val = obj.get("rssi")
            msg_ts = obj.get("ts")
    except Exception:
        return

    if anchor_id is None or rssi_val is None:
        return

    if isinstance(msg_ts, (int, float)) and msg_ts > 1e12:
        msg_ts = msg_ts / 1000.0

    update_state(anchor_id, float(rssi_val), msg_ts)

def start_mqtt_loop():
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
    client.loop_forever()

# ----------------------------
# FastAPI app
# ----------------------------
app = FastAPI(title="IPS Live Server", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def _startup():
    t = threading.Thread(target=start_mqtt_loop, daemon=True)
    t.start()

@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse("""
<!doctype html>
<html>
  <head>
    <meta charset="utf-8"/>
    <title>IPS Live Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-2.30.0.min.js"></script>
    <style>
      body { font-family: Arial, sans-serif; margin: 20px; }
      .grid {
        display: grid;
        grid-template-columns: 1.4fr 1fr;
        gap: 16px;
        align-items: start;
      }
      .box { padding: 10px; border: 1px solid #ddd; border-radius: 10px; }
      #status { margin-bottom: 12px; color: #333; }
      #posChart { width: 100%; height: 520px; }
      #rssiA1, #rssiA2 { width: 100%; height: 240px; }
      .rightCol { display: grid; grid-template-rows: 1fr 1fr; gap: 16px; }
    </style>
  </head>
  <body>
    <h2>Real-time Indoor Tracking (1D)</h2>
    <div class="box" id="status">Connecting...</div>

    <div class="grid">
      <div class="box">
        <div id="posChart"></div>
      </div>

      <div class="rightCol">
        <div class="box"><div id="rssiA1"></div></div>
        <div class="box"><div id="rssiA2"></div></div>
      </div>
    </div>

    <script>
      const statusEl = document.getElementById("status");

      // -------- Position chart (left, big) --------
      const posWCL = { x: [], y: [], mode: "lines+markers", name: "WCL (p=2)" };
      const posLS  = { x: [], y: [], mode: "lines+markers", name: "Least Squares" };

      Plotly.newPlot("posChart", [posWCL, posLS], {
        xaxis: { title: "Time" },
        yaxis: { title: "x (m)", range: [-0.2, 1.2] },
        margin: { t: 20 }
      });

      // -------- RSSI charts (right, stacked) --------
      function makeRssiPlot(divId, title) {
        const raw = { x: [], y: [], mode: "lines+markers", name: "RSSI (raw)" };
        const sm  = { x: [], y: [], mode: "lines+markers", name: "RSSI (smoothed)" };
        Plotly.newPlot(divId, [raw, sm], {
          title: { text: title, x: 0.05, xanchor: "left" },
          xaxis: { title: "Time" },
          yaxis: { title: "RSSI (dBm)", autorange: true },
          margin: { t: 35 }
        });
      }

      makeRssiPlot("rssiA1", "Anchor A1 RSSI");
      makeRssiPlot("rssiA2", "Anchor A2 RSSI");

      // Keep last N points
      const MAX_POINTS = 300;

      function extend(divId, t, ySeries) {
        // ySeries: array of y arrays, one per trace in that div
        Plotly.extendTraces(divId, { x: ySeries.map(_ => [t]), y: ySeries.map(y => [y]) }, 
                           [...Array(ySeries.length).keys()], MAX_POINTS);
      }

      function extendPos(t, x_wcl, x_ls) {
        Plotly.extendTraces("posChart",
          { x: [[t], [t]], y: [[x_wcl], [x_ls]] },
          [0, 1],
          MAX_POINTS
        );
      }

      // -------- SSE stream --------
      const es = new EventSource("/v1/stream/position");

      es.onmessage = (evt) => {
        try {
          const data = JSON.parse(evt.data);

          const ts = data.ts;
          if (!ts) return;
          const t = new Date(ts * 1000);

          const x_wcl = data.position?.x_wcl;
          const x_ls  = data.position?.x_ls;

          const rssiA1 = data.rssi?.A1;
          const rssiA2 = data.rssi?.A2;

          const rssiA1s = data.rssi_smooth?.A1;
          const rssiA2s = data.rssi_smooth?.A2;

          const d1 = data.distances?.d1;
          const d2 = data.distances?.d2;

          statusEl.innerHTML =
            `<b>Live</b> | d1=${(d1 ?? NaN).toFixed(2)}m d2=${(d2 ?? NaN).toFixed(2)}m ` +
            `| WCL=${(x_wcl ?? NaN).toFixed(2)}m LS=${(x_ls ?? NaN).toFixed(2)}m`;

          // Update position chart only when we have numbers
          if (x_wcl !== null && x_wcl !== undefined && x_ls !== null && x_ls !== undefined) {
            extendPos(t, x_wcl, x_ls);
          }

          // Update RSSI charts when present
          if (rssiA1 !== null && rssiA1 !== undefined) {
            extend("rssiA1", t, [rssiA1, (rssiA1s ?? rssiA1)]);
          }
          if (rssiA2 !== null && rssiA2 !== undefined) {
            extend("rssiA2", t, [rssiA2, (rssiA2s ?? rssiA2)]);
          }

        } catch (e) {
          // ignore parse errors
        }
      };

      es.onerror = () => {
        statusEl.textContent = "Disconnected. Retrying...";
      };
    </script>
  </body>
</html>
    """)


@app.get("/v1/position/latest")
def position_latest():
    with lock:
        return JSONResponse(content=latest)

@app.get("/v1/position/history")
def position_history(seconds: int = HISTORY_SECONDS_DEFAULT):
    now = time.time()
    with lock:
        items = [p for p in list(history) if (now - p["ts"]) <= seconds]
    return JSONResponse(content={"seconds": seconds, "points": items})

@app.get("/v1/stream/position")
def stream_position():
    def gen():
        last_seen = -1
        while True:
            time.sleep(0.1)
            with lock:
                cur_id = event_id
                payload = dict(latest)
            if cur_id != last_seen:
                last_seen = cur_id
                yield f"data: {json.dumps(payload)}\n\n"
    return StreamingResponse(gen(), media_type="text/event-stream")