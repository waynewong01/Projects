"""
FastAPI server — API routes, SSE streaming, and HTML dashboard.
Includes experiment mode UI for structured data collection.
"""

import asyncio
import json
import logging
import os
import time
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, PlainTextResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .config import config
from .engine import PositioningEngine
from .experiment import ExperimentManager
from .mqtt_handler import MQTTHandler

logger = logging.getLogger(__name__)


class ConfigUpdatePayload(BaseModel):
    mode: Optional[str] = None
    line_length: Optional[float] = None
    room_width: Optional[float] = None
    room_height: Optional[float] = None
    anchor_positions: Optional[Dict[str, List[float]]] = None
    smoothing_factors: Optional[Dict[str, float]] = None
    distance_window_size: Optional[int] = Field(default=None, ge=1, le=1000)
    stale_anchor_timeout_s: Optional[float] = Field(default=None, gt=0)
    update_rate_hz: Optional[float] = Field(default=None, gt=0)
    fusion_mode: Optional[str] = None
    pillar_enabled: Optional[bool] = None
    pillar_vertices: Optional[List[List[float]]] = None


class GroundTruthPayload(BaseModel):
    x: float
    y: float = 0.0
    auto_reset: bool = True


class DistancePayload(BaseModel):
    anchor: Optional[str] = None
    anchor_id: Optional[str] = None
    distance: float
    rx_power: float = 0.0


class ExperimentStartPayload(BaseModel):
    point_id: str
    num_samples: int = Field(default=100, ge=1, le=10000)


class ExperimentPointPayload(BaseModel):
    point_id: str
    zone: str
    gt_x: float
    gt_y: float


class ExperimentPointsPayload(BaseModel):
    points: List[ExperimentPointPayload] = Field(default_factory=list)

# -------------------------------------------------------------------------
# Application setup
# -------------------------------------------------------------------------

app = FastAPI(title="UWB IPS Dashboard")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
  allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = PositioningEngine()
experiment = ExperimentManager(engine)
mqtt_handler = MQTTHandler(engine)


# -------------------------------------------------------------------------
# SSE stream (integrates experiment sample collection)
# -------------------------------------------------------------------------

@app.get("/api/stream")
async def stream():
    async def gen():
        while True:
            try:
                results = engine.compute_all()

                # If experiment is recording, collect this cycle as a sample
                if experiment.recording:
                    experiment.collect_sample(results)

                data = {
                    "timestamp": time.time(),
                    "mode": config.mode,
                  "fusion_mode": config.fusion_mode,
                    "positions": {
                        n: {
                            "position": (
                                {"x": r.position.x, "y": r.position.y}
                                if r.position else None
                            ),
                            "distances": r.distances,
                            "error": r.error,
                        }
                        for n, r in results.items()
                    },
                    "raw_distances": engine.raw_distances,
                    "rx_powers": engine.rx_powers,
                    "statistics": engine.get_statistics(),
                    "fusion_ab": engine.get_fusion_ab_metrics(),
                    "ground_truth": (
                        {"x": engine.ground_truth.x, "y": engine.ground_truth.y}
                        if engine.ground_truth else None
                    ),
                    "experiment": experiment.get_status(),
                }
                yield f"data: {json.dumps(data)}\n\n"
            except Exception as e:
                logger.error(f"Stream error: {e}")
            rate_hz = max(0.1, float(getattr(config, "update_rate_hz", 10.0) or 10.0))
            await asyncio.sleep(1.0 / rate_hz)

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


# -------------------------------------------------------------------------
# Config routes
# -------------------------------------------------------------------------

@app.get("/api/config")
async def get_config():
    return {
        "mode": config.mode,
        "line_length": config.line_length,
        "room_width": config.room_width,
        "room_height": config.room_height,
        "anchor_positions": config.anchor_positions,
        "smoothing_factors": config.smoothing_factors,
        "distance_window_size": config.distance_window_size,
        "stale_anchor_timeout_s": config.stale_anchor_timeout_s,
        "update_rate_hz": config.update_rate_hz,
        "fusion_mode": config.fusion_mode,
        "pillar_enabled": config.pillar_enabled,
        "pillar_vertices": config.pillar_vertices,
        "ground_truth": (
            {"x": engine.ground_truth.x, "y": engine.ground_truth.y}
            if engine.ground_truth else None
        ),
    }


@app.post("/api/config")
async def update_config(payload: ConfigUpdatePayload):
  new_config = payload.model_dump(exclude_none=True)
  if "anchor_positions" in new_config:
    normalized = {}
    for aid, pos in new_config["anchor_positions"].items():
      if len(pos) != 2:
        raise HTTPException(status_code=422, detail=f"Anchor {aid} must have exactly two coordinates")
      normalized[aid] = [float(pos[0]), float(pos[1])]
    new_config["anchor_positions"] = normalized
  if "pillar_vertices" in new_config:
    vertices = new_config["pillar_vertices"]
    if len(vertices) != 4:
      raise HTTPException(status_code=422, detail="pillar_vertices must contain exactly 4 points")
    normalized_vertices = []
    for vertex in vertices:
      if len(vertex) != 2:
        raise HTTPException(status_code=422, detail="Each pillar vertex must have exactly 2 coordinates")
      normalized_vertices.append([float(vertex[0]), float(vertex[1])])
    new_config["pillar_vertices"] = normalized_vertices
  engine.update_config(new_config)
  return {"status": "ok"}


# -------------------------------------------------------------------------
# Ground truth routes
# -------------------------------------------------------------------------

@app.post("/api/ground-truth")
async def set_gt(payload: GroundTruthPayload):
    engine.set_ground_truth(float(payload.x), float(payload.y), payload.auto_reset)
    return {"status": "ok"}


@app.delete("/api/ground-truth")
async def clear_gt():
    engine.clear_ground_truth()
    return {"status": "ok"}


@app.post("/api/reset-stats")
async def reset():
    engine.clear_errors()
    return {"status": "ok"}


# -------------------------------------------------------------------------
# Manual distance input (testing without MQTT)
# -------------------------------------------------------------------------

@app.post("/api/distance")
async def add_distance(payload: DistancePayload):
    aid = payload.anchor or payload.anchor_id
    if not aid:
        raise HTTPException(status_code=422, detail="Provide 'anchor' or 'anchor_id'")
    engine.add_distance(aid, float(payload.distance), float(payload.rx_power))
    return {"status": "ok"}


# -------------------------------------------------------------------------
# Experiment routes
# -------------------------------------------------------------------------

@app.get("/api/experiment/status")
async def experiment_status():
    return experiment.get_status()


@app.post("/api/experiment/start")
async def experiment_start(payload: ExperimentStartPayload):
    return experiment.start_recording(payload.point_id, payload.num_samples)


@app.post("/api/experiment/stop")
async def experiment_stop():
    return experiment.stop_recording()


@app.post("/api/experiment/points")
async def experiment_set_points(payload: ExperimentPointsPayload):
    try:
        experiment.set_test_points([p.model_dump() for p in payload.points])
    except (KeyError, ValueError) as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    return {"status": "ok"}


@app.post("/api/experiment/points/add")
async def experiment_add_point(payload: ExperimentPointPayload):
    try:
        experiment.add_test_point(payload.point_id, payload.zone, float(payload.gt_x), float(payload.gt_y))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    return {"status": "ok"}


@app.delete("/api/experiment/points/{point_id}")
async def experiment_remove_point(point_id: str):
    experiment.remove_test_point(point_id)
    return {"status": "ok"}


@app.delete("/api/experiment/results/{point_id}")
async def experiment_clear_point(point_id: str):
    experiment.clear_point_result(point_id)
    return {"status": "ok"}


@app.delete("/api/experiment/results")
async def experiment_clear_all():
    experiment.clear_all_results()
    return {"status": "ok"}


@app.get("/api/experiment/export/csv")
async def experiment_export_csv():
    csv_data = experiment.export_csv()
    return PlainTextResponse(
        csv_data,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=experiment_results.csv"},
    )


@app.get("/api/experiment/export/json")
async def experiment_export_json():
    json_data = experiment.export_json()
    return JSONResponse(
        json_data,
        headers={"Content-Disposition": "attachment; filename=experiment_results.json"},
    )


# -------------------------------------------------------------------------
# Dashboard HTML
# -------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    return DASHBOARD_HTML


# -------------------------------------------------------------------------
# Startup / shutdown
# -------------------------------------------------------------------------

@app.on_event("startup")
async def startup():
    logger.info(f"Starting UWB IPS Dashboard (mode={config.mode})")
    logger.info(f"Anchors: {config.anchor_positions}")
    mqtt_handler.connect()
    logger.info(f"Dashboard: http://localhost:{config.server_port}")


@app.on_event("shutdown")
async def shutdown():
    mqtt_handler.disconnect()


# =========================================================================
# HTML TEMPLATE
# =========================================================================

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>UWB IPS Dashboard</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Instrument+Sans:wght@400;500;600;700&display=swap');
*{margin:0;padding:0;box-sizing:border-box}
:root{
  --bg:#f5f5f0;--card:#fff;--border:#ddd8d0;
  --text:#1c1917;--text2:#57534e;--muted:#a8a29e;
  --wcl:#2563eb;--tri:#ea580c;--bccp:#7c3aed;--fused:#dc2626;--gt:#16a34a;
  --a1:#0891b2;--a2:#7c3aed;--a3:#db2777;--a4:#ca8a04;
}
body{font-family:'Instrument Sans',sans-serif;background:var(--bg);color:var(--text)}
.container{max-width:1600px;margin:0 auto;padding:16px}
header{display:flex;justify-content:space-between;align-items:center;padding:16px 0;border-bottom:2px solid var(--border);margin-bottom:20px}
.logo{display:flex;align-items:center;gap:12px}
.logo-icon{width:40px;height:40px;background:#1c1917;border-radius:6px;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:13px;color:#fff;font-family:'DM Mono',monospace;letter-spacing:1px}
.logo h1{font-size:22px;font-weight:700;letter-spacing:-.5px}
.logo span{color:var(--text2);font-size:13px}
.badges{display:flex;gap:10px;align-items:center}
.badge{padding:5px 14px;border-radius:4px;font-size:12px;font-weight:600;font-family:'DM Mono',monospace;letter-spacing:.5px}
.badge-mode{background:#1c1917;color:#fff}
.badge-status{background:var(--card);border:1px solid var(--border);display:flex;align-items:center;gap:6px}
.dot{width:7px;height:7px;border-radius:50%;background:var(--gt);animation:pulse 2s infinite}
.dot.off{background:var(--fused);animation:none}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}

.grid{display:grid;grid-template-columns:1fr 360px;gap:20px}
.card{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:16px;margin-bottom:16px}
.card-title{font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:1.5px;color:var(--muted);margin-bottom:14px;font-family:'DM Mono',monospace;display:block}

/* Plot */
.plot{position:relative;width:100%;background:#fafaf7;border-radius:6px;overflow:hidden;cursor:crosshair;border:1px solid var(--border)}
.plot svg{width:100%;height:100%}
.plot.fullscreen{position:fixed;inset:10px;z-index:3000;border:2px solid var(--border);border-radius:8px;background:#fafaf7}
body.plot-fs-active{overflow:hidden}
.grid-line{stroke:var(--border);stroke-width:.5;opacity:.4}
.grid-line.major{stroke-width:1;opacity:.6}
.axis-label{fill:var(--muted);font-size:10px;font-family:'DM Mono',monospace}

.legend{display:flex;flex-wrap:wrap;gap:16px;margin-top:14px}
.legend-item{display:flex;align-items:center;gap:6px;font-size:12px;font-family:'DM Mono',monospace}
.legend-dot{width:10px;height:10px;border-radius:2px}

/* Algo cards */
.algo-card{display:grid;grid-template-columns:auto 1fr auto;gap:10px;align-items:center;padding:12px;border-radius:6px;border:1px solid var(--border);margin-bottom:8px;border-left:3px solid}
.algo-icon{width:34px;height:34px;border-radius:4px;display:flex;align-items:center;justify-content:center;font-weight:600;font-size:11px;font-family:'DM Mono',monospace}
.algo-info h4{font-size:13px;font-weight:600;margin-bottom:1px}
.algo-info .coords{font-family:'DM Mono',monospace;font-size:11px;color:var(--text2)}
.algo-err{text-align:right}
.algo-err .val{font-family:'DM Mono',monospace;font-size:16px;font-weight:600}
.algo-err .lbl{font-size:9px;color:var(--muted);text-transform:uppercase;letter-spacing:.5px}

/* Distance bars */
.dist-bar{padding:10px 12px;border-radius:6px;border:1px solid var(--border);margin-bottom:8px}
.dist-header{display:flex;justify-content:space-between;align-items:center;margin-bottom:6px}
.dist-name{font-size:12px;font-weight:600;font-family:'DM Mono',monospace}
.dist-val{font-family:'DM Mono',monospace;font-size:14px;font-weight:500}
.dist-raw{font-size:10px;color:var(--muted);font-family:'DM Mono',monospace}
.bar-bg{height:4px;background:#e7e5e0;border-radius:2px}
.bar-fill{height:100%;border-radius:2px;transition:width .3s}

/* Ground truth */
.gt-panel{background:#f0fdf4;border:1px solid #bbf7d0}
.gt-inputs{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:10px}
.input-group label{font-size:10px;text-transform:uppercase;letter-spacing:.5px;color:var(--text2);display:block;margin-bottom:3px;font-family:'DM Mono',monospace}
.input-group input,.input-group select{width:100%;background:#fff;border:1px solid var(--border);border-radius:4px;padding:8px 10px;font-family:'DM Mono',monospace;font-size:13px}
.input-group input:focus,.input-group select:focus{outline:none;border-color:var(--gt)}
.gt-btns{display:flex;gap:6px}
.btn{flex:1;padding:8px 14px;border:none;border-radius:4px;font-family:'Instrument Sans',sans-serif;font-size:12px;font-weight:600;cursor:pointer;text-align:center}
.btn-primary{background:#1c1917;color:#fff}
.btn-primary:hover{background:#292524}
.btn-secondary{background:#fff;color:var(--text);border:1px solid var(--border)}
.btn-record{background:#dc2626;color:#fff}
.btn-record:hover{background:#b91c1c}
.btn-record.recording{animation:rec-pulse 1s infinite}
@keyframes rec-pulse{0%,100%{opacity:1}50%{opacity:.6}}
.btn-export{background:#16a34a;color:#fff}
.btn-export:hover{background:#15803d}
.btn:disabled{opacity:.4;cursor:not-allowed}

/* Stats */
table{width:100%;border-collapse:collapse;font-size:12px}
th{text-align:left;padding:6px 8px;color:var(--muted);font-weight:500;border-bottom:1px solid var(--border);font-family:'DM Mono',monospace;font-size:10px;text-transform:uppercase;letter-spacing:.5px}
td{padding:6px 8px;font-family:'DM Mono',monospace;border-bottom:1px solid var(--border)}

/* Debug */
.debug{padding:10px;background:#fafaf7;border-radius:4px;font-family:'DM Mono',monospace;font-size:11px;border:1px solid var(--border);margin-top:12px}
.debug-title{font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:1px;margin-bottom:6px}

.hint{position:absolute;bottom:8px;left:50%;transform:translateX(-50%);font-size:10px;color:var(--text2);background:rgba(255,255,255,.9);padding:3px 10px;border-radius:3px;border:1px solid var(--border);font-family:'DM Mono',monospace}
.tooltip{position:absolute;background:#fff;border:1px solid var(--border);border-radius:4px;padding:6px 10px;font-size:11px;font-family:'DM Mono',monospace;pointer-events:none;z-index:100;opacity:0;transition:opacity .15s}
.tooltip.show{opacity:1}

/* Experiment panel */
.exp-panel{background:#fef3c7;border:1px solid #fbbf24}
.exp-point{display:flex;align-items:center;gap:8px;padding:6px 8px;border-radius:4px;margin-bottom:4px;font-family:'DM Mono',monospace;font-size:11px;border:1px solid var(--border);background:#fff;cursor:pointer;transition:background .15s}
.exp-point:hover{background:#fafaf7}
.exp-point.active{border-color:#dc2626;background:#fef2f2}
.exp-point.done{border-color:#16a34a;background:#f0fdf4}
.exp-point .dot-status{width:8px;height:8px;border-radius:50%;flex-shrink:0}
.exp-point .dot-status.pending{background:var(--muted)}
.exp-point .dot-status.done{background:var(--gt)}
.exp-point .dot-status.recording{background:#dc2626;animation:pulse 1s infinite}
.exp-point .zone-tag{font-size:9px;padding:1px 5px;border-radius:2px;background:#f5f5f0;color:var(--text2)}
.exp-progress{height:4px;background:#e7e5e0;border-radius:2px;margin:10px 0}
.exp-progress-fill{height:100%;background:#dc2626;border-radius:2px;transition:width .3s}
.exp-summary{font-family:'DM Mono',monospace;font-size:11px;color:var(--text2);margin-top:8px}
.exp-export{display:flex;gap:6px;margin-top:12px}
.mini-row{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:8px}
.ab-metric{display:flex;justify-content:space-between;font-family:'DM Mono',monospace;font-size:12px;padding:6px 0;border-bottom:1px solid var(--border)}
.ab-metric:last-child{border-bottom:none}
.ab-good{color:#16a34a}.ab-bad{color:#dc2626}.ab-neutral{color:var(--text2)}
.heatmap-legend{margin-top:8px;font-family:'DM Mono',monospace;font-size:10px;color:var(--text2)}
.pillar-row{display:grid;grid-template-columns:28px 1fr 1fr;gap:6px;align-items:center;margin-bottom:6px}

/* Collapsible sections */
.section-toggle{display:flex;justify-content:space-between;align-items:center;cursor:pointer;user-select:none}
.section-toggle .arrow{transition:transform .2s;font-size:10px;color:var(--muted)}
.section-toggle .arrow.collapsed{transform:rotate(-90deg)}
.section-body{overflow:hidden;transition:max-height .3s ease}
.section-body.collapsed{max-height:0 !important}

@media(max-width:1100px){.grid{grid-template-columns:1fr}}
</style>
</head>
<body>
<div class="container">
<header>
  <div class="logo">
    <div class="logo-icon">UWB</div>
    <div><h1>UWB Indoor Positioning</h1><span>EE4002D Real-Time Dashboard</span></div>
  </div>
  <div class="badges">
    <div class="badge badge-mode" id="modeBadge">2D</div>
    <div class="badge badge-status"><div class="dot" id="statusDot"></div><span id="statusText">Connecting</span></div>
  </div>
</header>

<div class="grid">
  <!-- LEFT COLUMN: Plot + Distances -->
  <div>
    <div class="card">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:14px">
        <span class="card-title" style="margin-bottom:0">Position Plot</span>
        <div style="display:flex;align-items:center;gap:8px">
          <button class="btn btn-secondary" id="plotFsBtn" style="padding:5px 10px;flex:0 0 auto">Fullscreen</button>
          <span id="updateRate" style="font-family:'DM Mono';font-size:11px;color:var(--muted)">0 Hz</span>
        </div>
      </div>
      <div class="mini-row" style="margin-bottom:10px">
        <div class="input-group">
          <label style="margin-bottom:2px">Heatmap</label>
          <select id="heatmapAlgo">
            <option value="FUSED">Fused</option>
            <option value="TRI">Trilateration</option>
            <option value="WCL">WCL</option>
            <option value="BCCP">BCCP</option>
          </select>
        </div>
        <div class="input-group">
          <label style="margin-bottom:2px">Metric</label>
          <select id="heatmapMetric">
            <option value="rmse">RMSE</option>
            <option value="mean">Mean</option>
            <option value="max">Max</option>
          </select>
        </div>
      </div>
      <div style="display:flex;align-items:center;gap:6px;margin-bottom:10px;font-family:'DM Mono',monospace;font-size:11px">
        <input type="checkbox" id="heatmapToggle" checked>
        <label for="heatmapToggle">Show Heatmap</label>
      </div>
      <div class="plot" id="posPlot">
        <svg id="plotSvg" preserveAspectRatio="xMidYMid meet"></svg>
        <div class="tooltip" id="tooltip"></div>
        <div class="hint">click to set ground truth</div>
      </div>
      <div class="heatmap-legend" id="heatmapLegend">Heatmap inactive</div>
      <div class="legend">
        <div class="legend-item"><div class="legend-dot" style="background:var(--wcl)"></div>WCL</div>
        <div class="legend-item"><div class="legend-dot" style="background:var(--tri)"></div>Trilateration</div>
        <div class="legend-item"><div class="legend-dot" style="background:var(--bccp)"></div>BCCP</div>
        <div class="legend-item"><div class="legend-dot" style="background:var(--fused)"></div>Fused</div>
        <div class="legend-item"><div class="legend-dot" style="background:var(--gt)"></div>Ground Truth</div>
        <div class="legend-item"><div class="legend-dot" style="background:#64748b"></div>Pillar</div>
        <div class="legend-item"><div class="legend-dot" style="background:var(--muted);border-radius:50%"></div>Test Pt (pending)</div>
        <div class="legend-item"><div class="legend-dot" style="background:var(--gt);border-radius:50%"></div>Test Pt (done)</div>
      </div>
    </div>
    <div class="card">
      <span class="card-title">UWB Distances (Time of Flight)</span>
      <div id="distBars"></div>
      <div class="debug">
        <div class="debug-title">Distance Debug</div>
        <div id="debugInfo">Waiting for data...</div>
      </div>
    </div>
  </div>

  <!-- RIGHT COLUMN: Panels -->
  <div>

    <!-- Experiment Panel -->
    <div class="card exp-panel">
      <div class="section-toggle" onclick="toggleSection('expBody','expArrow')">
        <span class="card-title" style="margin-bottom:0">Experiment Mode</span>
        <span class="arrow" id="expArrow">&#9660;</span>
      </div>
      <div class="section-body" id="expBody" style="max-height:800px">

        <div style="margin-top:12px">
          <div class="gt-inputs">
            <div class="input-group">
              <label>Test Point</label>
              <select id="expPointSelect"></select>
            </div>
            <div class="input-group">
              <label>Samples</label>
              <input type="number" id="expNumSamples" value="100" min="10" max="1000" step="10">
            </div>
          </div>

          <div class="gt-btns" style="margin-bottom:8px">
            <button class="btn btn-record" id="expRecordBtn" onclick="toggleRecord()">Record</button>
            <button class="btn btn-secondary" id="expClearBtn" onclick="clearPointResult()">Clear Pt</button>
          </div>

          <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:8px">
            <div class="input-group">
              <label>Point ID</label>
              <input type="text" id="expEditPointId" placeholder="C1" style="font-family:'DM Mono',monospace">
            </div>
            <div class="input-group">
              <label>Zone</label>
              <select id="expEditZone">
                <option value="Centre">Centre</option>
                <option value="Boundary">Boundary</option>
                <option value="Corner">Corner</option>
              </select>
            </div>
            <div class="input-group">
              <label>GT X</label>
              <input type="number" id="expEditX" step="0.01">
            </div>
            <div class="input-group">
              <label>GT Y</label>
              <input type="number" id="expEditY" step="0.01">
            </div>
          </div>

          <div class="gt-btns" style="margin-bottom:10px">
            <button class="btn btn-secondary" id="expLoadBtn">Load Selected</button>
            <button class="btn btn-primary" id="expSaveBtn">Save Point</button>
            <button class="btn btn-secondary" id="expDeletePointBtn">Delete Point</button>
          </div>

          <div class="exp-progress"><div class="exp-progress-fill" id="expProgressBar" style="width:0%"></div></div>
          <div class="exp-summary" id="expSummary">0 / 0 points recorded</div>
        </div>

        <div id="expPointList" style="margin-top:10px;max-height:200px;overflow-y:auto"></div>

        <div class="exp-export">
          <button class="btn btn-export" onclick="exportCSV()" id="expCsvBtn" disabled>Export CSV</button>
          <button class="btn btn-export" onclick="exportJSON()" id="expJsonBtn" disabled>Export JSON</button>
        </div>
      </div>
    </div>

    <!-- Algorithm Positions -->
    <div class="card">
      <span class="card-title">Algorithm Positions</span>
      <div id="algoCards"></div>
    </div>

    <!-- Ground Truth -->
    <div class="card gt-panel">
      <span class="card-title">Ground Truth</span>
      <div class="gt-inputs">
        <div class="input-group"><label>X (m)</label><input type="number" id="gtX" step="0.01"></div>
        <div class="input-group"><label>Y (m)</label><input type="number" id="gtY" step="0.01" value="0"></div>
      </div>
      <div class="gt-btns">
        <button class="btn btn-primary" id="setGtBtn">Set GT</button>
        <button class="btn btn-secondary" id="clearGtBtn">Clear</button>
        <button class="btn btn-secondary" id="resetBtn">Reset Stats</button>
      </div>
    </div>

    <!-- Error Statistics -->
    <div class="card">
      <span class="card-title">Error Statistics</span>
      <table><thead><tr><th>Algo</th><th>RMSE</th><th>Mean</th><th>N</th></tr></thead>
      <tbody id="statsBody"></tbody></table>
    </div>

    <!-- Fusion A/B -->
    <div class="card">
      <span class="card-title">Fusion A/B</span>
      <div class="input-group" style="margin-top:10px">
        <label>Fused Output Mode</label>
        <select id="fusionModeSwitch">
          <option value="fixed">Fixed Fusion</option>
          <option value="adaptive">Adaptive Fusion</option>
        </select>
      </div>
      <div class="ab-metric"><span>Fixed RMSE</span><span id="abFixedRmse">--</span></div>
      <div class="ab-metric"><span>Adaptive RMSE</span><span id="abAdaptiveRmse">--</span></div>
      <div class="ab-metric"><span>Delta (Adaptive - Fixed)</span><span id="abDeltaRmse" class="ab-neutral">--</span></div>
    </div>

    <!-- Configuration -->
    <div class="card">
      <div class="section-toggle" onclick="toggleSection('cfgBody','cfgArrow')">
        <span class="card-title" style="margin-bottom:0">Configuration</span>
        <span class="arrow" id="cfgArrow">&#9660;</span>
      </div>
      <div class="section-body" id="cfgBody" style="max-height:600px">
        <div style="margin-top:12px">
          <div class="gt-inputs" style="margin-bottom:8px">
            <div class="input-group"><label>Mode</label>
              <select id="cfgMode">
                <option value="1D">1D (2 anchors)</option>
                <option value="2D">2D (3+ anchors)</option>
              </select>
            </div>
            <div class="input-group"><label>Room Width (m)</label><input type="number" id="cfgWidth" step="0.1"></div>
          </div>
          <div class="gt-inputs" style="margin-bottom:8px">
            <div class="input-group"><label>Room Height (m)</label><input type="number" id="cfgHeight" step="0.1"></div>
            <div class="input-group"><label>Line Length (1D)</label><input type="number" id="cfgLen" step="0.1"></div>
          </div>
          <div class="gt-inputs" style="margin-bottom:8px">
            <div class="input-group"><label>Fusion Mode</label>
              <select id="cfgFusionMode">
                <option value="fixed">Fixed Fusion</option>
                <option value="adaptive">Adaptive Fusion</option>
              </select>
            </div>
            <div class="input-group"><label>Pillar Model</label>
              <select id="cfgPillarEnabled">
                <option value="off">Off</option>
                <option value="on">On</option>
              </select>
            </div>
          </div>
          <div id="pillarConfig" style="margin-bottom:8px"></div>
          <div id="anchorConfig" style="margin-bottom:8px"></div>
          <button class="btn btn-primary" id="applyBtn" style="width:100%;margin-top:8px">Apply Config</button>
        </div>
      </div>
    </div>

  </div>
</div>
</div>

<script>
// =====================================================================
// CONSTANTS & STATE
// =====================================================================
const C={WCL:'#2563eb',TRI:'#ea580c',BCCP:'#7c3aed',FUSED:'#dc2626',GT:'#22c55e',A1:'#0891b2',A2:'#7c3aed',A3:'#db2777',A4:'#ca8a04'};
const ALGOS=[{id:'WCL',name:'Weighted Centroid'},{id:'TRI',name:'Trilateration'},{id:'BCCP',name:'Barycentric'},{id:'FUSED',name:'Fused'}];
const PAD=50;

let mode='2D',lineLen=6,roomW=8,roomH=8,anchors={},gt=null,curData=null;
let updCount=0,lastUpdTime=Date.now();
let expEditorDirty=false;
let fusionMode='fixed';
let pillarEnabled=false,pillarVertices=[];

// =====================================================================
// COORDINATE HELPERS
// =====================================================================
function dimW(){return mode==='1D'?lineLen:roomW;}
function dimH(){return mode==='1D'?1.5:roomH;}
function plotW(){return dimW()*100+2*PAD;}
function plotH(){return dimH()*100+2*PAD;}
function sx(){return(plotW()-2*PAD)/dimW();}
function sy(){return(plotH()-2*PAD)/dimH();}

function m2px(x,y){
  if(mode==='1D')return{x:PAD+x*sx(),y:plotH()/2};
  return{x:PAD+x*sx(),y:plotH()-PAD-y*sy()};
}
function px2m(px,py){
  if(mode==='1D')return{x:(px-PAD)/sx(),y:0};
  return{x:(px-PAD)/sx(),y:(plotH()-PAD-py)/sy()};
}

// =====================================================================
// COLLAPSIBLE SECTIONS
// =====================================================================
function toggleSection(bodyId,arrowId){
  const b=document.getElementById(bodyId);
  const a=document.getElementById(arrowId);
  b.classList.toggle('collapsed');
  a.classList.toggle('collapsed');
}

function handlePlotFullscreenChange(){
  const plot=document.getElementById('posPlot');
  const btn=document.getElementById('plotFsBtn');
  const isFs=document.fullscreenElement===plot;
  if(plot){
    plot.classList.toggle('fullscreen',isFs);
    document.body.classList.toggle('plot-fs-active',isFs);
  }
  if(btn){
    btn.textContent=isFs?'Exit Fullscreen':'Fullscreen';
  }
  setTimeout(()=>{initPlot();updateUI();},40);
}

async function togglePlotFullscreen(){
  const plot=document.getElementById('posPlot');
  if(!plot)return;
  try{
    if(document.fullscreenElement===plot){
      await document.exitFullscreen();
    }else if(!document.fullscreenElement){
      await plot.requestFullscreen();
    }else{
      await document.exitFullscreen();
      await plot.requestFullscreen();
    }
  }catch(err){
    console.error('Fullscreen error',err);
  }
}

// =====================================================================
// PLOT RENDERING
// =====================================================================
function initPlot(){
  const svg=document.getElementById('plotSvg');svg.innerHTML='';
  const W=plotW(),H=plotH();
  svg.setAttribute('viewBox',`0 0 ${W} ${H}`);
  document.getElementById('posPlot').style.aspectRatio=`${W}/${H}`;

  const defs=document.createElementNS('http://www.w3.org/2000/svg','defs');
  defs.innerHTML=`<filter id="glow"><feGaussianBlur stdDeviation="3" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>`;
  svg.appendChild(defs);

  const g=document.createElementNS('http://www.w3.org/2000/svg','g');

  if(mode==='1D'){
    const y=H/2;
    let l=document.createElementNS('http://www.w3.org/2000/svg','line');
    l.setAttribute('x1',PAD);l.setAttribute('y1',y);l.setAttribute('x2',W-PAD);l.setAttribute('y2',y);
    l.setAttribute('stroke','#c0c0b8');l.setAttribute('stroke-width','2');g.appendChild(l);
    for(let x=0;x<=lineLen;x+=0.5){
      const px=PAD+x*sx();const major=x===Math.round(x);
      let t=document.createElementNS('http://www.w3.org/2000/svg','line');
      t.setAttribute('x1',px);t.setAttribute('y1',y-(major?10:5));
      t.setAttribute('x2',px);t.setAttribute('y2',y+(major?10:5));
      t.setAttribute('stroke','#a8a29e');t.setAttribute('stroke-width',major?'1.5':'0.5');
      g.appendChild(t);
      if(major){
        let lb=document.createElementNS('http://www.w3.org/2000/svg','text');
        lb.setAttribute('x',px);lb.setAttribute('y',y+28);
        lb.setAttribute('class','axis-label');lb.setAttribute('text-anchor','middle');
        lb.textContent=x+'m';g.appendChild(lb);
      }
    }
  } else {
    for(let x=0;x<=dimW();x++){
      const p=m2px(x,0);
      let l=document.createElementNS('http://www.w3.org/2000/svg','line');
      l.setAttribute('x1',p.x);l.setAttribute('y1',PAD);l.setAttribute('x2',p.x);l.setAttribute('y2',H-PAD);
      l.setAttribute('class',x%2===0?'grid-line major':'grid-line');g.appendChild(l);
      if(x%2===0){let t=document.createElementNS('http://www.w3.org/2000/svg','text');
        t.setAttribute('x',p.x);t.setAttribute('y',H-PAD+18);t.setAttribute('class','axis-label');
        t.setAttribute('text-anchor','middle');t.textContent=x+'m';g.appendChild(t);}
    }
    for(let y=0;y<=dimH();y++){
      const p=m2px(0,y);
      let l=document.createElementNS('http://www.w3.org/2000/svg','line');
      l.setAttribute('x1',PAD);l.setAttribute('y1',p.y);l.setAttribute('x2',W-PAD);l.setAttribute('y2',p.y);
      l.setAttribute('class',y%2===0?'grid-line major':'grid-line');g.appendChild(l);
      if(y%2===0){let t=document.createElementNS('http://www.w3.org/2000/svg','text');
        t.setAttribute('x',PAD-8);t.setAttribute('y',p.y+4);t.setAttribute('class','axis-label');
        t.setAttribute('text-anchor','end');t.textContent=y+'m';g.appendChild(t);}
    }
  }
  svg.appendChild(g);
  ['heatmap','pillar','testPoints','circles','positions','anchorMarkers'].forEach(id=>{
    const el=document.createElementNS('http://www.w3.org/2000/svg','g');
    el.id=id;svg.appendChild(el);
  });
  drawPillar();
  drawAnchors();
}

function drawPillar(){
  const g=document.getElementById('pillar');
  if(!g)return;
  g.innerHTML='';
  if(!pillarEnabled||!pillarVertices||pillarVertices.length<4)return;

  const pts=pillarVertices.map(v=>m2px(v[0],v[1]));
  const polygon=document.createElementNS('http://www.w3.org/2000/svg','polygon');
  polygon.setAttribute('points',pts.map(p=>`${p.x},${p.y}`).join(' '));
  polygon.setAttribute('fill','rgba(100,116,139,0.25)');
  polygon.setAttribute('stroke','#475569');
  polygon.setAttribute('stroke-width','2');
  g.appendChild(polygon);

  const label=document.createElementNS('http://www.w3.org/2000/svg','text');
  const cx=pts.reduce((s,p)=>s+p.x,0)/pts.length;
  const cy=pts.reduce((s,p)=>s+p.y,0)/pts.length;
  label.setAttribute('x',cx);
  label.setAttribute('y',cy+4);
  label.setAttribute('text-anchor','middle');
  label.setAttribute('font-size','10');
  label.setAttribute('font-family','DM Mono, monospace');
  label.setAttribute('fill','#334155');
  label.textContent='PILLAR';
  g.appendChild(label);
}

function heatColor(value,minV,maxV){
  if(value==null||Number.isNaN(value))return 'rgba(148,163,184,0.35)';
  const span=Math.max(1e-6,maxV-minV);
  const t=Math.max(0,Math.min(1,(value-minV)/span));
  const r=Math.round(34 + t*210);
  const g=Math.round(197 - t*150);
  const b=Math.round(94 - t*70);
  return `rgba(${r},${g},${b},0.45)`;
}

function drawHeatmap(expData){
  const g=document.getElementById('heatmap');
  if(!g)return;
  g.innerHTML='';
  const legend=document.getElementById('heatmapLegend');
  if(!legend)return;

  const enabled=document.getElementById('heatmapToggle')?.checked;
  const algo=document.getElementById('heatmapAlgo')?.value||'FUSED';
  const metric=document.getElementById('heatmapMetric')?.value||'rmse';
  if(!enabled||!expData||!expData.points){
    legend.textContent='Heatmap inactive';
    return;
  }

  const points=Object.values(expData.points||{}).filter(pt=>pt.recorded&&pt.stats&&pt.stats[algo]&&pt.stats[algo][metric]!=null);
  if(points.length===0){
    legend.textContent='Heatmap: no recorded points for selected metric';
    return;
  }

  const values=points.map(pt=>Number(pt.stats[algo][metric]));
  const minV=Math.min(...values),maxV=Math.max(...values);
  points.forEach(pt=>{
    const val=Number(pt.stats[algo][metric]);
    const p=m2px(pt.gt_x,pt.gt_y);

    const circle=document.createElementNS('http://www.w3.org/2000/svg','circle');
    circle.setAttribute('cx',p.x);circle.setAttribute('cy',p.y);
    circle.setAttribute('r',mode==='1D'?'16':'18');
    circle.setAttribute('fill',heatColor(val,minV,maxV));
    circle.setAttribute('stroke','rgba(28,25,23,0.25)');
    circle.setAttribute('stroke-width','1');
    g.appendChild(circle);

    const txt=document.createElementNS('http://www.w3.org/2000/svg','text');
    txt.setAttribute('x',p.x);txt.setAttribute('y',p.y+3);
    txt.setAttribute('text-anchor','middle');
    txt.setAttribute('font-size','8');
    txt.setAttribute('font-family','DM Mono, monospace');
    txt.setAttribute('fill','#1c1917');
    txt.textContent=val.toFixed(2);
    g.appendChild(txt);
  });
  legend.textContent=`Heatmap: ${algo} ${metric.toUpperCase()} (min ${minV.toFixed(3)} m, max ${maxV.toFixed(3)} m)`;
}

function drawAnchors(){
  const g=document.getElementById('anchorMarkers');g.innerHTML='';
  Object.entries(anchors).forEach(([id,pos])=>{
    const p=m2px(pos[0],pos[1]);const c=C[id]||'#888';
    let gl=document.createElementNS('http://www.w3.org/2000/svg','circle');
    gl.setAttribute('cx',p.x);gl.setAttribute('cy',p.y);gl.setAttribute('r','18');
    gl.setAttribute('fill',c);gl.setAttribute('opacity','0.15');g.appendChild(gl);
    let ci=document.createElementNS('http://www.w3.org/2000/svg','circle');
    ci.setAttribute('cx',p.x);ci.setAttribute('cy',p.y);ci.setAttribute('r','11');
    ci.setAttribute('fill',c);g.appendChild(ci);
    let t=document.createElementNS('http://www.w3.org/2000/svg','text');
    t.setAttribute('x',p.x);t.setAttribute('y',p.y+4);t.setAttribute('text-anchor','middle');
    t.setAttribute('fill','#fff');t.setAttribute('font-size','9');t.setAttribute('font-weight','bold');
    t.setAttribute('font-family','DM Mono, monospace');t.textContent=id;g.appendChild(t);
  });
}

function drawTestPoints(expData){
  const g=document.getElementById('testPoints');
  if(!g)return;
  g.innerHTML='';
  if(!expData||!expData.points)return;

  Object.values(expData.points).forEach(pt=>{
    const p=m2px(pt.gt_x,pt.gt_y);
    const done=pt.recorded;
    const isActive=expData.current_point===pt.point_id;
    const color=isActive?'#dc2626':done?'#16a34a':'#a8a29e';

    // Outer ring
    let outer=document.createElementNS('http://www.w3.org/2000/svg','circle');
    outer.setAttribute('cx',p.x);outer.setAttribute('cy',p.y);outer.setAttribute('r','8');
    outer.setAttribute('fill','none');outer.setAttribute('stroke',color);
    outer.setAttribute('stroke-width',isActive?'2.5':'1.5');
    outer.setAttribute('stroke-dasharray',done?'':'3,2');
    g.appendChild(outer);

    // Inner dot
    let inner=document.createElementNS('http://www.w3.org/2000/svg','circle');
    inner.setAttribute('cx',p.x);inner.setAttribute('cy',p.y);inner.setAttribute('r','3');
    inner.setAttribute('fill',color);
    g.appendChild(inner);

    // Label
    let lbl=document.createElementNS('http://www.w3.org/2000/svg','text');
    lbl.setAttribute('x',p.x);lbl.setAttribute('y',p.y-12);
    lbl.setAttribute('text-anchor','middle');lbl.setAttribute('fill',color);
    lbl.setAttribute('font-size','9');lbl.setAttribute('font-weight','600');
    lbl.setAttribute('font-family','DM Mono, monospace');
    lbl.textContent=pt.point_id;
    g.appendChild(lbl);
  });
}

function updateFusionABUI(ab){
  const f=document.getElementById('abFixedRmse');
  const a=document.getElementById('abAdaptiveRmse');
  const d=document.getElementById('abDeltaRmse');
  if(!f||!a||!d)return;
  if(!ab){
    f.textContent='--';a.textContent='--';d.textContent='--';d.className='ab-neutral';
    return;
  }
  f.textContent=ab.fixed&&ab.fixed.rmse!=null?ab.fixed.rmse.toFixed(3):'--';
  a.textContent=ab.adaptive&&ab.adaptive.rmse!=null?ab.adaptive.rmse.toFixed(3):'--';
  if(ab.delta_rmse==null){
    d.textContent='--';
    d.className='ab-neutral';
  }else{
    d.textContent=(ab.delta_rmse>=0?'+':'')+ab.delta_rmse.toFixed(3);
    d.className=ab.delta_rmse<0?'ab-good':ab.delta_rmse>0?'ab-bad':'ab-neutral';
  }
  const sw=document.getElementById('fusionModeSwitch');
  if(sw&&document.activeElement!==sw){
    sw.value=ab.mode||'fixed';
  }
}

async function setFusionMode(newMode){
  const modeVal=(newMode==='adaptive')?'adaptive':'fixed';
  const r=await fetch('/api/config',{
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({fusion_mode:modeVal})
  });
  if(!r.ok){
    const body=await r.text();
    alert('Failed to set fusion mode: '+body);
    return;
  }
  fusionMode=modeVal;
  const cfgSel=document.getElementById('cfgFusionMode');
  if(cfgSel)cfgSel.value=fusionMode;
}

function drawCircles(dists){
  const g=document.getElementById('circles');g.innerHTML='';
  if(!dists)return;
  Object.entries(dists).forEach(([id,d])=>{
    if(!anchors[id])return;
    const p=m2px(anchors[id][0],anchors[id][1]);
    const r=d*sx();
    let c=document.createElementNS('http://www.w3.org/2000/svg','circle');
    c.setAttribute('cx',p.x);c.setAttribute('cy',p.y);c.setAttribute('r',r);
    c.setAttribute('fill','none');c.setAttribute('stroke',C[id]||'#888');
    c.setAttribute('stroke-width','1.5');c.setAttribute('stroke-dasharray','6,4');
    c.setAttribute('opacity','0.35');g.appendChild(c);
  });
}

function drawPositions(pos){
  const g=document.getElementById('positions');g.innerHTML='';
  if(gt){
    const p=m2px(gt.x,gt.y);const s=12;
    [[-s,-s,s,s],[s,-s,-s,s]].forEach(([x1,y1,x2,y2])=>{
      let l=document.createElementNS('http://www.w3.org/2000/svg','line');
      l.setAttribute('x1',p.x+x1);l.setAttribute('y1',p.y+y1);
      l.setAttribute('x2',p.x+x2);l.setAttribute('y2',p.y+y2);
      l.setAttribute('stroke',C.GT);l.setAttribute('stroke-width','3');g.appendChild(l);
    });
  }
  if(!pos)return;
  ['WCL','TRI','BCCP','FUSED'].forEach(a=>{
    const r=pos[a];if(!r||!r.position)return;
    const p=m2px(r.position.x,r.position.y);
    let gl=document.createElementNS('http://www.w3.org/2000/svg','circle');
    gl.setAttribute('cx',p.x);gl.setAttribute('cy',p.y);gl.setAttribute('r','12');
    gl.setAttribute('fill',C[a]);gl.setAttribute('opacity','0.25');
    gl.setAttribute('filter','url(#glow)');g.appendChild(gl);
    let ci=document.createElementNS('http://www.w3.org/2000/svg','circle');
    ci.setAttribute('cx',p.x);ci.setAttribute('cy',p.y);
    ci.setAttribute('r',a==='FUSED'?'9':'6');
    ci.setAttribute('fill',C[a]);g.appendChild(ci);
  });
}

// =====================================================================
// SIDEBAR COMPONENTS
// =====================================================================
function initAlgoCards(){
  const c=document.getElementById('algoCards');c.innerHTML='';
  ALGOS.forEach(a=>{
    c.innerHTML+=`<div class="algo-card" style="border-left-color:${C[a.id]}">
      <div class="algo-icon" style="background:${C[a.id]}15;color:${C[a.id]}">${a.id.slice(0,3)}</div>
      <div class="algo-info"><h4>${a.name}</h4><div class="coords" id="c-${a.id}">X: --.--</div></div>
      <div class="algo-err"><div class="val" id="e-${a.id}" style="color:${C[a.id]}">--</div><div class="lbl">Error(m)</div></div>
    </div>`;
  });
}

function initDistBars(){
  const c=document.getElementById('distBars');c.innerHTML='';
  Object.keys(anchors).forEach(id=>{
    c.innerHTML+=`<div class="dist-bar">
      <div class="dist-header"><span class="dist-name" style="color:${C[id]||'#888'}">${id}</span>
      <div><span class="dist-val" id="dv-${id}">-- m</span>
      <div class="dist-raw" id="dr-${id}">raw: --</div></div></div>
      <div class="bar-bg"><div class="bar-fill" id="db-${id}" style="background:${C[id]||'#888'};width:0%"></div></div>
    </div>`;
  });
}

// =====================================================================
// EXPERIMENT UI
// =====================================================================
function normZone(v){
  const z=String(v||'').trim().toLowerCase();
  if(z==='center'||z==='centre')return 'Centre';
  if(z==='boundary')return 'Boundary';
  if(z==='corner')return 'Corner';
  return 'Centre';
}

function syncPointEditor(expData, includePointId=false){
  if(!expData||!expData.points)return;
  const pts=expData.points||{};
  const sel=document.getElementById('expPointSelect');
  if(!sel.value&&sel.options.length>0)sel.value=sel.options[0].value;
  const pt=pts[sel.value];
  if(!pt)return;
  if(includePointId){
    document.getElementById('expEditPointId').value=pt.point_id;
  }
  document.getElementById('expEditZone').value=normZone(pt.zone);
  document.getElementById('expEditX').value=pt.gt_x;
  document.getElementById('expEditY').value=pt.gt_y;
  expEditorDirty=false;
}

function isPointEditorBusy(){
  const active=document.activeElement;
  if(!active)return false;
  const ids=['expEditPointId','expEditZone','expEditX','expEditY'];
  return ids.includes(active.id);
}

async function refreshExperimentStatus(){
  const r=await fetch('/api/experiment/status');
  if(!r.ok)return;
  const exp=await r.json();
  if(!curData)curData={experiment:exp,positions:null,statistics:null};
  curData.experiment=exp;
  updateExpUI(exp);
}

async function savePointFromEditor(){
  const point_id=document.getElementById('expEditPointId').value.trim();
  const zone=normZone(document.getElementById('expEditZone').value);
  const gt_x=parseFloat(document.getElementById('expEditX').value);
  const gt_y=parseFloat(document.getElementById('expEditY').value);
  if(!point_id){alert('Point ID is required');return;}
  if(Number.isNaN(gt_x)||Number.isNaN(gt_y)){alert('GT coordinates must be numbers');return;}

  const r=await fetch('/api/experiment/points/add',{
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({point_id,zone,gt_x,gt_y})
  });
  if(!r.ok){
    const body=await r.text();
    alert('Failed to save point: '+body);
    return;
  }
  expEditorDirty=false;
  await refreshExperimentStatus();
  document.getElementById('expPointSelect').value=point_id;
  syncPointEditor(curData&&curData.experiment, false);
}

async function deleteSelectedPoint(){
  const sel=document.getElementById('expPointSelect');
  const pid=sel.value||document.getElementById('expEditPointId').value.trim();
  if(!pid)return;
  if(!confirm(`Delete test point ${pid}?`))return;
  const r=await fetch(`/api/experiment/points/${pid}`,{method:'DELETE'});
  if(!r.ok){
    const body=await r.text();
    alert('Failed to delete point: '+body);
    return;
  }
  await refreshExperimentStatus();
}

function updateExpUI(expData){
  if(!expData)return;

  // Update point selector
  const sel=document.getElementById('expPointSelect');
  const curVal=sel.value;
  const pts=expData.points||{};
  const order={Centre:0,Boundary:1,Corner:2};
  const sorted=Object.values(pts).sort((a,b)=>(order[a.zone]||9)-(order[b.zone]||9)||a.point_id.localeCompare(b.point_id));
  const sig=sorted.map(p=>`${p.point_id}:${p.zone}:${p.gt_x}:${p.gt_y}`).join('|');
  if(sel.dataset.sig!==sig){
    sel.innerHTML='';
    sorted.forEach(pt=>{
      const opt=document.createElement('option');
      opt.value=pt.point_id;
      opt.textContent=`${pt.point_id} (${pt.zone}) — ${pt.gt_x}, ${pt.gt_y}`;
      sel.appendChild(opt);
    });
    if(curVal)sel.value=curVal;
    if(!sel.value&&sel.options.length>0)sel.value=sel.options[0].value;
    sel.dataset.sig=sig;
  }

  // Update record button
  const btn=document.getElementById('expRecordBtn');
  if(expData.recording){
    btn.textContent='Stop';
    btn.classList.add('recording');
  }else{
    btn.textContent='Record';
    btn.classList.remove('recording');
  }

  // Progress bar
  const pct=expData.target>0?(expData.progress/expData.target*100):0;
  document.getElementById('expProgressBar').style.width=pct+'%';

  // Summary
  document.getElementById('expSummary').textContent=
    `${expData.completed} / ${expData.total} points recorded`;

  // Enable export if we have results
  document.getElementById('expCsvBtn').disabled=expData.completed===0;
  document.getElementById('expJsonBtn').disabled=expData.completed===0;

  // Point list
  const list=document.getElementById('expPointList');list.innerHTML='';
  sorted.forEach(pt=>{
    const isActive=expData.current_point===pt.point_id;
    const cls=isActive?'active':pt.recorded?'done':'';
    const dotCls=isActive?'recording':pt.recorded?'done':'pending';
    let statsHtml='';
    if(pt.recorded&&pt.stats){
      const f=pt.stats.FUSED;
      if(f&&f.rmse!=null)statsHtml=`<span style="margin-left:auto;color:var(--fused);font-weight:600">RMSE:${f.rmse.toFixed(3)}</span>`;
    }
    const div=document.createElement('div');
    div.className='exp-point '+cls;
    div.innerHTML=`<div class="dot-status ${dotCls}"></div>
      <span style="font-weight:600;min-width:24px">${pt.point_id}</span>
      <span class="zone-tag">${pt.zone}</span>
      <span style="color:var(--text2)">(${pt.gt_x},${pt.gt_y})</span>
      ${statsHtml}`;
    div.addEventListener('click',()=>{sel.value=pt.point_id;});
    list.appendChild(div);
  });

  // Draw heatmap + test points on plot
  drawHeatmap(expData);
  drawTestPoints(expData);
}

async function toggleRecord(){
  const expData=curData&&curData.experiment;
  if(expData&&expData.recording){
    await fetch('/api/experiment/stop',{method:'POST'});
  }else{
    const pid=document.getElementById('expPointSelect').value;
    const n=parseInt(document.getElementById('expNumSamples').value)||100;
    if(!pid)return;
    await fetch('/api/experiment/start',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({point_id:pid,num_samples:n})
    });
  }
}

async function clearPointResult(){
  const pid=document.getElementById('expPointSelect').value;
  if(!pid)return;
  if(!confirm(`Clear recorded data for ${pid}?`))return;
  await fetch(`/api/experiment/results/${pid}`,{method:'DELETE'});
}

function exportCSV(){
  window.open('/api/experiment/export/csv','_blank');
}
function exportJSON(){
  window.open('/api/experiment/export/json','_blank');
}

// =====================================================================
// MAIN UPDATE LOOP
// =====================================================================
function updateUI(){
  if(!curData)return;
  const pos=curData.positions;
  const dists=pos&&pos.WCL?pos.WCL.distances:null;

  drawCircles(dists);
  drawPositions(pos);

  // Update ground truth from stream
  if(curData.ground_truth)gt=curData.ground_truth;

  // Algo cards
  if(pos)Object.entries(pos).forEach(([a,r])=>{
    const ce=document.getElementById('c-'+a);
    const ee=document.getElementById('e-'+a);
    if(ce&&r&&r.position){
      ce.textContent=mode==='1D'?`X: ${r.position.x.toFixed(3)}m`:`X:${r.position.x.toFixed(2)} Y:${r.position.y.toFixed(2)}`;
      ee.textContent=r.error!=null?r.error.toFixed(3):'--';
    }
  });

  // Distance bars
  if(dists){
    const maxD=mode==='1D'?lineLen:Math.max(roomW,roomH);
    Object.entries(dists).forEach(([id,d])=>{
      const v=document.getElementById('dv-'+id);
      const b=document.getElementById('db-'+id);
      const raw=document.getElementById('dr-'+id);
      if(v)v.textContent=d.toFixed(3)+' m';
      if(b)b.style.width=Math.min(100,d/maxD*100)+'%';
      if(raw&&curData.raw_distances&&curData.raw_distances[id])
        raw.textContent='raw: '+curData.raw_distances[id].toFixed(3)+'m';
    });
  }

  // Debug
  const dbg=document.getElementById('debugInfo');
  if(dists&&Object.keys(dists).length>=2){
    let html='';
    Object.entries(dists).forEach(([id,d])=>{
      html+=`<span style="color:${C[id]||'#888'}">${id}</span>:${d.toFixed(3)}m &nbsp;`;
    });
    if(mode==='1D'){
      const ids=Object.keys(dists);
      if(ids.length>=2){
        const d1=dists[ids[0]],d2=dists[ids[1]];
        const sum=d1+d2;
        html+=`<br>d1+d2=${sum.toFixed(2)}m, line=${lineLen.toFixed(1)}m `;
        html+=sum>=lineLen?'<span style="color:#16a34a">✓ valid</span>':'<span style="color:#dc2626">✗ too short</span>';
      }
    }
    dbg.innerHTML=html;
  }

  // Stats table
  const tbody=document.getElementById('statsBody');tbody.innerHTML='';
  const st=curData.statistics;
  ALGOS.forEach(a=>{
    const s=st?st[a.id]:null;
    const r=document.createElement('tr');
    r.innerHTML=`<td style="color:${C[a.id]}">${a.id}</td>
      <td>${s&&s.rmse!=null?s.rmse.toFixed(3):'--'}</td>
      <td>${s&&s.mean!=null?s.mean.toFixed(3):'--'}</td>
      <td>${s?s.count:0}</td>`;
    tbody.appendChild(r);
  });

  // Experiment UI
  updateExpUI(curData.experiment);
  updateFusionABUI(curData.fusion_ab);
}

// =====================================================================
// SSE CONNECTION
// =====================================================================
function connectSSE(){
  const es=new EventSource('/api/stream');
  es.onopen=()=>{
    document.getElementById('statusDot').classList.remove('off');
    document.getElementById('statusText').textContent='Connected';
  };
  es.onmessage=e=>{
    try{
      curData=JSON.parse(e.data);
      updateUI();
      updCount++;
      const now=Date.now();
      if(now-lastUpdTime>=1000){
        document.getElementById('updateRate').textContent=updCount+' Hz';
        updCount=0;lastUpdTime=now;
      }
    }catch(err){console.error(err);}
  };
  es.onerror=()=>{
    document.getElementById('statusDot').classList.add('off');
    document.getElementById('statusText').textContent='Disconnected';
    setTimeout(()=>{es.close();connectSSE();},2000);
  };
}

// =====================================================================
// EVENTS & INIT
// =====================================================================
function setupEvents(){
  document.addEventListener('fullscreenchange',handlePlotFullscreenChange);
  document.getElementById('plotFsBtn').addEventListener('click',togglePlotFullscreen);
  document.getElementById('posPlot').addEventListener('dblclick',e=>{e.preventDefault();togglePlotFullscreen();});

  document.getElementById('posPlot').addEventListener('click',e=>{
    const svg=document.getElementById('plotSvg');
    const rect=svg.getBoundingClientRect();
    const scX=plotW()/rect.width,scY=plotH()/rect.height;
    const m=px2m((e.clientX-rect.left)*scX,(e.clientY-rect.top)*scY);
    const x=Math.max(0,Math.min(dimW(),m.x));
    const y=mode==='1D'?0:Math.max(0,Math.min(dimH(),m.y));
    document.getElementById('gtX').value=x.toFixed(2);
    document.getElementById('gtY').value=y.toFixed(2);
    setGT(x,y);
  });

  const svg=document.getElementById('plotSvg');
  const tt=document.getElementById('tooltip');
  const plot=document.getElementById('posPlot');
  svg.addEventListener('mousemove',e=>{
    const rect=svg.getBoundingClientRect();
    const m=px2m((e.clientX-rect.left)*(plotW()/rect.width),(e.clientY-rect.top)*(plotH()/rect.height));
    if(m.x>=0&&m.x<=dimW()){
      tt.textContent=mode==='1D'?`X: ${m.x.toFixed(2)}m`:`X:${m.x.toFixed(2)} Y:${m.y.toFixed(2)}`;
      tt.style.left=(e.clientX-plot.getBoundingClientRect().left+12)+'px';
      tt.style.top=(e.clientY-plot.getBoundingClientRect().top+12)+'px';
      tt.classList.add('show');
    }else tt.classList.remove('show');
  });
  svg.addEventListener('mouseleave',()=>tt.classList.remove('show'));

  document.getElementById('setGtBtn').addEventListener('click',()=>{
    const x=parseFloat(document.getElementById('gtX').value);
    const y=parseFloat(document.getElementById('gtY').value||0);
    if(!isNaN(x))setGT(x,y);
  });
  document.getElementById('clearGtBtn').addEventListener('click',()=>{
    gt=null;fetch('/api/ground-truth',{method:'DELETE'});updateUI();
  });
  document.getElementById('resetBtn').addEventListener('click',()=>fetch('/api/reset-stats',{method:'POST'}));
  document.getElementById('applyBtn').addEventListener('click',applyConfig);
  document.getElementById('fusionModeSwitch').addEventListener('change',e=>setFusionMode(e.target.value));
  ['heatmapToggle','heatmapAlgo','heatmapMetric'].forEach(id=>{
    const el=document.getElementById(id);
    if(el)el.addEventListener('change',()=>{ if(curData&&curData.experiment) updateExpUI(curData.experiment); });
  });

  document.getElementById('expPointSelect').addEventListener('change',()=>{});
  document.getElementById('expLoadBtn').addEventListener('click',()=>{expEditorDirty=false;syncPointEditor(curData&&curData.experiment, true);});
  document.getElementById('expSaveBtn').addEventListener('click',savePointFromEditor);
  document.getElementById('expDeletePointBtn').addEventListener('click',deleteSelectedPoint);

  ['expEditPointId','expEditZone','expEditX','expEditY'].forEach(id=>{
    const el=document.getElementById(id);
    if(el){
      el.addEventListener('input',()=>{expEditorDirty=true;});
      el.addEventListener('change',()=>{expEditorDirty=true;});
    }
  });
}

async function setGT(x,y){
  gt={x,y};
  await fetch('/api/ground-truth',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({x,y,auto_reset:true})});
  updateUI();
}

async function applyConfig(){
  const newMode=document.getElementById('cfgMode').value;
  const w=parseFloat(document.getElementById('cfgWidth').value);
  const h=parseFloat(document.getElementById('cfgHeight').value);
  const len=parseFloat(document.getElementById('cfgLen').value);
  const cfgFusion=document.getElementById('cfgFusionMode').value;
  const cfgPillar=document.getElementById('cfgPillarEnabled').value==='on';
  const cfg={mode:newMode,room_width:w,room_height:h,line_length:len,fusion_mode:cfgFusion,pillar_enabled:cfgPillar};

  const ap={};
  document.querySelectorAll('.anchor-pos-row').forEach(row=>{
    const id=row.dataset.anchor;
    const ax=parseFloat(row.querySelector('.anc-x').value);
    const ay=parseFloat(row.querySelector('.anc-y').value);
    if(id&&!isNaN(ax)&&!isNaN(ay)) ap[id]=[ax,ay];
  });
  if(Object.keys(ap).length>0) cfg.anchor_positions=ap;

  const vertices=[];
  document.querySelectorAll('.pillar-row').forEach(row=>{
    const px=parseFloat(row.querySelector('.pillar-x').value);
    const py=parseFloat(row.querySelector('.pillar-y').value);
    if(!Number.isNaN(px)&&!Number.isNaN(py))vertices.push([px,py]);
  });
  if(vertices.length===4)cfg.pillar_vertices=vertices;

  await fetch('/api/config',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(cfg)});
  await loadConfig();
  initPlot();initDistBars();initAlgoCards();
}

async function loadConfig(){
  try{
    const r=await fetch('/api/config');const c=await r.json();
    mode=c.mode||'2D';lineLen=c.line_length||6;roomW=c.room_width||8;roomH=c.room_height||8;
    fusionMode=(c.fusion_mode||'fixed');
    pillarEnabled=!!c.pillar_enabled;
    pillarVertices=(c.pillar_vertices||[]).map(v=>[Number(v[0]),Number(v[1])]);
    anchors={};
    if(c.anchor_positions)Object.entries(c.anchor_positions).forEach(([id,pos])=>{anchors[id]=pos;});
    if(c.ground_truth)gt=c.ground_truth;

    document.getElementById('modeBadge').textContent=mode;
    document.getElementById('cfgMode').value=mode;
    document.getElementById('cfgWidth').value=roomW;
    document.getElementById('cfgHeight').value=roomH;
    document.getElementById('cfgLen').value=lineLen;
    document.getElementById('cfgFusionMode').value=fusionMode;
    document.getElementById('fusionModeSwitch').value=fusionMode;
    document.getElementById('cfgPillarEnabled').value=pillarEnabled?'on':'off';

    const pc=document.getElementById('pillarConfig');pc.innerHTML='';
    const defaultVerts=pillarVertices.length===4?pillarVertices:[[5.8,2.8],[6.3,2.8],[6.3,4.1],[5.8,4.1]];
    defaultVerts.forEach((pos,idx)=>{
      const row=document.createElement('div');
      row.className='pillar-row';
      row.innerHTML=`<span style="font-family:'DM Mono',monospace;font-size:11px;color:var(--text2)">P${idx+1}</span>
        <input type="number" class="pillar-x" step="0.1" value="${pos[0]}" style="width:100%;padding:6px 8px;border:1px solid var(--border);border-radius:4px;font-family:'DM Mono',monospace;font-size:12px" placeholder="X">
        <input type="number" class="pillar-y" step="0.1" value="${pos[1]}" style="width:100%;padding:6px 8px;border:1px solid var(--border);border-radius:4px;font-family:'DM Mono',monospace;font-size:12px" placeholder="Y">`;
      pc.appendChild(row);
    });

    const ac=document.getElementById('anchorConfig');ac.innerHTML='';
    Object.entries(anchors).forEach(([id,pos])=>{
      const row=document.createElement('div');
      row.className='anchor-pos-row';row.dataset.anchor=id;
      row.style.cssText='display:grid;grid-template-columns:40px 1fr 1fr;gap:6px;align-items:center;margin-bottom:6px';
      row.innerHTML=`<span style="font-family:'DM Mono',monospace;font-size:12px;font-weight:600;color:${C[id]||'#888'}">${id}</span>
        <input type="number" class="anc-x" step="0.1" value="${pos[0]}" style="width:100%;padding:6px 8px;border:1px solid var(--border);border-radius:4px;font-family:'DM Mono',monospace;font-size:12px" placeholder="X">
        <input type="number" class="anc-y" step="0.1" value="${pos[1]}" style="width:100%;padding:6px 8px;border:1px solid var(--border);border-radius:4px;font-family:'DM Mono',monospace;font-size:12px" placeholder="Y">`;
      ac.appendChild(row);
    });

    if(mode==='1D'){document.getElementById('gtY').parentElement.style.display='none';}
    else{document.getElementById('gtY').parentElement.style.display='';}
  }catch(e){console.error(e);}
}

async function init(){
  await loadConfig();
  initPlot();initDistBars();initAlgoCards();
  setupEvents();connectSSE();
}
document.addEventListener('DOMContentLoaded',init);
</script>
</body></html>"""
