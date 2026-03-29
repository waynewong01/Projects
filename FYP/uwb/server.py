"""
FastAPI server for the UWB Indoor Positioning System dashboard.
"""

import asyncio
import json
import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, StreamingResponse

from .app_state import build_stream_payload, engine, experiment, mqtt_handler
from .config import config
from .dashboard import load_dashboard_html
from .server_models import (
    ConfigUpdatePayload,
    DistancePayload,
    ExperimentPointPayload,
    ExperimentPointsPayload,
    ExperimentStartPayload,
    GroundTruthPayload,
)

logger = logging.getLogger(__name__)

app = FastAPI(title="UWB IPS Dashboard")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _normalize_config_update(new_config: dict) -> dict:
    if "anchor_positions" in new_config:
        normalized = {}
        for anchor_id, pos in new_config["anchor_positions"].items():
            if len(pos) != 2:
                raise HTTPException(
                    status_code=422,
                    detail=f"Anchor {anchor_id} must have exactly two coordinates",
                )
            normalized[anchor_id] = [float(pos[0]), float(pos[1])]
        new_config["anchor_positions"] = normalized

    if "pillar_vertices" in new_config:
        vertices = new_config["pillar_vertices"]
        if len(vertices) != 4:
            raise HTTPException(
                status_code=422,
                detail="pillar_vertices must contain exactly 4 points",
            )

        normalized_vertices = []
        for vertex in vertices:
            if len(vertex) != 2:
                raise HTTPException(
                    status_code=422,
                    detail="Each pillar vertex must have exactly 2 coordinates",
                )
            normalized_vertices.append([float(vertex[0]), float(vertex[1])])
        new_config["pillar_vertices"] = normalized_vertices

    return new_config


@app.get("/api/stream")
async def stream():
    async def gen():
        while True:
            try:
                results = engine.compute_all()
                if experiment.recording:
                    experiment.collect_sample(results)
                yield f"data: {json.dumps(build_stream_payload(results))}\n\n"
            except Exception as exc:
                logger.error("Stream error: %s", exc)

            rate_hz = max(0.1, float(getattr(config, "update_rate_hz", 10.0) or 10.0))
            await asyncio.sleep(1.0 / rate_hz)

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


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
    engine.update_config(_normalize_config_update(payload.model_dump(exclude_none=True)))
    return {"status": "ok"}


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


@app.post("/api/distance")
async def add_distance(payload: DistancePayload):
    anchor_id = payload.anchor or payload.anchor_id
    if not anchor_id:
        raise HTTPException(status_code=422, detail="Provide 'anchor' or 'anchor_id'")

    engine.add_distance(anchor_id, float(payload.distance), float(payload.rx_power))
    return {"status": "ok"}


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
        experiment.set_test_points([point.model_dump() for point in payload.points])
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"status": "ok"}


@app.post("/api/experiment/points/add")
async def experiment_add_point(payload: ExperimentPointPayload):
    try:
        experiment.add_test_point(
            payload.point_id,
            payload.zone,
            float(payload.gt_x),
            float(payload.gt_y),
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
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
    return PlainTextResponse(
        experiment.export_csv(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=experiment_results.csv"},
    )


@app.get("/api/experiment/export/json")
async def experiment_export_json():
    return JSONResponse(
        experiment.export_json(),
        headers={"Content-Disposition": "attachment; filename=experiment_results.json"},
    )


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    return load_dashboard_html()


@app.on_event("startup")
async def startup():
    logger.info("Starting UWB IPS Dashboard (mode=%s)", config.mode)
    logger.info("Anchors: %s", config.anchor_positions)
    mqtt_handler.connect()
    logger.info("Dashboard: http://localhost:%s", config.server_port)


@app.on_event("shutdown")
async def shutdown():
    mqtt_handler.disconnect()
