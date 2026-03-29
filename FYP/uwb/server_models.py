"""
Request payload models for the FastAPI server.
"""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


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
