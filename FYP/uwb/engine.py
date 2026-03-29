"""
Positioning Engine — manages distance history, applies smoothing,
runs all four algorithms, and tracks error statistics.
"""

import math
import time
from collections import deque
from typing import Dict, List, Optional

from .config import config
from .models import Position, GroundTruth, AlgorithmResult
from .algorithms import PositioningAlgorithms


ALGO_NAMES = ["WCL", "TRI", "BCCP", "FUSED"]
FUSION_VARIANTS = ["FUSED_FIXED", "FUSED_ADAPTIVE"]


class PositioningEngine:
    def __init__(self):
        self.distance_history: Dict[str, deque] = {
            aid: deque(maxlen=config.distance_window_size)
            for aid in config.anchor_positions
        }
        self.smoothed_positions: Dict[str, Optional[Position]] = {
            k: None for k in (ALGO_NAMES + FUSION_VARIANTS)
        }
        self.ground_truth: Optional[GroundTruth] = None
        self.error_history: Dict[str, List[float]] = {
            k: [] for k in ALGO_NAMES
        }
        self.fusion_error_history: Dict[str, List[float]] = {
            k: [] for k in FUSION_VARIANTS
        }
        self.last_fusion_errors: Dict[str, Optional[float]] = {
            k: None for k in FUSION_VARIANTS
        }
        self.current_distances: Dict[str, float] = {}
        self.raw_distances: Dict[str, float] = {}
        self.rx_powers: Dict[str, float] = {}
        self.last_update: Dict[str, float] = {}

    # -----------------------------------------------------------------
    # Pillar geometry helpers
    # -----------------------------------------------------------------

    @staticmethod
    def _pillar_vertices() -> List[tuple]:
        vertices = getattr(config, "pillar_vertices", []) or []
        normalized = []
        for vertex in vertices:
            if len(vertex) != 2:
                continue
            normalized.append((float(vertex[0]), float(vertex[1])))
        return normalized

    @staticmethod
    def _point_in_polygon(x: float, y: float, polygon: List[tuple]) -> bool:
        inside = False
        count = len(polygon)
        if count < 3:
            return False
        j = count - 1
        for i in range(count):
            xi, yi = polygon[i]
            xj, yj = polygon[j]
            intersects = ((yi > y) != (yj > y)) and (
                x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi
            )
            if intersects:
                inside = not inside
            j = i
        return inside

    @staticmethod
    def _nearest_point_on_segment(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> tuple:
        abx = bx - ax
        aby = by - ay
        ab2 = abx * abx + aby * aby
        if ab2 <= 1e-12:
            return ax, ay
        t = ((px - ax) * abx + (py - ay) * aby) / ab2
        t = max(0.0, min(1.0, t))
        return ax + t * abx, ay + t * aby

    @classmethod
    def _distance_to_polygon(cls, x: float, y: float, polygon: List[tuple]) -> tuple:
        best_dist = float("inf")
        best_point = (x, y)
        for i in range(len(polygon)):
            ax, ay = polygon[i]
            bx, by = polygon[(i + 1) % len(polygon)]
            qx, qy = cls._nearest_point_on_segment(x, y, ax, ay, bx, by)
            dist = math.sqrt((qx - x) ** 2 + (qy - y) ** 2)
            if dist < best_dist:
                best_dist = dist
                best_point = (qx, qy)
        return best_dist, best_point

    @staticmethod
    def _orientation(ax: float, ay: float, bx: float, by: float, cx: float, cy: float) -> float:
        return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)

    @staticmethod
    def _on_segment(ax: float, ay: float, bx: float, by: float, px: float, py: float, eps: float = 1e-9) -> bool:
        return (
            min(ax, bx) - eps <= px <= max(ax, bx) + eps
            and min(ay, by) - eps <= py <= max(ay, by) + eps
        )

    @classmethod
    def _segments_intersect(cls, p1: tuple, p2: tuple, q1: tuple, q2: tuple) -> bool:
        eps = 1e-9
        o1 = cls._orientation(p1[0], p1[1], p2[0], p2[1], q1[0], q1[1])
        o2 = cls._orientation(p1[0], p1[1], p2[0], p2[1], q2[0], q2[1])
        o3 = cls._orientation(q1[0], q1[1], q2[0], q2[1], p1[0], p1[1])
        o4 = cls._orientation(q1[0], q1[1], q2[0], q2[1], p2[0], p2[1])

        if abs(o1) <= eps and cls._on_segment(p1[0], p1[1], p2[0], p2[1], q1[0], q1[1], eps):
            return True
        if abs(o2) <= eps and cls._on_segment(p1[0], p1[1], p2[0], p2[1], q2[0], q2[1], eps):
            return True
        if abs(o3) <= eps and cls._on_segment(q1[0], q1[1], q2[0], q2[1], p1[0], p1[1], eps):
            return True
        if abs(o4) <= eps and cls._on_segment(q1[0], q1[1], q2[0], q2[1], p2[0], p2[1], eps):
            return True

        return ((o1 > eps) != (o2 > eps)) and ((o3 > eps) != (o4 > eps))

    @classmethod
    def _segment_intersects_polygon(cls, p1: tuple, p2: tuple, polygon: List[tuple]) -> bool:
        if cls._point_in_polygon(p1[0], p1[1], polygon) or cls._point_in_polygon(p2[0], p2[1], polygon):
            return True
        for i in range(len(polygon)):
            q1 = polygon[i]
            q2 = polygon[(i + 1) % len(polygon)]
            if cls._segments_intersect(p1, p2, q1, q2):
                return True
        return False

    def apply_pillar_constraint(self, pos: Optional[Position], distances: Dict[str, float]) -> Optional[Position]:
        if pos is None or not getattr(config, "pillar_enabled", False):
            return pos

        polygon = self._pillar_vertices()
        if len(polygon) < 4:
            return pos

        if self._point_in_polygon(pos.x, pos.y, polygon):
            _, nearest = self._distance_to_polygon(pos.x, pos.y, polygon)
            return Position(x=nearest[0], y=nearest[1])

        near_threshold = float(getattr(config, "pillar_near_threshold_m", 0.8))
        soft_strength = float(getattr(config, "pillar_soft_strength", 0.7))
        distance_to_poly, nearest = self._distance_to_polygon(pos.x, pos.y, polygon)
        if distance_to_poly > near_threshold:
            return pos

        blocked_count = 0
        for aid in distances:
            anchor = config.anchor_positions.get(aid)
            if not anchor:
                continue
            if self._segment_intersects_polygon(anchor, (pos.x, pos.y), polygon):
                blocked_count += 1

        if blocked_count == 0:
            return pos

        blend = min(1.0, soft_strength * (1.0 - distance_to_poly / max(near_threshold, 1e-6)))
        corrected_x = (1.0 - blend) * pos.x + blend * nearest[0]
        corrected_y = (1.0 - blend) * pos.y + blend * nearest[1]
        return Position(x=corrected_x, y=corrected_y)

    # -----------------------------------------------------------------
    # Distance input
    # -----------------------------------------------------------------

    def add_distance(self, anchor_id: str, distance: float, rx_power: float = 0.0):
        """Add a UWB distance measurement from an anchor."""
        if anchor_id not in self.distance_history:
            self.distance_history[anchor_id] = deque(maxlen=config.distance_window_size)
        self.distance_history[anchor_id].append(distance)
        self.raw_distances[anchor_id] = distance
        self.rx_powers[anchor_id] = rx_power
        self.last_update[anchor_id] = time.time()

    # -----------------------------------------------------------------
    # Smoothing
    # -----------------------------------------------------------------

    def get_smoothed_distances(self) -> Dict[str, float]:
        """Sliding window average with trimmed mean."""
        distances = {}
        now = time.time()

        for aid, history in self.distance_history.items():
            if not history:
                continue
            if aid in self.last_update and now - self.last_update[aid] > config.stale_anchor_timeout_s:
                continue

            values = list(history)
            if len(values) >= 3:
                s = sorted(values)
                trimmed = s[1:-1]
                distances[aid] = sum(trimmed) / len(trimmed)
            else:
                distances[aid] = sum(values) / len(values)

        self.current_distances = distances
        return distances

    def smooth_position(self, name: str, new_pos: Optional[Position]) -> Optional[Position]:
        """Apply EMA smoothing to a position estimate."""
        if new_pos is None:
            return self.smoothed_positions.get(name)
        prev = self.smoothed_positions.get(name)
        alpha = config.smoothing_factors.get(name, 0.3)
        if prev is None:
            self.smoothed_positions[name] = new_pos
            return new_pos
        smoothed = Position(
            x=alpha * new_pos.x + (1 - alpha) * prev.x,
            y=alpha * new_pos.y + (1 - alpha) * prev.y,
        )
        self.smoothed_positions[name] = smoothed
        return smoothed

    def clear_smoothed_positions(self):
        for name in self.smoothed_positions:
            self.smoothed_positions[name] = None

    # -----------------------------------------------------------------
    # Error computation
    # -----------------------------------------------------------------

    def calc_error(self, pos: Optional[Position]) -> Optional[float]:
        if pos is None or self.ground_truth is None:
            return None
        return math.sqrt(
            (pos.x - self.ground_truth.x) ** 2
            + (pos.y - self.ground_truth.y) ** 2
        )

    # -----------------------------------------------------------------
    # Run all algorithms
    # -----------------------------------------------------------------

    def compute_all(self) -> Dict[str, AlgorithmResult]:
        distances = self.get_smoothed_distances()
        results: Dict[str, AlgorithmResult] = {}

        algo_fns = [
            ("WCL", PositioningAlgorithms.weighted_centroid),
            ("TRI", PositioningAlgorithms.least_squares_trilateration),
            ("BCCP", PositioningAlgorithms.bccp),
        ]

        for name, algo_fn in algo_fns:
            raw = algo_fn(distances)
            raw = self.apply_pillar_constraint(raw, distances)
            smoothed = self.smooth_position(name, raw)
            err = self.calc_error(smoothed)
            if err is not None:
                self.error_history[name].append(err)
            results[name] = AlgorithmResult(name, smoothed, distances.copy(), err)

        # Compute both fusion variants for A/B comparison.
        fused_fixed_raw = PositioningAlgorithms.fused_fixed(
            results["WCL"].position,
            results["TRI"].position,
            results["BCCP"].position,
        )
        fused_adaptive_raw = PositioningAlgorithms.fused_adaptive(
            results["WCL"].position,
            results["TRI"].position,
            results["BCCP"].position,
        )

        fused_fixed_raw = self.apply_pillar_constraint(fused_fixed_raw, distances)
        fused_adaptive_raw = self.apply_pillar_constraint(fused_adaptive_raw, distances)

        fused_fixed_smoothed = self.smooth_position("FUSED_FIXED", fused_fixed_raw)
        fused_adaptive_smoothed = self.smooth_position("FUSED_ADAPTIVE", fused_adaptive_raw)

        fixed_err = self.calc_error(fused_fixed_smoothed)
        adaptive_err = self.calc_error(fused_adaptive_smoothed)

        self.last_fusion_errors["FUSED_FIXED"] = fixed_err
        self.last_fusion_errors["FUSED_ADAPTIVE"] = adaptive_err
        if fixed_err is not None:
            self.fusion_error_history["FUSED_FIXED"].append(fixed_err)
        if adaptive_err is not None:
            self.fusion_error_history["FUSED_ADAPTIVE"].append(adaptive_err)

        fusion_mode = str(getattr(config, "fusion_mode", "fixed")).lower()
        selected = "FUSED_ADAPTIVE" if fusion_mode == "adaptive" else "FUSED_FIXED"
        selected_raw = fused_adaptive_smoothed if selected == "FUSED_ADAPTIVE" else fused_fixed_smoothed
        fused_smoothed = self.smooth_position("FUSED", selected_raw)
        fused_err = self.calc_error(fused_smoothed)
        if fused_err is not None:
            self.error_history["FUSED"].append(fused_err)
        results["FUSED"] = AlgorithmResult("FUSED", fused_smoothed, distances.copy(), fused_err)

        return results

    # -----------------------------------------------------------------
    # Ground truth & statistics
    # -----------------------------------------------------------------

    def set_ground_truth(self, x: float, y: float, auto_reset: bool = True):
        if auto_reset:
            self.clear_errors(reset_smoothing=True)
        self.ground_truth = GroundTruth(x=x, y=y)

    def clear_ground_truth(self):
        self.ground_truth = None

    def clear_errors(self, reset_smoothing: bool = False):
        for k in self.error_history:
            self.error_history[k] = []
        for k in self.fusion_error_history:
            self.fusion_error_history[k] = []
        for k in self.last_fusion_errors:
            self.last_fusion_errors[k] = None
        if reset_smoothing:
            self.clear_smoothed_positions()

    def get_fusion_ab_metrics(self) -> dict:
        def _agg(name: str) -> dict:
            vals = self.fusion_error_history.get(name, [])
            if not vals:
                return {"count": 0, "mean": None, "rmse": None, "last": self.last_fusion_errors.get(name)}
            n = len(vals)
            rmse = math.sqrt(sum(v * v for v in vals) / n)
            return {
                "count": n,
                "mean": sum(vals) / n,
                "rmse": rmse,
                "last": self.last_fusion_errors.get(name),
            }

        fixed = _agg("FUSED_FIXED")
        adaptive = _agg("FUSED_ADAPTIVE")
        delta_rmse = None
        if fixed["rmse"] is not None and adaptive["rmse"] is not None:
            delta_rmse = adaptive["rmse"] - fixed["rmse"]
        return {
            "mode": str(getattr(config, "fusion_mode", "fixed")).lower(),
            "fixed": fixed,
            "adaptive": adaptive,
            "delta_rmse": delta_rmse,
        }

    def get_statistics(self) -> dict:
        stats = {}
        for name, errors in self.error_history.items():
            if errors:
                n = len(errors)
                stats[name] = {
                    "count": n,
                    "mean": sum(errors) / n,
                    "rmse": math.sqrt(sum(e ** 2 for e in errors) / n),
                    "min": min(errors),
                    "max": max(errors),
                }
            else:
                stats[name] = {"count": 0, "mean": None, "rmse": None, "min": None, "max": None}
        return stats

    # -----------------------------------------------------------------
    # Runtime config updates
    # -----------------------------------------------------------------

    def update_config(self, new_config: dict):
        if "mode" in new_config:
            config.mode = new_config["mode"]
        if "line_length" in new_config:
            config.line_length = float(new_config["line_length"])
        if "room_width" in new_config:
            config.room_width = float(new_config["room_width"])
        if "room_height" in new_config:
            config.room_height = float(new_config["room_height"])
        if "anchor_positions" in new_config:
            for aid, pos in new_config["anchor_positions"].items():
                config.anchor_positions[aid] = tuple(pos)
            for aid in config.anchor_positions:
                if aid not in self.distance_history:
                    self.distance_history[aid] = deque(maxlen=config.distance_window_size)
        if "smoothing_factors" in new_config:
            config.smoothing_factors.update(new_config["smoothing_factors"])
        if "distance_window_size" in new_config:
            config.distance_window_size = max(1, int(new_config["distance_window_size"]))
            # Rebuild histories with new window length while preserving recent values.
            for aid, history in list(self.distance_history.items()):
                self.distance_history[aid] = deque(history, maxlen=config.distance_window_size)
        if "stale_anchor_timeout_s" in new_config:
            config.stale_anchor_timeout_s = max(0.1, float(new_config["stale_anchor_timeout_s"]))
        if "update_rate_hz" in new_config:
            config.update_rate_hz = max(0.1, float(new_config["update_rate_hz"]))
        if "fusion_mode" in new_config:
            mode = str(new_config["fusion_mode"]).strip().lower()
            config.fusion_mode = "adaptive" if mode == "adaptive" else "fixed"
        if "pillar_enabled" in new_config:
            config.pillar_enabled = bool(new_config["pillar_enabled"])
        if "pillar_vertices" in new_config:
            vertices = []
            for vertex in new_config["pillar_vertices"]:
                if len(vertex) != 2:
                    continue
                vertices.append((float(vertex[0]), float(vertex[1])))
            if len(vertices) >= 4:
                config.pillar_vertices = vertices[:4]
