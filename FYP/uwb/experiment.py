"""
Experiment Manager â€” structured data collection for Chapter 7 evaluation.

Workflow:
  1. Define test points with spatial zones (Centre, Boundary, Corner)
  2. Walk to a point, select it, hit Record
  3. System collects N samples, computes per-algorithm stats
  4. Repeat for all points
  5. Export CSV/JSON matching Chapter 7 table format
"""

import csv
import io
import json
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from .engine import PositioningEngine, ALGO_NAMES


# -------------------------------------------------------------------------
# Data structures
# -------------------------------------------------------------------------

class Zone(str, Enum):
    CENTRE = "Centre"
    BOUNDARY = "Boundary"
    CORNER = "Corner"


@dataclass
class TestPoint:
    point_id: str       # e.g. "C1", "B2", "K1"
    zone: Zone
    gt_x: float
    gt_y: float


@dataclass
class Sample:
    """One snapshot from a compute_all() cycle during recording."""
    timestamp: float
    positions: Dict[str, Optional[dict]]   # algo -> {x, y} or None
    distances: Dict[str, float]            # anchor -> smoothed distance
    errors: Dict[str, Optional[float]]     # algo -> error (m) or None


@dataclass
class PointResult:
    """Aggregated results for a single test point."""
    point: TestPoint
    samples: List[Sample] = field(default_factory=list)
    stats: Dict[str, dict] = field(default_factory=dict)   # algo -> {mean, rmse, max, min, std, n}
    recorded_at: Optional[float] = None

    def compute_stats(self):
        """Compute per-algorithm error statistics from collected samples."""
        for algo in ALGO_NAMES:
            errors = [
                s.errors[algo] for s in self.samples
                if s.errors.get(algo) is not None
            ]
            if errors:
                sorted_errors = sorted(errors)
                n = len(errors)
                mean = sum(errors) / n
                rmse = math.sqrt(sum(e ** 2 for e in errors) / n)
                std = math.sqrt(sum((e - mean) ** 2 for e in errors) / n) if n > 1 else 0.0
                if n % 2 == 1:
                    median = sorted_errors[n // 2]
                else:
                    median = (sorted_errors[n // 2 - 1] + sorted_errors[n // 2]) / 2
                self.stats[algo] = {
                    "n": n,
                    "mean": round(mean, 4),
                    "rmse": round(rmse, 4),
                    "max": round(max(errors), 4),
                    "min": round(min(errors), 4),
                    "std": round(std, 4),
                    "median": round(median, 4),
                }
            else:
                self.stats[algo] = {"n": 0, "mean": None, "rmse": None,
                                     "max": None, "min": None, "std": None, "median": None}


# -------------------------------------------------------------------------
# Default test points for a 6.5 x 7.3 room with anchors at corners
# -------------------------------------------------------------------------

DEFAULT_TEST_POINTS = [
    # Centre zone (7 points)
    TestPoint("C1", Zone.CENTRE, 3.25, 3.65),
    TestPoint("C2", Zone.CENTRE, 2.15, 2.43),
    TestPoint("C3", Zone.CENTRE, 4.35, 2.43),
    TestPoint("C4", Zone.CENTRE, 2.15, 4.87),
    TestPoint("C5", Zone.CENTRE, 4.35, 4.87),
    TestPoint("C6", Zone.CENTRE, 2.15, 3.65),
    TestPoint("C7", Zone.CENTRE, 4.35, 3.65),

    # Boundary zone (8 points)
    TestPoint("B1", Zone.BOUNDARY, 0.00, 2.43),
    TestPoint("B2", Zone.BOUNDARY, 0.00, 4.87),
    TestPoint("B3", Zone.BOUNDARY, 6.50, 2.43),
    TestPoint("B4", Zone.BOUNDARY, 6.50, 4.87),
    TestPoint("B5", Zone.BOUNDARY, 3.25, 0.00),
    TestPoint("B6", Zone.BOUNDARY, 3.25, 7.30),
    TestPoint("B7", Zone.BOUNDARY, 0.00, 3.65),
    TestPoint("B8", Zone.BOUNDARY, 6.50, 3.65),

    # Corner zone (4 points)
    TestPoint("K1", Zone.CORNER, 1.08, 1.22),
    TestPoint("K2", Zone.CORNER, 5.42, 1.22),
    TestPoint("K3", Zone.CORNER, 1.08, 6.08),
    TestPoint("K4", Zone.CORNER, 5.42, 6.08),
]

# Experiment Manager
# -------------------------------------------------------------------------

class ExperimentManager:
    def __init__(self, engine: PositioningEngine):
        self.engine = engine
        self.test_points: Dict[str, TestPoint] = {
            tp.point_id: tp for tp in DEFAULT_TEST_POINTS
        }
        self.results: Dict[str, PointResult] = {}

        # Recording state
        self.recording: bool = False
        self.current_point_id: Optional[str] = None
        self.target_samples: int = 100
        self._current_samples: List[Sample] = []

    # -----------------------------------------------------------------
    # Test point management
    # -----------------------------------------------------------------

    @staticmethod
    def parse_zone(zone: str) -> Zone:
        """Parse zone value from API input with common aliases."""
        if zone is None:
            raise ValueError("Zone is required")
        z = str(zone).strip().lower()
        mapping = {
            "centre": Zone.CENTRE,
            "center": Zone.CENTRE,
            "boundary": Zone.BOUNDARY,
            "corner": Zone.CORNER,
        }
        if z not in mapping:
            raise ValueError("Zone must be one of: Centre, Boundary, Corner")
        return mapping[z]

    def set_test_points(self, points: List[dict]):
        """Replace test points from API. Each dict: {point_id, zone, gt_x, gt_y}."""
        new_test_points: Dict[str, TestPoint] = {}
        for p in points:
            tp = TestPoint(
                point_id=p["point_id"],
                zone=self.parse_zone(p["zone"]),
                gt_x=float(p["gt_x"]),
                gt_y=float(p["gt_y"]),
            )
            new_test_points[tp.point_id] = tp

        # Keep only still-valid results (same point id and unchanged GT/zone).
        new_results: Dict[str, PointResult] = {}
        for pid, tp in new_test_points.items():
            prev = self.results.get(pid)
            if not prev:
                continue
            prev_tp = prev.point
            if (
                prev_tp.zone == tp.zone
                and prev_tp.gt_x == tp.gt_x
                and prev_tp.gt_y == tp.gt_y
            ):
                prev.point = tp
                new_results[pid] = prev

        self.test_points = new_test_points
        self.results = new_results

        # Abort active recording if the selected point no longer exists.
        if self.current_point_id not in self.test_points:
            self.recording = False
            self.current_point_id = None
            self._current_samples = []

    def add_test_point(self, point_id: str, zone: str, gt_x: float, gt_y: float):
        """Add or update a single test point."""
        zone_enum = self.parse_zone(zone)
        new_point = TestPoint(point_id, zone_enum, gt_x, gt_y)

        # If the point changed, clear prior results because GT baseline changed.
        prev = self.test_points.get(point_id)
        if prev and (prev.zone != new_point.zone or prev.gt_x != new_point.gt_x or prev.gt_y != new_point.gt_y):
            self.results.pop(point_id, None)

        self.test_points[point_id] = new_point

    def remove_test_point(self, point_id: str):
        self.test_points.pop(point_id, None)
        self.results.pop(point_id, None)

    # -----------------------------------------------------------------
    # Recording
    # -----------------------------------------------------------------

    def start_recording(self, point_id: str, num_samples: int = 100) -> dict:
        """Begin recording samples for a test point."""
        if point_id not in self.test_points:
            return {"status": "error", "message": f"Unknown point: {point_id}"}
        if self.recording:
            return {"status": "error", "message": "Already recording"}

        tp = self.test_points[point_id]

        # Set the engine's ground truth to this test point
        self.engine.set_ground_truth(tp.gt_x, tp.gt_y, auto_reset=True)

        self.recording = True
        self.current_point_id = point_id
        self.target_samples = num_samples
        self._current_samples = []

        return {"status": "recording", "point_id": point_id, "target": num_samples}

    def collect_sample(self, results: dict):
        """Called each compute cycle while recording. Pass in compute_all() output."""
        if not self.recording:
            return

        sample = Sample(
            timestamp=time.time(),
            positions={},
            distances={},
            errors={},
        )

        if results:
            sample.distances = next(iter(results.values())).distances.copy()

        for algo_name, algo_result in results.items():
            if algo_result.position:
                sample.positions[algo_name] = {
                    "x": round(algo_result.position.x, 4),
                    "y": round(algo_result.position.y, 4),
                }
            else:
                sample.positions[algo_name] = None
            sample.errors[algo_name] = (
                round(algo_result.error, 4) if algo_result.error is not None else None
            )

        self._current_samples.append(sample)

        # Check if done
        if len(self._current_samples) >= self.target_samples:
            self._finish_recording()

    def stop_recording(self) -> dict:
        """Manually stop recording early."""
        if not self.recording:
            return {"status": "error", "message": "Not recording"}
        point_id = self.current_point_id
        self._finish_recording()
        return {"status": "ok", "point_id": point_id}

    def _finish_recording(self):
        """Finalise the current recording."""
        point_id = self.current_point_id
        tp = self.test_points[point_id]

        result = PointResult(point=tp, samples=self._current_samples)
        result.recorded_at = time.time()
        result.compute_stats()
        self.results[point_id] = result

        self.recording = False
        self.current_point_id = None
        self._current_samples = []

    def clear_point_result(self, point_id: str):
        """Delete recorded data for a point so it can be re-recorded."""
        self.results.pop(point_id, None)

    def clear_all_results(self):
        self.results.clear()

    # -----------------------------------------------------------------
    # Status
    # -----------------------------------------------------------------

    def get_status(self) -> dict:
        points_status = {}
        for pid, tp in self.test_points.items():
            recorded = pid in self.results
            points_status[pid] = {
                "point_id": pid,
                "zone": tp.zone.value,
                "gt_x": tp.gt_x,
                "gt_y": tp.gt_y,
                "recorded": recorded,
                "stats": self.results[pid].stats if recorded else None,
            }

        return {
            "recording": self.recording,
            "current_point": self.current_point_id,
            "progress": len(self._current_samples) if self.recording else 0,
            "target": self.target_samples if self.recording else 0,
            "points": points_status,
            "completed": sum(1 for pid in self.test_points if pid in self.results),
            "total": len(self.test_points),
        }

    # -----------------------------------------------------------------
    # Export â€” CSV (Chapter 7 tables)
    # -----------------------------------------------------------------

    def export_csv(self) -> str:
        """
        Export results as CSV matching Chapter 7 Table 3 + Table 4 format.
        Section 1: Per-point errors by algorithm and zone.
        Section 2: Overall summary statistics.
        """
        output = io.StringIO()
        writer = csv.writer(output)

        # ---- Table 3: Per-Point Results ----
        writer.writerow(["# TABLE 3: Per-Point Localisation Error (metres)"])
        writer.writerow([
            "Point", "Zone", "GT_X", "GT_Y",
            "LS_Error", "WCL_Error", "BCCP_Error", "Fused_Error",
            "LS_RMSE", "WCL_RMSE", "BCCP_RMSE", "Fused_RMSE",
            "N_Samples",
        ])

        # Group by zone order: Centre, Boundary, Corner
        zone_order = [Zone.CENTRE, Zone.BOUNDARY, Zone.CORNER]
        sorted_points = sorted(
            self.results.values(),
            key=lambda r: (zone_order.index(r.point.zone), r.point.point_id),
        )

        for pr in sorted_points:
            tp = pr.point
            row = [tp.point_id, tp.zone.value, tp.gt_x, tp.gt_y]
            # Mean errors
            for algo in ["TRI", "WCL", "BCCP", "FUSED"]:
                s = pr.stats.get(algo, {})
                row.append(s.get("mean", ""))
            # RMSE
            for algo in ["TRI", "WCL", "BCCP", "FUSED"]:
                s = pr.stats.get(algo, {})
                row.append(s.get("rmse", ""))
            row.append(len(pr.samples))
            writer.writerow(row)

        writer.writerow([])

        # ---- Table 4: Overall Summary ----
        writer.writerow(["# TABLE 4: Overall Localisation Performance"])
        writer.writerow(["Algorithm", "Mean_Error(m)", "RMSE(m)", "Max_Error(m)", "Std_Dev(m)", "N_Points"])

        for algo, label in [("TRI", "Least Squares"), ("WCL", "WCL"),
                             ("BCCP", "BCCP"), ("FUSED", "Fused (Proposed)")]:
            all_means = []
            all_errors_sq = []
            all_max = []
            for pr in self.results.values():
                s = pr.stats.get(algo, {})
                if s.get("mean") is not None:
                    all_means.append(s["mean"])
                    all_errors_sq.append(s["rmse"] ** 2)
                    all_max.append(s["max"])

            if all_means:
                n = len(all_means)
                overall_mean = sum(all_means) / n
                overall_rmse = math.sqrt(sum(all_errors_sq) / n)
                overall_max = max(all_max)
                overall_std = (
                    math.sqrt(sum((m - overall_mean) ** 2 for m in all_means) / n)
                    if n > 1 else 0.0
                )
                writer.writerow([
                    label,
                    round(overall_mean, 4),
                    round(overall_rmse, 4),
                    round(overall_max, 4),
                    round(overall_std, 4),
                    n,
                ])
            else:
                writer.writerow([label, "", "", "", "", 0])

        return output.getvalue()

    # -----------------------------------------------------------------
    # Export â€” JSON (full raw data)
    # -----------------------------------------------------------------

    def export_json(self) -> dict:
        """Export full experiment data including raw samples."""
        from .config import config as cfg
        data = {
            "experiment_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "room": {
                "width": cfg.room_width,
                "height": cfg.room_height,
                "anchors": {k: list(v) for k, v in cfg.anchor_positions.items()},
            },
            "points": {},
        }

        for pid, pr in self.results.items():
            tp = pr.point
            data["points"][pid] = {
                "point_id": pid,
                "zone": tp.zone.value,
                "gt": {"x": tp.gt_x, "y": tp.gt_y},
                "recorded_at": pr.recorded_at,
                "stats": pr.stats,
                "samples": [
                    {
                        "timestamp": s.timestamp,
                        "positions": s.positions,
                        "distances": s.distances,
                        "errors": s.errors,
                    }
                    for s in pr.samples
                ],
            }

        return data
