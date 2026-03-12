"""
Localisation algorithms: WCL, Least Squares, BCCP, and Fused.
All algorithms work for both 1D (2 anchors) and 2D (3+ anchors).
"""

import math
from typing import Dict, Optional

from .config import config
from .models import Position


class PositioningAlgorithms:

    @staticmethod
    def _weighted_average(entries):
        if not entries:
            return None
        tw = sum(w for _, w in entries)
        if tw <= 0:
            return None
        x = sum(p.x * w for p, w in entries) / tw
        y = sum(p.y * w for p, w in entries) / tw
        x = max(0.0, min(config.room_width, x))
        y = max(0.0, min(config.room_height, y))
        return Position(x=x, y=y)

    # -----------------------------------------------------------------
    # 1D helper (shared by LS and BCCP when only 2 anchors available)
    # -----------------------------------------------------------------

    @staticmethod
    def position_1d_linear(distances: Dict[str, float]) -> Optional[Position]:
        """
        1D positioning with 2 anchors on a line.
        A1 at x=x1, A2 at x=x2, distances d1, d2.
        x = (L^2 + d1^2 - d2^2) / (2L) + x1
        """
        aids = list(config.anchor_positions.keys())
        if len(aids) < 2:
            return None
        a1, a2 = aids[0], aids[1]
        if a1 not in distances or a2 not in distances:
            return None

        d1 = distances[a1]
        d2 = distances[a2]
        x1 = config.anchor_positions[a1][0]
        x2 = config.anchor_positions[a2][0]
        L = abs(x2 - x1)
        if L < 0.01:
            return None

        x = (L ** 2 + d1 ** 2 - d2 ** 2) / (2 * L)
        x += min(x1, x2)
        return Position(x=x, y=0.0)

    # -----------------------------------------------------------------
    # Weighted Centroid Localisation (WCL)
    # -----------------------------------------------------------------

    @staticmethod
    def weighted_centroid(distances: Dict[str, float]) -> Optional[Position]:
        if len(distances) < 2:
            return None

        total_w = 0.0
        wx = 0.0
        wy = 0.0

        for aid, d in distances.items():
            if aid not in config.anchor_positions or d <= 0:
                continue
            pos = config.anchor_positions[aid]
            w = 1.0 / (d ** config.wcl_exponent)
            wx += w * pos[0]
            wy += w * pos[1]
            total_w += w

        if total_w == 0:
            return None
        return Position(x=wx / total_w, y=wy / total_w)

    # -----------------------------------------------------------------
    # Least Squares Trilateration
    # -----------------------------------------------------------------

    @staticmethod
    def least_squares_trilateration(distances: Dict[str, float]) -> Optional[Position]:
        anchors = [
            (aid, config.anchor_positions[aid], d)
            for aid, d in distances.items()
            if aid in config.anchor_positions and d > 0
        ]

        if len(anchors) < 2:
            return None
        if len(anchors) == 2:
            return PositioningAlgorithms.position_1d_linear(distances)

        # 3+ anchors: full 2D least squares
        ref_id, (x_n, y_n), r_n = anchors[-1]
        A = []
        b = []
        for i in range(len(anchors) - 1):
            _, (x_i, y_i), r_i = anchors[i]
            A.append([2 * (x_n - x_i), 2 * (y_n - y_i)])
            b.append(r_i ** 2 - r_n ** 2 - x_i ** 2 - y_i ** 2 + x_n ** 2 + y_n ** 2)

        try:
            # A^T A
            ata = [[0, 0], [0, 0]]
            atb = [0, 0]
            for idx, row in enumerate(A):
                ata[0][0] += row[0] * row[0]
                ata[0][1] += row[0] * row[1]
                ata[1][0] += row[1] * row[0]
                ata[1][1] += row[1] * row[1]
                atb[0] += row[0] * b[idx]
                atb[1] += row[1] * b[idx]

            det = ata[0][0] * ata[1][1] - ata[0][1] * ata[1][0]
            if abs(det) < 1e-10:
                return None

            x = (ata[1][1] * atb[0] - ata[0][1] * atb[1]) / det
            y = (-ata[1][0] * atb[0] + ata[0][0] * atb[1]) / det

            # Constrain to room boundaries
            x = max(0, min(config.room_width, x))
            y = max(0, min(config.room_height, y))
            return Position(x=x, y=y)
        except Exception:
            return None

    # -----------------------------------------------------------------
    # BCCP (Best Circle Crossing Point)
    # -----------------------------------------------------------------

    @staticmethod
    def bccp(distances: Dict[str, float]) -> Optional[Position]:
        anchors = [
            (aid, config.anchor_positions[aid], d)
            for aid, d in distances.items()
            if aid in config.anchor_positions and d > 0
        ]

        if len(anchors) < 2:
            return None
        if len(anchors) == 2:
            return PositioningAlgorithms.position_1d_linear(distances)
        if len(anchors) < 3:
            return None

        def circle_intersect(c1, r1, c2, r2):
            dx = c2[0] - c1[0]
            dy = c2[1] - c1[1]
            d = math.sqrt(dx * dx + dy * dy)
            if d > r1 + r2 or d < abs(r1 - r2) or d == 0:
                ratio = r1 / (r1 + r2) if (r1 + r2) > 0 else 0.5
                return [(c1[0] + ratio * dx, c1[1] + ratio * dy)]
            a = (r1 ** 2 - r2 ** 2 + d ** 2) / (2 * d)
            h = math.sqrt(max(0, r1 ** 2 - a ** 2))
            px = c1[0] + a * dx / d
            py = c1[1] + a * dy / d
            if h == 0:
                return [(px, py)]
            return [
                (px + h * dy / d, py - h * dx / d),
                (px - h * dy / d, py + h * dx / d),
            ]

        points = []
        for i in range(len(anchors)):
            for j in range(i + 1, len(anchors)):
                _, c1, r1 = anchors[i]
                _, c2, r2 = anchors[j]
                pts = circle_intersect(c1, r1, c2, r2)
                if len(pts) == 2:
                    others = [anchors[k][1] for k in range(len(anchors)) if k != i and k != j]
                    if others:
                        cx = sum(c[0] for c in others) / len(others)
                        cy = sum(c[1] for c in others) / len(others)
                        d0 = (pts[0][0] - cx) ** 2 + (pts[0][1] - cy) ** 2
                        d1 = (pts[1][0] - cx) ** 2 + (pts[1][1] - cy) ** 2
                        points.append(pts[0] if d0 < d1 else pts[1])
                    else:
                        points.append(pts[0])
                elif pts:
                    points.append(pts[0])

        if not points:
            return None
        x = sum(p[0] for p in points) / len(points)
        y = sum(p[1] for p in points) / len(points)
        x = max(0, min(config.room_width, x))
        y = max(0, min(config.room_height, y))
        return Position(x=x, y=y)

    # -----------------------------------------------------------------
    # Fused Algorithm (weighted combination)
    # -----------------------------------------------------------------

    @staticmethod
    def fused_fixed(
        wcl: Optional[Position],
        tri: Optional[Position],
        bccp: Optional[Position],
    ) -> Optional[Position]:
        """Fixed weighted fusion of available algorithm outputs."""
        entries = []
        if wcl:
            entries.append((wcl, 0.205))
        if tri:
            entries.append((tri, 0.463))
        if bccp:
            entries.append((bccp, 0.332))
        return PositioningAlgorithms._weighted_average(entries)

    @staticmethod
    def fused_adaptive(
        wcl: Optional[Position],
        tri: Optional[Position],
        bccp: Optional[Position],
    ) -> Optional[Position]:
        """Adaptive fusion using inter-algorithm agreement with prior trust."""
        candidates = {}
        if wcl:
            candidates["WCL"] = wcl
        if tri:
            candidates["TRI"] = tri
        if bccp:
            candidates["BCCP"] = bccp

        if not candidates:
            return None
        if len(candidates) == 1:
            return next(iter(candidates.values()))

        base_weights = {"WCL": 0.205, "TRI": 0.463, "BCCP": 0.332}
        eps = 1e-6
        outlier_floor = 0.15

        conf = {}
        names = list(candidates.keys())
        for name in names:
            p = candidates[name]
            dsum = 0.0
            cnt = 0
            for other_name in names:
                if other_name == name:
                    continue
                q = candidates[other_name]
                dsum += math.sqrt((p.x - q.x) ** 2 + (p.y - q.y) ** 2)
                cnt += 1
            avg_dist = dsum / cnt if cnt > 0 else 0.0
            conf[name] = 1.0 / (eps + avg_dist)

        conf_sum = sum(conf.values())
        if conf_sum <= eps:
            weights = {n: base_weights[n] for n in names}
        else:
            weights = {}
            for n in names:
                norm = conf[n] / conf_sum
                weights[n] = base_weights[n] * max(outlier_floor, norm)

        entries = [(candidates[n], weights[n]) for n in names]
        return PositioningAlgorithms._weighted_average(entries)

    @staticmethod
    def fused(
        wcl: Optional[Position],
        tri: Optional[Position],
        bccp: Optional[Position],
    ) -> Optional[Position]:
        """Backwards-compatible default fused output (fixed mode)."""
        return PositioningAlgorithms.fused_fixed(wcl, tri, bccp)
