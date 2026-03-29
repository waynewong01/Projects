#!/usr/bin/env python3
"""
Synthetic BLE ranging simulator for generating error results.

Examples:
  python -m files.ble_simulator --point-x 3.25 --point-y 3.65 --samples 300 --rssi-sigma-db 6
  python -m files.ble_simulator --point-x 3.25 --point-y 3.65 --samples 300 --rssi-sigma-db 12

Rule of thumb:
  - 4 to 6 dB RSSI noise often produces roughly 1 to 2 m distance error indoors
  - 10 to 20 dB RSSI noise represents severe multipath / NLOS behaviour
"""

import argparse
import math
import random
from pathlib import Path
from typing import Dict, Tuple

from .config import config
from .engine import PositioningEngine


def distance_to_rssi(distance_m: float, tx_power_dbm: float, path_loss_exponent: float) -> float:
    distance_m = max(distance_m, 0.1)
    return tx_power_dbm - 10.0 * path_loss_exponent * math.log10(distance_m)


def rssi_to_distance(rssi_dbm: float, tx_power_dbm: float, path_loss_exponent: float) -> float:
    return 10.0 ** ((tx_power_dbm - rssi_dbm) / (10.0 * path_loss_exponent))


def simulate_anchor_measurement(
    tag_x: float,
    tag_y: float,
    anchor_xy: Tuple[float, float],
    tx_power_dbm: float,
    path_loss_exponent: float,
    rssi_sigma_db: float,
    distance_bias_m: float,
) -> Tuple[float, float, float]:
    true_distance = math.dist((tag_x, tag_y), anchor_xy)
    ideal_rssi = distance_to_rssi(true_distance, tx_power_dbm, path_loss_exponent)
    noisy_rssi = ideal_rssi + random.gauss(0.0, rssi_sigma_db)
    estimated_distance = rssi_to_distance(noisy_rssi, tx_power_dbm, path_loss_exponent)
    estimated_distance = max(0.05, estimated_distance + distance_bias_m)
    return true_distance, noisy_rssi, estimated_distance


def run_simulation(args: argparse.Namespace) -> dict:
    random.seed(args.seed)

    engine = PositioningEngine()
    engine.set_ground_truth(args.point_x, args.point_y, auto_reset=True)

    anchor_error_history = {aid: [] for aid in config.anchor_positions}
    rssi_history = {aid: [] for aid in config.anchor_positions}
    algo_error_history = {}

    for _ in range(args.samples):
        for aid, anchor_xy in config.anchor_positions.items():
            true_distance, noisy_rssi, estimated_distance = simulate_anchor_measurement(
                tag_x=args.point_x,
                tag_y=args.point_y,
                anchor_xy=anchor_xy,
                tx_power_dbm=args.tx_power_dbm,
                path_loss_exponent=args.path_loss_exponent,
                rssi_sigma_db=args.rssi_sigma_db,
                distance_bias_m=args.distance_bias_m,
            )
            anchor_error_history[aid].append(abs(estimated_distance - true_distance))
            rssi_history[aid].append(noisy_rssi)
            engine.add_distance(aid, estimated_distance, noisy_rssi)

        results = engine.compute_all()
        for algo_name, algo_result in results.items():
            if algo_name not in algo_error_history:
                algo_error_history[algo_name] = []
            if algo_result.error is not None:
                algo_error_history[algo_name].append(algo_result.error)

    algo_stats = engine.get_statistics()
    anchor_summary = {}
    for aid in config.anchor_positions:
        errors = anchor_error_history[aid]
        rssis = rssi_history[aid]
        anchor_summary[aid] = {
            "mean_distance_error_m": sum(errors) / len(errors),
            "max_distance_error_m": max(errors),
            "min_rssi_dbm": min(rssis),
            "max_rssi_dbm": max(rssis),
        }

    return {
        "samples": args.samples,
        "point": {"x": args.point_x, "y": args.point_y},
        "ble_model": {
            "tx_power_dbm": args.tx_power_dbm,
            "path_loss_exponent": args.path_loss_exponent,
            "rssi_sigma_db": args.rssi_sigma_db,
            "distance_bias_m": args.distance_bias_m,
            "seed": args.seed,
        },
        "anchor_summary": anchor_summary,
        "anchor_error_history": anchor_error_history,
        "rssi_history": rssi_history,
        "algo_error_history": algo_error_history,
        "algorithm_stats": algo_stats,
    }


def format_report(result: dict) -> str:
    lines = []
    model = result["ble_model"]
    point = result["point"]

    lines.append(
        f"Point ({point['x']:.2f}, {point['y']:.2f}), samples={result['samples']}, "
        f"RSSI sigma={model['rssi_sigma_db']:.1f} dB, bias={model['distance_bias_m']:.2f} m"
    )
    lines.append("")
    lines.append("Per-anchor ranging summary:")
    for aid, stats in result["anchor_summary"].items():
        lines.append(
            f"  {aid}: mean |distance error|={stats['mean_distance_error_m']:.3f} m, "
            f"max={stats['max_distance_error_m']:.3f} m, "
            f"RSSI span=[{stats['min_rssi_dbm']:.1f}, {stats['max_rssi_dbm']:.1f}] dBm"
        )

    lines.append("")
    lines.append("Localisation summary:")
    for algo, stats in result["algorithm_stats"].items():
        if stats["count"] == 0:
            lines.append(f"  {algo}: no result")
            continue
        lines.append(
            f"  {algo}: mean error={stats['mean']:.3f} m, rmse={stats['rmse']:.3f} m, "
            f"max={stats['max']:.3f} m"
        )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate synthetic BLE ranging errors.")
    parser.add_argument("--point-x", type=float, default=3.25)
    parser.add_argument("--point-y", type=float, default=3.65)
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument("--tx-power-dbm", type=float, default=-59.0)
    parser.add_argument("--path-loss-exponent", type=float, default=2.0)
    parser.add_argument("--rssi-sigma-db", type=float, default=6.0)
    parser.add_argument("--distance-bias-m", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--plot", action="store_true", help="Create a PNG summary plot.")
    parser.add_argument("--plot-path", type=str, default="ble_simulation_plot.png")
    return parser


def save_plot(result: dict, plot_path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    figure.suptitle("Synthetic BLE Ranging and Localisation Summary")

    ax = axes[0][0]
    anchor_ids = list(result["anchor_summary"].keys())
    mean_anchor_errors = [
        result["anchor_summary"][aid]["mean_distance_error_m"] for aid in anchor_ids
    ]
    ax.bar(anchor_ids, mean_anchor_errors, color="#4C78A8")
    ax.set_title("Mean Distance Error per Anchor")
    ax.set_ylabel("Error (m)")

    ax = axes[0][1]
    for aid, values in result["rssi_history"].items():
        ax.hist(values, bins=30, alpha=0.45, label=aid)
    ax.set_title("RSSI Distribution")
    ax.set_xlabel("RSSI (dBm)")
    ax.set_ylabel("Count")
    ax.legend()

    ax = axes[1][0]
    for aid, values in result["anchor_error_history"].items():
        ax.plot(values, label=aid, alpha=0.85)
    ax.set_title("Distance Error Across Samples")
    ax.set_xlabel("Sample")
    ax.set_ylabel("Absolute Error (m)")
    ax.legend()

    ax = axes[1][1]
    algo_names = []
    algo_means = []
    for algo, stats in result["algorithm_stats"].items():
        if stats["mean"] is None:
            continue
        algo_names.append(algo)
        algo_means.append(stats["mean"])
    ax.bar(algo_names, algo_means, color="#F58518")
    ax.set_title("Mean Localisation Error by Algorithm")
    ax.set_ylabel("Error (m)")

    figure.tight_layout()
    output_path = Path(plot_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main():
    args = build_parser().parse_args()
    result = run_simulation(args)
    print(format_report(result))
    if args.plot:
        save_plot(result, args.plot_path)
        print(f"\nPlot saved to: {Path(args.plot_path).resolve()}")


if __name__ == "__main__":
    main()
