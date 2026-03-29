from pathlib import Path

import pandas as pd
import numpy as np

CSV_FILE = Path(__file__).resolve().parents[1] / "data" / "calibration" / "calibration_A8.csv"

def fit_anchor(df_anchor):
    # Use median RSSI at each distance to reduce outliers
    g = df_anchor.groupby("distance_m")["rssi"].median().reset_index()

    d = g["distance_m"].to_numpy()
    rssi = g["rssi"].to_numpy()

    x = np.log10(d)
    y = rssi

    # Linear model: y = a + b*x, where a = RSSI_1m, b = -10n
    A = np.vstack([np.ones_like(x), x]).T
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]

    rssi_1m = float(a)
    n = float(-b / 10.0)

    # simple fit quality
    yhat = a + b*x
    rmse = float(np.sqrt(np.mean((y - yhat)**2)))

    return rssi_1m, n, rmse, g

def main():
    df = pd.read_csv(CSV_FILE)

    for anchor, dfa in df.groupby("anchor"):
        rssi_1m, n, rmse, table = fit_anchor(dfa)
        print(f"\nAnchor {anchor}")
        print(table.to_string(index=False))
        print(f"Estimated RSSI_1m = {rssi_1m:.2f} dBm")
        print(f"Estimated n       = {n:.3f}")
        print(f"Fit RMSE (RSSI)   = {rmse:.2f} dB")

if __name__ == "__main__":
    main()
