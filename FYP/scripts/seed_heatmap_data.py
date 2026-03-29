import json
import math
import random
import time
import urllib.request

BASE = "http://localhost:8000"
RNG = random.Random(7)


def req(method: str, path: str, payload=None):
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    r = urllib.request.Request(BASE + path, data=data, method=method, headers=headers)
    with urllib.request.urlopen(r, timeout=10) as resp:
        body = resp.read().decode("utf-8")
        if body:
            return json.loads(body)
        return {}


def seed_point(point_id: str, gt_x: float, gt_y: float, zone: str):
    zone_sigma = {
        "Centre": 0.10,
        "Boundary": 0.20,
        "Corner": 0.28,
    }
    noise_sigma = zone_sigma.get(zone, 0.16)

    req("POST", "/api/experiment/start", {"point_id": point_id, "num_samples": 80})

    anchors = {
        "A1": (0.0, 0.0),
        "A2": (8.0, 0.0),
        "A3": (0.0, 8.0),
        "A4": (8.0, 8.0),
    }

    for _ in range(90):
        for aid, (ax, ay) in anchors.items():
            d = math.sqrt((gt_x - ax) ** 2 + (gt_y - ay) ** 2)
            d += RNG.gauss(0.0, noise_sigma)
            d = max(0.05, d)
            req("POST", "/api/distance", {"anchor": aid, "distance": round(d, 4)})
        time.sleep(0.03)

    req("POST", "/api/experiment/stop")


def main():
    status = req("GET", "/api/experiment/status")
    points = list(status.get("points", {}).values())
    if not points:
        print("No test points found")
        return

    req("DELETE", "/api/experiment/results")
    for p in points:
        seed_point(p["point_id"], float(p["gt_x"]), float(p["gt_y"]), p["zone"])

    done = req("GET", "/api/experiment/status")
    print(json.dumps({
        "completed": done.get("completed"),
        "total": done.get("total"),
        "points": list(done.get("points", {}).keys()),
    }, indent=2))


if __name__ == "__main__":
    main()
