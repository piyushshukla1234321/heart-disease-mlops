"""Generate realistic traffic so the Grafana dashboard has something to show.

    python scripts/send_traffic.py --n 300                    # local / docker-compose
    python scripts/send_traffic.py --url http://localhost:8080 --n 300   # Kubernetes
About 5% of requests are deliberately invalid so the error-rate panel is non-empty.
"""
import argparse
import json
import random
import time
import urllib.error
import urllib.request


def random_patient(rng):
    sick = rng.random() < 0.45
    return {
        "age": int(rng.gauss(57 if sick else 52, 8)),
        "sex": int(rng.random() < (0.8 if sick else 0.55)),
        "cp": 4 if sick and rng.random() < 0.7 else rng.choice([1, 2, 3]),
        "trestbps": round(rng.gauss(134 if sick else 129, 17)),
        "chol": round(min(max(rng.gauss(251 if sick else 242, 45), 120), 560)),
        "fbs": int(rng.random() < 0.15),
        "restecg": rng.choice([0, 1, 2]),
        "thalach": round(min(max(rng.gauss(139 if sick else 158, 20), 75), 200)),
        "exang": int(rng.random() < (0.55 if sick else 0.15)),
        "oldpeak": round(max(rng.gauss(1.6 if sick else 0.6, 1.0), 0), 1),
        "slope": rng.choice([2, 2, 3] if sick else [1, 1, 2]),
        "ca": rng.choice([1, 2, 3] if sick else [0, 0, 0, 1]),
        "thal": 7 if sick and rng.random() < 0.65 else 3,
    }


def post(url, payload):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--delay", type=float, default=0.1, help="seconds between requests")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    counts = {}
    for i in range(args.n):
        patient = random_patient(rng)
        if rng.random() < 0.05:
            patient["age"] = -1  # invalid on purpose -> 422
        status = post(f"{args.url.rstrip('/')}/predict", patient)
        counts[status] = counts.get(status, 0) + 1
        if (i + 1) % 50 == 0:
            print(f"{i + 1}/{args.n} sent, status counts so far: {counts}")
        time.sleep(args.delay)
    print("done:", counts)


if __name__ == "__main__":
    main()
