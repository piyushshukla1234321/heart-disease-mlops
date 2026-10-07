"""End-to-end check of a running API. Standard library only, so it runs anywhere.

    python scripts/smoke_test.py                          # http://localhost:8000
    python scripts/smoke_test.py --url http://localhost:8080
Exits non-zero on any failure (used by CI after `docker run`).
"""
import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

SAMPLE = json.loads((Path(__file__).parent / "sample_request.json").read_text())


def call(url, payload=None, timeout=5):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


def wait_until_ready(base, seconds):
    deadline = time.time() + seconds
    while time.time() < deadline:
        try:
            if call(f"{base}/health")[0] == 200:
                return True
        except (urllib.error.URLError, ConnectionError, OSError):
            pass
        time.sleep(2)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--wait", type=int, default=60, help="seconds to wait for /health")
    args = ap.parse_args()
    base = args.url.rstrip("/")
    failures = []

    if not wait_until_ready(base, args.wait):
        print(f"FAIL: {base}/health never returned 200")
        sys.exit(1)
    print(f"OK   /health is ready at {base}")

    status, body = call(f"{base}/predict", SAMPLE)
    print(f"     POST /predict -> {status} {body}")
    if status != 200:
        failures.append("/predict did not return 200")
    else:
        out = json.loads(body)
        for key in ("prediction", "probability_disease", "confidence", "risk_band"):
            if key not in out:
                failures.append(f"missing key {key}")
        if not 0 <= out.get("probability_disease", -1) <= 1:
            failures.append("probability out of range")

    status, _ = call(f"{base}/predict", {**SAMPLE, "age": -1})
    print(f"     POST /predict (invalid age) -> {status}")
    if status != 422:
        failures.append("invalid input was not rejected with 422")

    status, body = call(f"{base}/metrics")
    if status != 200 or "api_requests_total" not in body:
        failures.append("/metrics missing api_requests_total")
    else:
        print("OK   /metrics exposes Prometheus counters")

    if failures:
        print("FAIL:", *failures, sep="\n  - ")
        sys.exit(1)
    print("ALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
