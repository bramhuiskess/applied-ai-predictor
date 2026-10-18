"""Measure latency and throughput of a running backend (validation for SQ2/SQ3/SQ4).

Works for any use case: valid random records are generated from the feature
schema that ``GET /models/{id}`` returns. Standard library only.

    python scripts/benchmark.py --url http://127.0.0.1:8000 --use-case machine-failure
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
import urllib.request


def call(url: str, body: bytes | None = None, content_type: str = "application/json") -> bytes:
    request = urllib.request.Request(url, data=body, headers={"content-type": content_type})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def random_features(inputs: list[dict]) -> dict:
    features = {}
    for spec in inputs:
        if spec["allowed_values"]:
            features[spec["name"]] = random.choice(spec["allowed_values"])
        else:
            low, high = spec["training_range"]
            value = random.uniform(low, high)
            features[spec["name"]] = round(value) if spec["kind"] == "integer" else round(value, 3)
    return features


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--use-case", default="machine-failure")
    parser.add_argument("--single", type=int, default=300, help="number of single-record requests")
    parser.add_argument("--batch", type=int, default=20_000, help="records in the JSONL batch")
    args = parser.parse_args()
    base = f"{args.url}/models/{args.use_case}"
    meta = json.loads(call(base))
    random.seed(0)

    # Warm up, then single-record latency (online inference).
    one = json.dumps({"records": [{"features": random_features(meta["inputs"])}]}).encode()
    for _ in range(20):
        call(f"{base}/predict", one)
    latencies = []
    for _ in range(args.single):
        body = json.dumps({"records": [{"features": random_features(meta["inputs"])}]}).encode()
        started = time.perf_counter()
        call(f"{base}/predict", body)
        latencies.append((time.perf_counter() - started) * 1000)
    latencies.sort()

    records = [{"record_id": str(i), "features": random_features(meta["inputs"])} for i in range(args.batch)]
    json_body = json.dumps({"records": records[:10_000]}).encode()
    jsonl_body = ("\n".join(json.dumps(r) for r in records) + "\n").encode()

    started = time.perf_counter()
    call(f"{base}/predict", json_body)
    json_seconds = time.perf_counter() - started

    started = time.perf_counter()
    out = call(f"{base}/predict/batch", jsonl_body, "application/x-ndjson").decode().splitlines()
    jsonl_seconds = time.perf_counter() - started
    errors = sum("errors" in json.loads(line) for line in out)

    n_json = min(args.batch, 10_000)
    print(f"## Benchmark: {args.use_case} (model v{meta['model_version']}, {meta['algorithm']}, "
          f"{len(meta['inputs'])} features)\n")
    print("| measurement | result |\n|---|---|")
    print(f"| single record, p50 latency | {statistics.median(latencies):.2f} ms |")
    print(f"| single record, p95 latency | {latencies[int(0.95 * len(latencies)) - 1]:.2f} ms |")
    print(f"| single record, p99 latency | {latencies[int(0.99 * len(latencies)) - 1]:.2f} ms |")
    print(f"| JSON request, {n_json} records | {json_seconds:.2f} s ({n_json / json_seconds:,.0f} records/s), "
          f"{len(json_body) / 1e6:.1f} MB |")
    print(f"| JSONL batch, {args.batch} records | {jsonl_seconds:.2f} s ({args.batch / jsonl_seconds:,.0f} "
          f"records/s), {len(jsonl_body) / 1e6:.1f} MB, {errors} errors |")


if __name__ == "__main__":
    main()
