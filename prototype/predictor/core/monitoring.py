"""Minimal monitoring: per-use-case counters and structured log lines.

Feature values are only logged when the use-case policy allows it.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import asdict, dataclass
from typing import Any

logger = logging.getLogger("predictor")


@dataclass
class UseCaseStats:
    requests: int = 0
    records_predicted: int = 0
    records_rejected: int = 0
    out_of_range_warnings: int = 0
    total_latency_ms: float = 0.0
    max_latency_ms: float = 0.0


class Monitor:
    def __init__(self) -> None:
        self._stats: dict[str, UseCaseStats] = {}
        self._lock = threading.Lock()

    def record(
        self,
        use_case_id: str,
        *,
        predicted: int,
        rejected: int,
        warnings: int,
        latency_ms: float,
        model_version: str | None,
        features: list[dict[str, Any]] | None = None,
    ) -> None:
        with self._lock:
            stats = self._stats.setdefault(use_case_id, UseCaseStats())
            stats.requests += 1
            stats.records_predicted += predicted
            stats.records_rejected += rejected
            stats.out_of_range_warnings += warnings
            stats.total_latency_ms += latency_ms
            stats.max_latency_ms = max(stats.max_latency_ms, latency_ms)
        logger.info(
            "predict use_case=%s model_version=%s predicted=%d rejected=%d warnings=%d latency_ms=%.2f",
            use_case_id, model_version, predicted, rejected, warnings, latency_ms,
        )
        if features is not None:
            logger.debug("features use_case=%s values=%s", use_case_id, features)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            out = {}
            for use_case_id, stats in self._stats.items():
                data = asdict(stats)
                data["avg_latency_ms"] = round(stats.total_latency_ms / stats.requests, 3) if stats.requests else 0.0
                data["total_latency_ms"] = round(stats.total_latency_ms, 3)
                data["max_latency_ms"] = round(stats.max_latency_ms, 3)
                out[use_case_id] = data
            return out
