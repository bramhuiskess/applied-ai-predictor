"""Ports: the interfaces through which the core talks to the outside world.

The core only depends on these protocols. Concrete technology (ONNX Runtime,
the filesystem, CSV files) lives in ``predictor.adapters`` (Cockburn, 2005).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


@dataclass
class RunnerOutput:
    """Normalised model output, independent of the model technology.

    ``values`` holds one prediction per row (a number, a class label or
    ``"anomaly"``/``"normal"``). ``probabilities`` is filled for classifiers,
    ``scores`` for anomaly detectors (lower = more anomalous).
    """

    values: list[Any]
    probabilities: list[dict[str, float]] | None = None
    scores: list[float] | None = None


class ModelRunner(Protocol):
    def predict(self, features: np.ndarray) -> RunnerOutput:
        """Predict for a 2-D float32 array in the feature order of the schema."""
        ...


@dataclass
class LoadedModel:
    descriptor: dict[str, Any]
    feature_schema: dict[str, Any]
    runner: ModelRunner

    @property
    def version(self) -> str:
        return str(self.descriptor["model_version"])


class ModelRegistry(Protocol):
    def resolve(self, use_case_id: str, version: str = "latest") -> LoadedModel | None:
        """Return the requested model version, or ``None`` if it has not been trained."""
        ...

    def versions(self, use_case_id: str) -> list[str]:
        ...
