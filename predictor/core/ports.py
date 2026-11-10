"""Ports: the interfaces through which the core talks to the outside world.

The core only depends on these protocols. Concrete technology (ONNX Runtime,
the filesystem, CSV files) lives in ``predictor.adapters`` (Cockburn, 2005).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

if TYPE_CHECKING:
    from predictor.core.joins import JoinEngine


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
class OutputModel:
    """The trained model of one output of a use case."""

    name: str
    descriptor: dict[str, Any]
    runner: ModelRunner


@dataclass
class LoadedModel:
    """One published version: a model per output, sharing the inputs (schema), joins and support."""

    outputs: dict[str, OutputModel]
    feature_schema: dict[str, Any]
    support: dict[str, Any] | None = None  # map of the training data, for "comparable cases"
    joins: JoinEngine | None = None  # lookup tables the model was trained with, to fill joined inputs

    @property
    def primary(self) -> OutputModel:
        """The first output: what single-output code (and the flat API shape) means."""
        return next(iter(self.outputs.values()))

    @property
    def descriptor(self) -> dict[str, Any]:
        return self.primary.descriptor

    @property
    def runner(self) -> ModelRunner:
        return self.primary.runner

    @property
    def version(self) -> str:
        return str(self.descriptor["model_version"])

    def only(self, output: str) -> LoadedModel:
        """The same version seen as a single-output model (used for the aliases of an output)."""
        return LoadedModel({output: self.outputs[output]}, self.feature_schema, self.support, self.joins)


class ModelRegistry(Protocol):
    def resolve(self, use_case_id: str, version: str = "latest") -> LoadedModel | None:
        """Return the requested model version, or ``None`` if it has not been trained."""
        ...

    def versions(self, use_case_id: str) -> list[str]:
        ...
