"""Model-runner adapter for ONNX models (the default runner)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

from predictor.core.ports import RunnerOutput


class OnnxRunner:
    def __init__(self, artifact: Path, descriptor: dict[str, Any]) -> None:
        options = ort.SessionOptions()
        options.log_severity_level = 3
        self._session = ort.InferenceSession(
            str(artifact), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self._input = self._session.get_inputs()[0].name
        self._outputs = [o.name for o in self._session.get_outputs()]
        self._task = descriptor["task_type"]
        self._classes = descriptor.get("classes")

    def predict(self, features: np.ndarray) -> RunnerOutput:
        raw = dict(zip(self._outputs, self._session.run(None, {self._input: features.astype(np.float32)})))
        if self._task == "regression":
            return RunnerOutput(values=[float(v) for v in raw[self._outputs[0]].ravel()])
        if self._task == "classification":
            probabilities = raw["probabilities"]
            return RunnerOutput(
                values=[self._classes[i] for i in probabilities.argmax(axis=1)],
                probabilities=[
                    {label: float(p) for label, p in zip(self._classes, row)} for row in probabilities
                ],
            )
        if self._task == "anomaly":
            # IsolationForest convention: label -1 = anomaly, score < 0 = anomalous.
            return RunnerOutput(
                values=["anomaly" if v == -1 else "normal" for v in raw["label"].ravel()],
                scores=[float(s) for s in raw["scores"].ravel()],
            )
        raise ValueError(f"unsupported task type {self._task!r}")
