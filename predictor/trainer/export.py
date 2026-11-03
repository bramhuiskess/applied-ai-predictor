"""Export a fitted sklearn Pipeline to ONNX and verify it gives the same answers."""

from __future__ import annotations

import re
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import Booster, LGBMClassifier, LGBMRegressor
from onnxmltools.convert.lightgbm.operator_converters.LightGbm import convert_lightgbm
from skl2onnx import convert_sklearn, update_registered_converter
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx.common.shape_calculator import (
    calculate_linear_classifier_output_shapes,
    calculate_linear_regressor_output_shapes,
)
from sklearn.pipeline import Pipeline

from predictor.adapters.onnx_runner import OnnxRunner

TARGET_OPSET = {"": 17, "ai.onnx.ml": 3}
PARITY_RTOL = 1e-4
PARITY_ATOL = 1e-4

update_registered_converter(
    LGBMClassifier, "LightGbmLGBMClassifier", calculate_linear_classifier_output_shapes, convert_lightgbm,
    options={"nocl": [True, False], "zipmap": [True, False, "columns"]},
)
update_registered_converter(
    LGBMRegressor, "LightGbmLGBMRegressor", calculate_linear_regressor_output_shapes, convert_lightgbm,
    options={"split": None},
)


def make_thresholds_float32_safe(model: Pipeline) -> None:
    """Snap LightGBM split thresholds down to the nearest float32.

    LightGBM keeps thresholds as float64 while ONNX tree ensembles use float32.
    For a float32 input ``x``, ``x <= t`` equals ``x <= floor32(t)``, so after this
    every split decides identically in LightGBM and in ONNX Runtime. Serving
    always sends float32, so predictions are unchanged.
    """
    est = model.named_steps["model"]
    if not isinstance(est, (LGBMRegressor, LGBMClassifier)):
        return
    text = est.booster_.model_to_string()

    def snap(match: re.Match[str]) -> str:
        values = []
        for raw in match.group(1).split():
            t = float(raw)
            f = np.float32(t)
            if float(f) > t:
                f = np.nextafter(f, np.float32(-np.inf))
            values.append(repr(float(f)))
        return "threshold=" + " ".join(values)

    text = re.sub(r"^threshold=(.*)$", snap, text, flags=re.MULTILINE)
    # tree_sizes holds byte offsets that no longer match; LightGBM parses fine without it.
    text = re.sub(r"^tree_sizes=.*\n", "", text, flags=re.MULTILINE)
    est._Booster = Booster(model_str=text)


def to_onnx(model: Pipeline, n_features: int, task: str) -> bytes:
    options = None
    if task == "classification":
        options = {id(model.named_steps["model"]): {"zipmap": False}}
    onx = convert_sklearn(
        model,
        initial_types=[("features", FloatTensorType([None, n_features]))],
        target_opset=TARGET_OPSET,
        options=options,
    )
    return onx.SerializeToString()


def check_parity(model: Pipeline, artifact: bytes, X: pd.DataFrame, descriptor: dict[str, Any]) -> dict[str, Any]:
    """Compare the ONNX model, run through the real serving runner, with the original."""
    X32 = X.to_numpy(dtype=np.float32)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "model.onnx"
        path.write_bytes(artifact)
        out = OnnxRunner(path, descriptor).predict(X32)

    task = descriptor["task_type"]
    if task == "regression":
        expected, actual = model.predict(X32), np.array(out.values)
        agreement = None
    elif task == "classification":
        classes = descriptor["classes"]
        expected = model.predict_proba(X32)
        actual = np.array([[p[c] for c in classes] for p in out.probabilities])
        agreement = float(np.mean(np.array([classes.index(v) for v in out.values]) == expected.argmax(axis=1)))
    else:
        expected, actual = model.decision_function(X32), np.array(out.scores)
        labels = np.where(model.predict(X32) == -1, "anomaly", "normal")
        agreement = float(np.mean(labels == np.array(out.values)))

    diff = np.abs(np.asarray(expected, dtype=float) - np.asarray(actual, dtype=float))
    passed = bool(np.allclose(expected, actual, rtol=PARITY_RTOL, atol=PARITY_ATOL))
    if agreement is not None:
        passed = passed and agreement >= 0.999
    return {
        "rows_compared": int(len(X32)),
        "max_abs_diff": float(diff.max()) if diff.size else 0.0,
        "label_agreement": agreement,
        "passed": passed,
    }
