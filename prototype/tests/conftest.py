"""Shared fixtures: a throw-away project with generated data and trained models.

Everything is generated locally, so the tests run offline.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import make_datasets  # noqa: E402

from predictor import bootstrap  # noqa: E402
from predictor.trainer.run import train_use_case  # noqa: E402


def pump_data(n: int = 1500, seed: int = 7) -> pd.DataFrame:
    """Small synthetic classification dataset with a categorical and a binary column."""
    rng = np.random.default_rng(seed)
    pump_type = rng.choice(["A", "B", "C"], n)
    vibration = rng.normal(3, 1, n)
    pressure = rng.normal(6, 0.8, n)
    filter_ok = rng.integers(0, 2, n)
    logit = 1.8 * (vibration - 4) + 1.2 * (pressure - 6.5) - 1.5 * filter_ok + (pump_type == "C") * 0.8
    failure = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return pd.DataFrame({
        "pump_id": [f"p{i}" for i in range(n)],
        "pump_type": pump_type,
        "vibration_mm_s": vibration.round(3),
        "pressure_bar": pressure.round(3),
        "filter_ok": filter_ok,
        "failure": failure,
    })


def write_manifest(usecases: Path, manifest: dict) -> Path:
    directory = usecases / manifest["use_case_id"]
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


@pytest.fixture(scope="session")
def project(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("project")
    data, usecases, models = root / "data", root / "usecases", root / "models"
    data.mkdir()
    make_datasets.greenhouse(days=20).to_csv(data / "greenhouse.csv", index=False)
    absence = make_datasets.absence(departments=24, months=24)
    absence.to_csv(data / "absence.csv", index=False)
    bad = absence.head(100).copy()
    bad["diagnosis"] = "flu"
    bad.to_csv(data / "absence_with_diagnosis.csv", index=False)
    pump_data().to_csv(data / "pumps.csv", index=False)

    # The real manifests from the repository, pointed at the generated data.
    for name, file in [("greenhouse-climate", "greenhouse.csv"), ("absence-forecast", "absence.csv")]:
        manifest = json.loads((ROOT / "usecases" / name / "manifest.json").read_text(encoding="utf-8"))
        manifest["data_source"]["path"] = str(data / file)
        write_manifest(usecases, manifest)

    base = {"manifest_version": "1.0", "data_source": {"adapter": "csv-file", "path": str(data / "pumps.csv")},
            "policy": {"log_feature_values": True}}
    write_manifest(usecases, {**base, "use_case_id": "pump-failure",
                              "training": {"target": "failure", "drop_columns": ["pump_id"]}})
    write_manifest(usecases, {**base, "use_case_id": "pump-anomaly",
                              "training": {"task": "anomaly", "evaluation_label": "failure",
                                           "drop_columns": ["pump_id"]}})
    write_manifest(usecases, {**base, "use_case_id": "untrained", "training": {"target": "failure"}})

    registry = bootstrap.build_registry(models)
    for name in ["greenhouse-climate", "absence-forecast", "pump-failure", "pump-anomaly"]:
        train_use_case(usecases / name, registry)
    return root


@pytest.fixture(scope="session")
def client(project: Path) -> TestClient:
    app = bootstrap.build_app(project / "usecases", project / "models")
    return TestClient(app)


@pytest.fixture
def scratch(tmp_path: Path, project: Path) -> Path:
    """A writable copy of the data directory for tests that create use cases."""
    shutil.copytree(project / "data", tmp_path / "data")
    (tmp_path / "usecases").mkdir()
    return tmp_path


GREENHOUSE = {
    "hour_of_day": 13, "outside_temp_c": 9.5, "solar_radiation_wm2": 300.0, "inside_humidity_pct": 80.0,
    "co2_ppm": 520.0, "ventilation_on": 0, "heating_on": 0, "inside_temp_c": 21.4,
}
ABSENCE = {
    "department_type": "care", "headcount": 30, "avg_age": 44.0, "overtime_hours_per_fte": 6.0,
    "workload_index": 1.1, "flu_season": 1, "previous_month_absence_pct": 7.2,
}
PUMP = {"pump_type": "C", "vibration_mm_s": 5.2, "pressure_bar": 7.1, "filter_ok": 0}
