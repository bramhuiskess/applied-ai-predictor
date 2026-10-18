"""Architecture tests (SQ1, section 1.7): keep the core reusable.

These fail when the core starts depending on a specific ML library, on an
adapter, or on a use case, and when a use case starts containing code.
"""

import ast
import json
from pathlib import Path

import pytest

from predictor import bootstrap
from predictor.core.manifest import ManifestError, load_manifest

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "predictor" / "core"

FORBIDDEN_IN_CORE = (
    "sklearn", "lightgbm", "xgboost", "onnx", "onnxruntime", "skl2onnx", "onnxmltools", "pandas", "torch",
    "predictor.adapters", "predictor.trainer", "predictor.bootstrap", "predictor.cli",
)


def imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


@pytest.mark.parametrize("path", sorted(CORE.glob("*.py")), ids=lambda p: p.name)
def test_core_does_not_depend_on_ml_libraries_adapters_or_use_cases(path):
    bad = {m for m in imported_modules(path) for f in FORBIDDEN_IN_CORE if m == f or m.startswith(f + ".")}
    assert not bad, f"{path.name} imports {sorted(bad)}"


def test_use_cases_are_configuration_only():
    files = [p for p in (ROOT / "usecases").rglob("*") if p.is_file()]
    assert files, "no use cases found"
    assert all(p.name == "manifest.json" for p in files), files


def test_all_repository_manifests_are_valid():
    manifests = bootstrap.load_manifests(ROOT / "usecases")
    assert {m.use_case_id for m in manifests} >= {"greenhouse-climate", "absence-forecast", "machine-failure"}


def _write(tmp_path, data, name=None):
    directory = tmp_path / (name or data.get("use_case_id", "x"))
    directory.mkdir()
    (directory / "manifest.json").write_text(json.dumps(data))
    return directory


VALID = {"manifest_version": "1.0", "use_case_id": "demo", "data_source": {"adapter": "csv-file"},
         "training": {"target": "y"}}


@pytest.mark.parametrize(
    "change, message",
    [
        ({"trainig": {}}, "Extra inputs"),  # typo in a key
        ({"manifest_version": "9.9"}, "unsupported manifest_version"),
        ({"policy": {"contains_personal_data": True, "log_feature_values": True}}, "may not be logged"),
        ({"training": {"task": "regression"}}, "needs a 'target'"),
        ({"training": {"target": "y", "horizon_steps": 3}}, "time_column"),
    ],
)
def test_invalid_manifests_fail_fast(tmp_path, change, message):
    with pytest.raises(ManifestError, match=message):
        load_manifest(_write(tmp_path, {**VALID, **change}))


def test_manifest_must_live_in_a_directory_with_its_id(tmp_path):
    with pytest.raises(ManifestError, match="must match its directory"):
        load_manifest(_write(tmp_path, VALID, name="other"))


def test_unknown_adapter_is_rejected_at_startup(tmp_path):
    _write(tmp_path, {**VALID, "data_source": {"adapter": "excel-file"}})
    with pytest.raises(ManifestError, match="unknown data_source adapter"):
        bootstrap.load_manifests(tmp_path)
