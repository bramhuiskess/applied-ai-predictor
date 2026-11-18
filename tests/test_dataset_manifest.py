"""Dataset manifests (dataset.json): schema, verification by the doctor, and the demo-dataset table."""

import json

import pytest

from predictor import data_fetch
from predictor.core.dataset import (
    DatasetError, DatasetManifest, dataset_path, describe_file, load_dataset_manifest, resolve_file,
    sha256_file, write_dataset_manifest,
)
from predictor.core.doctor import ERROR, OK, WARN, run_doctor

CSV = "a,b,when\n1,x,2024-01-01\n2,y,2024-01-02\n3.5,z,2024-01-03\n"


def make_dataset(home, dataset_id="demo-set"):
    (home / "data").mkdir(exist_ok=True)
    (home / "data" / "demo.csv").write_text(CSV)
    file = describe_file(home, "data/demo.csv")
    manifest = DatasetManifest(dataset_version="1.0", id=dataset_id, files=[file])
    write_dataset_manifest(home / "datasets", manifest)
    return manifest


def test_describe_file_counts_rows_and_guesses_types(tmp_path):
    file = describe_file(tmp_path, make_dataset(tmp_path).files[0].path)
    assert file.rows == 3 and file.sha256 == sha256_file(tmp_path / "data" / "demo.csv")
    assert {c.name: c.dtype for c in file.columns} == {"a": "number", "b": "string", "when": "datetime"}


def test_round_trip_and_helpers(tmp_path):
    written = make_dataset(tmp_path)
    loaded = load_dataset_manifest(tmp_path / "datasets" / "demo-set")
    assert loaded == written
    assert dataset_path(loaded, tmp_path, "demo.csv") == (tmp_path / "data" / "demo.csv").resolve()
    with pytest.raises(DatasetError, match="no file"):
        dataset_path(loaded, tmp_path, "other.csv")


@pytest.mark.parametrize("change, message", [
    ({"dataset_version": "9"}, "unsupported dataset_version"),
    ({"id": "Other"}, "lowercase slug"),
    ({"id": "other-set"}, "must match its directory"),
    ({"files": []}, "at least one file"),
    ({"extra": 1}, "Extra inputs"),
])
def test_invalid_dataset_manifests_fail(tmp_path, change, message):
    make_dataset(tmp_path)
    path = tmp_path / "datasets" / "demo-set" / "dataset.json"
    path.write_text(json.dumps({**json.loads(path.read_text()), **change}))
    with pytest.raises(DatasetError, match=message):
        load_dataset_manifest(path)


def test_a_file_cannot_leave_the_project_folder(tmp_path):
    manifest = make_dataset(tmp_path)
    escaping = manifest.files[0].model_copy(update={"path": "../outside.csv"})
    with pytest.raises(DatasetError, match="inside the project"):
        resolve_file(tmp_path, escaping)


def doctor(home):
    return run_doctor(home, home / "usecases", home / "models", home / "datasets")


def test_doctor_verifies_a_pasted_dataset_folder(tmp_path):
    make_dataset(tmp_path)
    found = [f for f in doctor(tmp_path) if f.check == "dataset demo-set"]
    assert [f.level for f in found] == [OK]
    (tmp_path / "data" / "demo.csv").write_text(CSV + "4,w,2024-01-04\n")  # a changed byte
    assert [f.level for f in doctor(tmp_path) if f.check == "dataset demo-set"] == [ERROR]
    (tmp_path / "data" / "demo.csv").unlink()
    assert [f.level for f in doctor(tmp_path) if f.check == "dataset demo-set"] == [WARN]


def test_doctor_reports_manifest_columns_missing_from_the_data(tmp_path):
    use_case = tmp_path / "usecases" / "demo"
    use_case.mkdir(parents=True)
    (use_case / "manifest.json").write_text(json.dumps({
        "manifest_version": "1.0", "use_case_id": "demo", "data_source": {"adapter": "csv-file", "path": "data/demo.csv"},
        "training": {"target": "nope"}}))
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "demo.csv").write_text(CSV)
    errors = [f for f in doctor(tmp_path) if f.level == ERROR]
    assert errors and "nope" in errors[0].message


def test_doctor_does_not_require_data_to_serve(tmp_path):
    """Serve-only: a use case with a model but no data is fine."""
    use_case = tmp_path / "usecases" / "demo"
    use_case.mkdir(parents=True)
    (use_case / "manifest.json").write_text(json.dumps({
        "manifest_version": "1.0", "use_case_id": "demo", "data_source": {"adapter": "csv-file", "path": "data/demo.csv"},
        "training": {"target": "y"}}))
    version = tmp_path / "models" / "demo" / "1"
    (version / "y").mkdir(parents=True)
    (version / "feature_schema.json").write_text("{}")
    (version / "use_case.json").write_text('{"outputs": ["y"]}')
    (version / "y" / "descriptor.json").write_text("{}")
    (version / "y" / "model.onnx").write_bytes(b"x")
    assert not [f for f in doctor(tmp_path) if f.level == ERROR]


def test_fetch_describes_existing_files_without_running_a_generator(tmp_path):
    dataset = data_fetch.DEMO_DATASETS[3]  # tomato-season: one file
    (tmp_path / "data").mkdir()
    for f in dataset.files:
        (tmp_path / f).write_text(CSV)
    assert data_fetch.fetch(tmp_path, [dataset.id]) == [dataset.id]
    manifest = load_dataset_manifest(tmp_path / "datasets" / dataset.id)
    assert [f.path for f in manifest.files] == list(dataset.files)
    assert next(s for s in data_fetch.state(tmp_path) if s["id"] == dataset.id)["described"]


def test_fetch_rejects_unknown_ids(tmp_path):
    with pytest.raises(data_fetch.FetchError, match="unknown dataset"):
        data_fetch.fetch(tmp_path, ["nope"])


def test_every_demo_script_exists_and_ids_are_unique():
    assert all((data_fetch.scripts_dir() / d.script).is_file() for d in data_fetch.DEMO_DATASETS)
    assert len(set(data_fetch.known_ids())) == len(data_fetch.DEMO_DATASETS)
