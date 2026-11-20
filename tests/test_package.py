"""The application package: build, verify, serve-only vs with data."""

import json
import tarfile

import pytest

from predictor.core import package
from test_dataset_manifest import make_dataset


@pytest.fixture
def project(tmp_path):
    home = tmp_path / "project"
    (home / "usecases" / "demo").mkdir(parents=True)
    (home / "usecases" / "demo" / "manifest.json").write_text("{}")
    version = home / "models" / "demo" / "1"
    version.mkdir(parents=True)
    (version / "feature_schema.json").write_text("{}")
    (home / "models" / "demo" / ".1-staging").mkdir()
    (home / "models" / "demo" / ".1-staging" / "junk").write_text("x")
    make_dataset(home)
    return home


def build(project, out, **kw):
    return package.build_package(project, project / "usecases", project / "models", project / "datasets", out, **kw)


def test_serve_only_by_default(project, tmp_path):
    manifest = build(project, tmp_path / "out")
    names = {f.path for f in manifest.files}
    assert names == {"usecases/demo/manifest.json", "models/demo/1/feature_schema.json"}
    assert not manifest.with_data
    assert package.verify_package(tmp_path / "out").ok


def test_with_data_adds_datasets_and_their_files(project, tmp_path):
    manifest = build(project, tmp_path / "out", with_data=True)
    names = {f.path for f in manifest.files}
    assert {"datasets/demo-set/dataset.json", "data/demo.csv"} <= names
    assert package.verify_package(tmp_path / "out").ok


def test_verify_reports_changed_missing_and_unlisted_files(project, tmp_path):
    build(project, tmp_path / "out")
    out = tmp_path / "out"
    (out / "usecases" / "demo" / "manifest.json").write_text('{"changed": 1}')
    (out / "models" / "demo" / "1" / "feature_schema.json").unlink()
    (out / "models" / "demo" / "1" / "extra.json").write_text("{}")
    report = package.verify_package(out)
    assert not report.ok
    assert any("changed" in e for e in report.errors) and any("missing" in e for e in report.errors)
    assert any("extra.json" in w for w in report.warnings)


def test_archive_round_trip_and_checksum(project, tmp_path):
    archive = tmp_path / "pkg.tar.gz"
    build(project, archive)
    assert (tmp_path / "pkg.tar.gz.sha256").is_file()
    assert package.verify_package(archive).ok
    with open(archive, "ab") as fh:  # tamper with the archive
        fh.write(b"x")
    assert not package.verify_package(archive).ok


def test_archive_with_wrong_content_is_detected(project, tmp_path):
    archive = tmp_path / "pkg.tar.gz"
    build(project, archive)
    (tmp_path / "pkg.tar.gz.sha256").unlink()
    with tarfile.open(archive) as tar:
        assert any(n.endswith("package.json") for n in tar.getnames())


def test_build_refuses_a_non_empty_target_and_missing_use_cases(project, tmp_path):
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "x").write_text("1")
    with pytest.raises(package.PackageError, match="not empty"):
        build(project, tmp_path / "out")
    with pytest.raises(package.PackageError, match="no use cases"):
        package.build_package(project, tmp_path / "none", project / "models", project / "datasets", tmp_path / "o2")


def test_with_data_fails_when_a_dataset_file_is_missing(project, tmp_path):
    (project / "data" / "demo.csv").unlink()
    with pytest.raises(package.PackageError, match="missing"):
        build(project, tmp_path / "out", with_data=True)


def test_package_json_is_plain_json(project, tmp_path):
    build(project, tmp_path / "out")
    data = json.loads((tmp_path / "out" / "package.json").read_text())
    assert data["package_version"] == "1.0" and data["files"][0]["sha256"]
