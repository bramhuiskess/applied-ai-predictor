"""The application package: use cases, models and (optionally) datasets in one verifiable bundle.

    <package>/
      package.json              what is inside, with a sha256 per file
      usecases/<id>/manifest.json
      models/<id>/<n>/...
      datasets/<id>/dataset.json     only with data
      data/...                       only with data (the files the dataset manifests point at)

Serve-only (default) is ``usecases`` + ``models``. A package is used like a project: point
``PREDICTOR_HOME`` at it, or run ``predictor run`` inside it. Pure standard library.
"""

from __future__ import annotations

import json
import platform
import shutil
import tarfile
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version as pkg_version
from pathlib import Path

from pydantic import BaseModel, ConfigDict, ValidationError

from predictor.core.dataset import DatasetError, load_all_dataset_manifests, resolve_file, sha256_file

PACKAGE_FILENAME = "package.json"
PACKAGE_VERSION = "1.0"
ARCHIVE_SUFFIXES = (".tar.gz", ".tgz")


class PackageError(ValueError):
    """A package cannot be built or read."""


class PackageFile(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str
    sha256: str
    size: int


class PackageManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    package_version: str
    name: str
    created_at: str
    with_data: bool
    environment: dict[str, str] = {}
    files: list[PackageFile]


def environment() -> dict[str, str]:
    env = {"python": platform.python_version()}
    for pkg in ("onnxruntime", "predictor", "applied-ai-predictor"):
        try:
            env[pkg] = pkg_version(pkg)
        except PackageNotFoundError:
            pass
    return env


def _skip(path: Path) -> bool:
    return any(part.startswith(".") or part == "__pycache__" for part in path.parts)


def _tree(root: Path, prefix: str) -> list[tuple[Path, str]]:
    root = Path(root)
    if not root.is_dir():
        return []
    return [(p, f"{prefix}/{p.relative_to(root).as_posix()}") for p in sorted(root.rglob("*"))
            if p.is_file() and not _skip(p.relative_to(root))]


def collect(home: Path, usecases: Path, models: Path, datasets: Path, with_data: bool) -> list[tuple[Path, str]]:
    """(source file, path inside the package) for everything that goes into the package."""
    items = _tree(usecases, "usecases") + _tree(models, "models")
    if with_data:
        try:
            for manifest in load_all_dataset_manifests(datasets):
                items.append((Path(datasets) / manifest.id / "dataset.json", f"datasets/{manifest.id}/dataset.json"))
                for file in manifest.files:
                    source = resolve_file(home, file)
                    if not source.is_file():
                        raise PackageError(f"dataset {manifest.id!r}: file {file.path} is missing "
                                           f"(run `predictor data fetch`)")
                    items.append((source, Path(file.path).as_posix()))
        except DatasetError as exc:
            raise PackageError(str(exc)) from exc
    seen: dict[str, Path] = {}
    for source, target in items:
        seen.setdefault(target, source)  # a data file shared by two datasets is stored once
    return [(source, target) for target, source in seen.items()]


def build_package(home: Path, usecases: Path, models: Path, datasets: Path, out: Path, *,
                  with_data: bool = False, name: str | None = None) -> PackageManifest:
    """Write the package to a folder, or to a .tar.gz (plus a .sha256 next to it)."""
    out = Path(out)
    archive = out.name.endswith(ARCHIVE_SUFFIXES)
    items = collect(home, usecases, models, datasets, with_data)
    if not any(t.startswith("usecases/") for _, t in items):
        raise PackageError(f"no use cases found in {usecases}")
    if archive:
        staging_cm = tempfile.TemporaryDirectory()
        staging = Path(staging_cm.name) / "package"
    else:
        if out.exists() and any(out.iterdir()):
            raise PackageError(f"{out} already exists and is not empty")
        staging_cm, staging = None, out
    try:
        files = []
        for source, target in items:
            dest = staging / target
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dest)
            files.append(PackageFile(path=target, sha256=sha256_file(dest), size=dest.stat().st_size))
        manifest = PackageManifest(
            package_version=PACKAGE_VERSION, name=name or out.name.split(".")[0] or "package",
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"), with_data=with_data,
            environment=environment(), files=files)
        (staging / PACKAGE_FILENAME).write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
        if archive:
            out.parent.mkdir(parents=True, exist_ok=True)
            with tarfile.open(out, "w:gz") as tar:
                tar.add(staging, arcname=manifest.name)
            Path(str(out) + ".sha256").write_text(f"{sha256_file(out)}  {out.name}\n", encoding="utf-8")
    finally:
        if staging_cm:
            staging_cm.cleanup()
    return manifest


@dataclass
class VerifyReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    manifest: PackageManifest | None = None

    @property
    def ok(self) -> bool:
        return not self.errors


def verify_package(path: Path) -> VerifyReport:
    """Offline check of a package folder or archive: every listed file present, unchanged, nothing unlisted."""
    path = Path(path)
    if path.is_file() and path.name.endswith(ARCHIVE_SUFFIXES):
        with tempfile.TemporaryDirectory() as tmp:
            report = VerifyReport()
            checksum = Path(str(path) + ".sha256")
            if checksum.is_file() and checksum.read_text(encoding="utf-8").split()[:1] != [sha256_file(path)]:
                report.errors.append(f"{path.name}: does not match {checksum.name}")
                return report
            try:
                with tarfile.open(path) as tar:
                    tar.extractall(tmp, filter="data")
            except (tarfile.TarError, OSError) as exc:
                report.errors.append(f"{path.name}: cannot unpack ({exc})")
                return report
            roots = [p for p in Path(tmp).iterdir() if p.is_dir()]
            return verify_package(roots[0]) if len(roots) == 1 else _error(f"{path.name}: unexpected layout")
    report = VerifyReport()
    try:
        manifest = PackageManifest.model_validate(json.loads((path / PACKAGE_FILENAME).read_text(encoding="utf-8")))
    except FileNotFoundError:
        report.errors.append(f"{path}: no {PACKAGE_FILENAME}")
        return report
    except (json.JSONDecodeError, ValidationError) as exc:
        report.errors.append(f"{path / PACKAGE_FILENAME}: {exc}")
        return report
    report.manifest = manifest
    if manifest.package_version != PACKAGE_VERSION:
        report.errors.append(f"unsupported package_version {manifest.package_version!r}")
    listed = set()
    root = path.resolve()
    for entry in manifest.files:
        listed.add(entry.path)
        target = (root / entry.path).resolve()
        if root not in target.parents:
            report.errors.append(f"{entry.path}: points outside the package")
        elif not target.is_file():
            report.errors.append(f"{entry.path}: missing")
        elif sha256_file(target) != entry.sha256:
            report.errors.append(f"{entry.path}: changed (sha256 differs)")
    for source, target in _tree(path / "usecases", "usecases") + _tree(path / "models", "models"):
        if target not in listed:
            report.warnings.append(f"{target}: not in {PACKAGE_FILENAME}")
    now, then = environment(), manifest.environment
    if then.get("python", "")[:4] != now["python"][:4]:
        report.warnings.append(f"built with Python {then.get('python')}, running {now['python']}")
    if then.get("onnxruntime") and then.get("onnxruntime") != now.get("onnxruntime"):
        report.warnings.append(f"models were exported with onnxruntime {then['onnxruntime']}, "
                               f"installed is {now.get('onnxruntime', 'none')}")
    return report


def _error(message: str) -> VerifyReport:
    return VerifyReport(errors=[message])
