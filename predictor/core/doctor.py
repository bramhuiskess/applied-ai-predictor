"""``predictor doctor``: a read-only report on whether a project (or package) is ready to train and to serve.

Answers "what is missing and which command fixes it". Pure standard library (csv header only; the core
never imports pandas), so it also runs in the serve-only install.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from predictor.core.dataset import (
    DatasetError, csv_header, load_all_dataset_manifests, resolve_file, sha256_file,
)
from predictor.core.manifest import MANIFEST_FILENAME, Manifest, ManifestError, check_references, load_manifest

OK, WARN, ERROR = "ok", "warn", "error"
TRAIN_MODULES = ("pandas", "sklearn", "lightgbm", "skl2onnx")


@dataclass
class Finding:
    level: str
    check: str
    message: str
    hint: str = ""

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


def has_errors(findings: list[Finding]) -> bool:
    return any(f.level == ERROR for f in findings)


def installed_train_extra() -> bool:
    return all(importlib.util.find_spec(m) is not None for m in TRAIN_MODULES)


def model_state(models: Path, use_case_id: str) -> tuple[str | None, list[str]]:
    """(latest version, problems) of the trained model of a use case; version None when it is not trained."""
    base = Path(models) / use_case_id
    versions = sorted((p.name for p in base.iterdir() if p.name.isdigit()
                       and ((p / "use_case.json").is_file() or (p / "descriptor.json").is_file())),
                      key=int) if base.is_dir() else []
    if not versions:
        return None, []
    directory = base / versions[-1]
    problems = [] if (directory / "feature_schema.json").is_file() else ["feature_schema.json missing"]
    if (directory / "use_case.json").is_file():
        try:
            names = json.loads((directory / "use_case.json").read_text("utf-8"))["outputs"]
        except (OSError, ValueError, KeyError):
            return versions[-1], ["use_case.json unreadable"]
        folders = [(n, directory / n) for n in names]
    else:
        folders = [(".", directory)]
    for name, folder in folders:
        for file in ("descriptor.json", "model.onnx"):
            if not (folder / file).is_file():
                problems.append(f"{name}/{file} missing")
    return versions[-1], problems


def run_doctor(home: Path, usecases: Path, models: Path, datasets: Path, *,
               known_sources: set[str] | None = None, known_runners: set[str] | None = None) -> list[Finding]:
    home = Path(home)
    out: list[Finding] = []

    def add(level: str, check: str, message: str, hint: str = "") -> None:
        out.append(Finding(level, check, message, hint))

    if sys.version_info < (3, 11):
        add(ERROR, "python", f"Python {sys.version_info.major}.{sys.version_info.minor} is too old (needs 3.11+)",
            "install Python 3.12, e.g. `brew install python@3.12`")
    else:
        add(OK, "python", f"Python {sys.version_info.major}.{sys.version_info.minor}")
    if installed_train_extra():
        add(OK, "install", "training libraries installed")
    else:
        add(WARN, "install", "training libraries not installed: you can serve models, not train them",
            "pip install -e '.[train]'  (or: uv sync --extra train)")

    manifests: list[Manifest] = []
    for path in sorted(Path(usecases).glob(f"*/{MANIFEST_FILENAME}")):
        try:
            manifest = load_manifest(path)
            if known_sources is not None and known_runners is not None:
                check_references(manifest, known_sources=known_sources, known_runners=known_runners)
            manifests.append(manifest)
        except ManifestError as exc:
            add(ERROR, f"usecase {path.parent.name}", str(exc), "fix the manifest")
    if manifests:
        add(OK, "usecases", f"{len(manifests)} use case(s) valid")
    elif not has_errors(out):
        add(ERROR, "usecases", f"no use cases in {usecases}",
            "add usecases/<id>/manifest.json (`predictor init <id> --data file.csv --target col`) or use a package")

    declared: set[str] = set()
    try:
        dataset_manifests = load_all_dataset_manifests(datasets)
    except DatasetError as exc:
        dataset_manifests = []
        add(ERROR, "datasets", str(exc), "fix or remove that dataset.json")
    for dataset in dataset_manifests:
        for file in dataset.files:
            try:
                path = resolve_file(home, file)
            except DatasetError as exc:
                add(ERROR, f"dataset {dataset.id}", str(exc))
                continue
            declared.add(path.as_posix())
            if not path.is_file():
                add(WARN, f"dataset {dataset.id}", f"{file.path} is missing", "run `predictor data fetch`")
            elif file.sha256 and sha256_file(path) != file.sha256:
                add(ERROR, f"dataset {dataset.id}", f"{file.path} differs from its dataset.json (sha256)",
                    "restore the file, or update dataset.json if the change is intended")
            elif file.format == "csv" and (gone := [c.name for c in file.columns if c.name not in csv_header(path)]):
                add(ERROR, f"dataset {dataset.id}", f"{file.path}: columns missing: {', '.join(gone)}")
            else:
                add(OK, f"dataset {dataset.id}", f"{file.path} verified")

    for manifest in manifests:
        label = f"usecase {manifest.use_case_id}"
        version, problems = model_state(models, manifest.use_case_id)
        paths = [p for p in [manifest.data_source.path, *(j.source.path for j in manifest.joins)] if p]
        absent = [p for p in paths if not (home / p).is_file()]
        if version is None:
            if absent:
                add(WARN, label, f"not trained and data missing: {', '.join(absent)}",
                    "run `predictor data fetch` (demo data) or add the data file, then `predictor train`")
            else:
                add(WARN, label, "not trained", f"run `predictor train {manifest.use_case_id}`")
        elif problems:
            add(ERROR, label, f"model v{version} is incomplete: {'; '.join(problems)}",
                f"run `predictor train {manifest.use_case_id}`")
        else:
            add(OK, label, f"model v{version} present" + (" (serve-only: no data here)" if absent else ""))
        if not absent and manifest.data_source.adapter == "csv-file" and manifest.data_source.path:
            _check_columns(manifest, home / manifest.data_source.path, label, add)

    data_dir = home / "data"
    if data_dir.is_dir():
        loose = [p for p in data_dir.glob("*.csv") if p.resolve().as_posix() not in declared]
        if loose:
            add(WARN, "datasets", f"{len(loose)} file(s) in data/ have no dataset.json",
                "run `predictor data fetch` for the demo data; your own files keep working")
    return out


def _check_columns(manifest: Manifest, path: Path, label: str, add) -> None:
    try:
        header = set(csv_header(path))
    except (OSError, UnicodeDecodeError) as exc:
        add(ERROR, label, f"cannot read {path}: {exc}")
        return
    t = manifest.training
    renamed = set(t.rename_columns.values())
    wanted = [*t.output_names, *([t.time_column] if t.time_column else []), *t.rename_columns]
    missing = [c for c in wanted if c not in header and c not in renamed]
    if missing:
        add(ERROR, label, f"{path.name}: column(s) named in the manifest not found: {', '.join(missing)}",
            "check training.target / time_column / rename_columns")
