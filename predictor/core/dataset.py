"""The dataset manifest (``dataset.json``): what a dataset consists of, so a pasted folder can be verified.

Schema and helpers only, no registry and no logic that resolves datasets for use cases (a separate backlog
item). Pure standard library: the core never imports pandas.

``files[].path`` is relative to the project home (``PREDICTOR_HOME``), e.g. ``data/greenhouse.csv`` or
``datasets/my-set/my-set.csv``. The manifest itself lives at ``datasets/<id>/dataset.json``.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError

DATASET_FILENAME = "dataset.json"
SUPPORTED_VERSION = "1.0"
_SLUG = re.compile(r"[a-z0-9][a-z0-9-]{1,62}")
_DATE = re.compile(r"\d{4}-\d{2}-\d{2}([T ]\d{2}:\d{2}(:\d{2})?)?")


class DatasetError(ValueError):
    """A dataset manifest is missing, unreadable or invalid."""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DatasetColumn(_Strict):
    name: str
    dtype: Literal["integer", "number", "boolean", "datetime", "string"]


class DatasetFile(_Strict):
    path: str
    format: Literal["csv", "jsonl", "parquet"] = "csv"
    sha256: str | None = None
    rows: int | None = None
    columns: list[DatasetColumn] = []


class DatasetManifest(_Strict):
    dataset_version: str
    id: str
    description: str = ""
    files: list[DatasetFile]
    source: str = ""  # free text: url, licence, how it was generated


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_dataset_manifest(path: Path) -> DatasetManifest:
    path = Path(path)
    if path.is_dir():
        path = path / DATASET_FILENAME
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise DatasetError(f"{path}: file not found") from exc
    except json.JSONDecodeError as exc:
        raise DatasetError(f"{path}: invalid JSON ({exc})") from exc
    try:
        manifest = DatasetManifest.model_validate(raw)
    except ValidationError as exc:
        problems = "; ".join(f"{'.'.join(str(p) for p in e['loc']) or '<root>'}: {e['msg']}" for e in exc.errors())
        raise DatasetError(f"{path}: {problems}") from exc
    if manifest.dataset_version != SUPPORTED_VERSION:
        raise DatasetError(f"{path}: unsupported dataset_version {manifest.dataset_version!r} "
                           f"(this version reads {SUPPORTED_VERSION!r})")
    if not _SLUG.fullmatch(manifest.id):
        raise DatasetError(f"{path}: id must be a lowercase slug, e.g. 'greenhouse-demo'")
    if path.parent.name != manifest.id:
        raise DatasetError(f"{path}: id {manifest.id!r} must match its directory name {path.parent.name!r}")
    if not manifest.files:
        raise DatasetError(f"{path}: a dataset needs at least one file")
    return manifest


def load_all_dataset_manifests(datasets_dir: Path) -> list[DatasetManifest]:
    return [load_dataset_manifest(p) for p in sorted(Path(datasets_dir).glob(f"*/{DATASET_FILENAME}"))]


def resolve_file(home: Path, file: DatasetFile) -> Path:
    """The file on disk. A path that leaves the project home is refused."""
    home = Path(home).resolve()
    target = (home / file.path).resolve()
    if home != target and home not in target.parents:
        raise DatasetError(f"{file.path}: a dataset file must stay inside the project folder")
    return target


def dataset_path(manifest: DatasetManifest, home: Path, name: str) -> Path:
    """The path of a file of this dataset, found by its file name (the one helper scripts and tests use)."""
    for file in manifest.files:
        if Path(file.path).name == name:
            return resolve_file(home, file)
    raise DatasetError(f"dataset {manifest.id!r} has no file {name!r}")


def csv_header(path: Path) -> list[str]:
    with open(path, newline="", encoding="utf-8-sig") as fh:
        return next(csv.reader(fh), [])


def _dtype(values: list[str]) -> str:
    values = [v for v in values if v != ""]
    if not values:
        return "string"
    for kind, test in (("integer", lambda v: re.fullmatch(r"[+-]?\d+", v)),
                       ("number", _is_float),
                       ("boolean", lambda v: v.lower() in {"true", "false"}),
                       ("datetime", lambda v: _DATE.fullmatch(v))):
        if all(test(v) for v in values):
            return kind
    return "string"


def _is_float(value: str) -> bool:
    try:
        float(value)
    except ValueError:
        return False
    return True


def describe_csv(path: Path, sample: int = 500) -> tuple[int, list[DatasetColumn]]:
    """Row count and column types (guessed from the first ``sample`` rows) of a csv file."""
    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.reader(fh)
        header = next(reader, [])
        samples: list[list[str]] = [[] for _ in header]
        rows = 0
        for row in reader:
            rows += 1
            if rows <= sample:
                for i, value in enumerate(row[:len(header)]):
                    samples[i].append(value.strip())
    return rows, [DatasetColumn(name=name, dtype=_dtype(values)) for name, values in zip(header, samples)]


def describe_file(home: Path, relative: str) -> DatasetFile:
    """A ``DatasetFile`` entry (hash, rows, columns) for a data file that exists; csv files only get rows/columns."""
    home = Path(home)
    path = home / relative
    suffix = path.suffix.lower().lstrip(".")
    fmt = {"csv": "csv", "jsonl": "jsonl", "parquet": "parquet"}.get(suffix)
    if fmt is None:
        raise DatasetError(f"{relative}: unsupported file type {path.suffix!r}")
    rows, columns = describe_csv(path) if fmt == "csv" else (None, [])
    return DatasetFile(path=relative, format=fmt, sha256=sha256_file(path), rows=rows, columns=columns)


def write_dataset_manifest(datasets_dir: Path, manifest: DatasetManifest) -> Path:
    directory = Path(datasets_dir) / manifest.id
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / DATASET_FILENAME
    path.write_text(json.dumps(manifest.model_dump(exclude_none=True), indent=2) + "\n", encoding="utf-8")
    return path
