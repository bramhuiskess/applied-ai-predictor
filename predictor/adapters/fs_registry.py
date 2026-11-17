"""Model-registry adapter that stores versioned models on the filesystem.

Layout (one version holds one model per output of the use case)::

    models/<use_case_id>/<version>/
        use_case.json        which outputs this version has (written last: a version only counts once it exists)
        feature_schema.json  JSON Schema of the inputs, shared by all outputs
        <output>/model.onnx  the artefact of one output (preprocessing + model in one graph)
        <output>/descriptor.json  that model's metadata (task, algorithm, metrics, ...)
        report.md            human-readable training report
        <output>/support.json  (optional) map of that output's training data, for "comparable cases"
        joins.json           (optional) the joins the model was trained with
        lookups/<name>.npz   (optional) snapshot of each lookup table, so the service needs no data files

Versions written before outputs existed keep ``descriptor.json`` and ``model.onnx`` in the version folder itself;
they are read as a use case with one output.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from predictor.core.joins import JoinEngine
from predictor.core.ports import LoadedModel, ModelRunner, OutputModel

DESCRIPTOR = "descriptor.json"
SCHEMA = "feature_schema.json"
SUPPORT = "support.json"
JOINS = "joins.json"
USE_CASE = "use_case.json"
ARTIFACT_NAME = "model.onnx"

RunnerFactory = Callable[[Path, dict[str, Any]], ModelRunner]


@dataclass
class PublishedOutput:
    descriptor: dict[str, Any]
    artifact: bytes
    files: dict[str, str] | None = None  # more files for this output's folder, e.g. support.json


class FilesystemRegistry:
    def __init__(self, root: Path, runners: dict[str, RunnerFactory]) -> None:
        self.root = Path(root)
        self._runners = runners

    def versions(self, use_case_id: str) -> list[str]:
        base = self.root / use_case_id
        if not base.is_dir():
            return []
        found = [p.name for p in base.iterdir()
                 if p.name.isdigit() and ((p / USE_CASE).is_file() or (p / DESCRIPTOR).is_file())]
        return sorted(found, key=int)

    def resolve(self, use_case_id: str, version: str = "latest") -> LoadedModel | None:
        versions = self.versions(use_case_id)
        if not versions:
            return None
        if version == "latest":
            version = versions[-1]
        if version not in versions:
            raise LookupError(f"{use_case_id}: model version {version!r} not found (have {versions})")
        directory = self.root / use_case_id / version
        schema = json.loads((directory / SCHEMA).read_text(encoding="utf-8"))
        if (directory / USE_CASE).is_file():
            names = json.loads((directory / USE_CASE).read_text(encoding="utf-8"))["outputs"]
            folders = {name: directory / name for name in names}
        else:  # a version from before outputs existed: one output, files in the version folder
            legacy = json.loads((directory / DESCRIPTOR).read_text(encoding="utf-8"))
            folders = {legacy.get("output") or legacy.get("target_name") or "anomaly": directory}
        outputs = {}
        for name, folder in folders.items():
            descriptor = json.loads((folder / DESCRIPTOR).read_text(encoding="utf-8"))
            factory = self._runners.get(descriptor["runner"])
            if factory is None:
                raise LookupError(f"{use_case_id} v{version}: no runner installed for {descriptor['runner']!r}")
            support_path = folder / SUPPORT  # in the version folder itself for a version from before outputs existed
            support = json.loads(support_path.read_text(encoding="utf-8")) if support_path.is_file() else None
            outputs[name] = OutputModel(name, descriptor, factory(folder / descriptor["artifact"], descriptor), support)
        joins = None
        if (directory / JOINS).is_file():
            # The model is self-contained: joins come from this version's snapshot, never from the current manifest.
            try:
                joins = JoinEngine.from_snapshot(lambda name: (directory / name).read_bytes())
            except (OSError, ValueError, KeyError) as exc:
                raise LookupError(f"{use_case_id} v{version}: cannot load the join snapshot ({exc})") from exc
        return LoadedModel(outputs=outputs, feature_schema=schema, joins=joins)

    def publish(
        self,
        use_case_id: str,
        *,
        outputs: Mapping[str, PublishedOutput],
        feature_schema: dict[str, Any],
        extra_files: dict[str, str | bytes] | None = None,
    ) -> str:
        """Store a new version (a model per output) atomically and return its version number."""
        if not outputs:
            raise ValueError("a version needs at least one output")
        versions = self.versions(use_case_id)
        version = str(int(versions[-1]) + 1 if versions else 1)
        base = self.root / use_case_id
        base.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{version}-", dir=base))
        try:
            staging.chmod(0o755)  # mkdtemp uses 0700; the serving user must be able to read it
            (staging / SCHEMA).write_text(json.dumps(feature_schema, indent=2), encoding="utf-8")
            for name, content in (extra_files or {}).items():
                target = (staging / name).resolve()
                if staging.resolve() not in target.parents:
                    raise ValueError(f"extra file {name!r} must stay inside the model directory")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.parent.chmod(0o755)
                if isinstance(content, bytes):
                    target.write_bytes(content)
                else:
                    target.write_text(content, encoding="utf-8")
            taken = {SCHEMA, USE_CASE, "report.md", *(Path(n).parts[0] for n in (extra_files or {}))}
            for name, published in outputs.items():
                if not name or "/" in name or name.startswith(".") or name in taken:
                    raise ValueError(f"output name {name!r} cannot be used as a folder name in the model directory")
                folder = staging / name
                folder.mkdir()
                folder.chmod(0o755)
                descriptor = {**published.descriptor, "model_version": version, "artifact": ARTIFACT_NAME}
                (folder / ARTIFACT_NAME).write_bytes(published.artifact)
                (folder / DESCRIPTOR).write_text(json.dumps(descriptor, indent=2), encoding="utf-8")
                for file_name, content in (published.files or {}).items():
                    if "/" in file_name or file_name.startswith("."):
                        raise ValueError(f"file {file_name!r} of output {name!r} must be a plain file name")
                    (folder / file_name).write_text(content, encoding="utf-8")
            # Written last: a version only counts once it exists.
            (staging / USE_CASE).write_text(json.dumps({"model_version": version, "outputs": list(outputs)}, indent=2),
                                            encoding="utf-8")
            os.replace(staging, base / version)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        return version
