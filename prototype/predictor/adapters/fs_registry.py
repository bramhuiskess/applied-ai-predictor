"""Model-registry adapter that stores versioned models on the filesystem.

Layout::

    models/<use_case_id>/<version>/
        descriptor.json      model metadata (task, algorithm, metrics, ...)
        feature_schema.json  JSON Schema of the inputs
        model.onnx           the artefact (preprocessing + model in one graph)
        report.md            human-readable training report
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable

from predictor.core.ports import LoadedModel, ModelRunner

DESCRIPTOR = "descriptor.json"
SCHEMA = "feature_schema.json"

RunnerFactory = Callable[[Path, dict[str, Any]], ModelRunner]


class FilesystemRegistry:
    def __init__(self, root: Path, runners: dict[str, RunnerFactory]) -> None:
        self.root = Path(root)
        self._runners = runners

    def versions(self, use_case_id: str) -> list[str]:
        base = self.root / use_case_id
        if not base.is_dir():
            return []
        found = [p.name for p in base.iterdir() if p.name.isdigit() and (p / DESCRIPTOR).is_file()]
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
        descriptor = json.loads((directory / DESCRIPTOR).read_text(encoding="utf-8"))
        schema = json.loads((directory / SCHEMA).read_text(encoding="utf-8"))
        factory = self._runners.get(descriptor["runner"])
        if factory is None:
            raise LookupError(f"{use_case_id} v{version}: no runner installed for {descriptor['runner']!r}")
        runner = factory(directory / descriptor["artifact"], descriptor)
        return LoadedModel(descriptor=descriptor, feature_schema=schema, runner=runner)

    def publish(
        self,
        use_case_id: str,
        *,
        descriptor: dict[str, Any],
        feature_schema: dict[str, Any],
        artifact_name: str,
        artifact: bytes,
        extra_files: dict[str, str] | None = None,
    ) -> str:
        """Store a new version atomically and return its version number."""
        versions = self.versions(use_case_id)
        version = str(int(versions[-1]) + 1 if versions else 1)
        base = self.root / use_case_id
        base.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{version}-", dir=base))
        try:
            staging.chmod(0o755)  # mkdtemp uses 0700; the serving user must be able to read it
            descriptor = {**descriptor, "model_version": version, "artifact": artifact_name}
            (staging / artifact_name).write_bytes(artifact)
            (staging / SCHEMA).write_text(json.dumps(feature_schema, indent=2), encoding="utf-8")
            for name, content in (extra_files or {}).items():
                (staging / name).write_text(content, encoding="utf-8")
            # The descriptor is written last: a version only counts once it exists.
            (staging / DESCRIPTOR).write_text(json.dumps(descriptor, indent=2), encoding="utf-8")
            os.replace(staging, base / version)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        return version
