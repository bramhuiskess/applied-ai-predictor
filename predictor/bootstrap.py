"""Composition root: the only place where the core is wired to concrete adapters."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from fastapi import FastAPI

from predictor.adapters.fs_registry import FilesystemRegistry
from predictor.adapters.onnx_runner import OnnxRunner
from predictor.adapters.sources import SOURCES
from predictor.core.api import create_app
from predictor.core.manifest import Manifest, check_references, load_all_manifests
from predictor.core.service import PredictionService

RUNNERS = {"onnx": OnnxRunner}


def usecases_dir() -> Path:
    return Path(os.environ.get("PREDICTOR_USECASES_DIR", "usecases"))


def models_dir() -> Path:
    return Path(os.environ.get("PREDICTOR_MODELS_DIR", "models"))


def build_registry(root: Path | None = None) -> FilesystemRegistry:
    return FilesystemRegistry(root or models_dir(), RUNNERS)


def load_manifests(directory: Path | None = None) -> list[Manifest]:
    manifests = load_all_manifests(directory or usecases_dir())
    for manifest in manifests:
        check_references(manifest, known_sources=set(SOURCES), known_runners=set(RUNNERS))
    return manifests


def build_service(usecases: Path | None = None, models: Path | None = None) -> PredictionService:
    return PredictionService.from_manifests(load_manifests(usecases), build_registry(models))


def build_app(usecases: Path | None = None, models: Path | None = None) -> FastAPI:
    logging.basicConfig(level=os.environ.get("PREDICTOR_LOG_LEVEL", "INFO"),
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    return create_app(build_service(usecases, models))
