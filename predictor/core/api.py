"""HTTP API (FastAPI). Thin layer over :class:`PredictionService`."""

from __future__ import annotations

import json
import tempfile
from importlib.resources import files
from typing import Any, Iterator, Literal

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from predictor.core.service import (
    SCHEMA_VERSION,
    STREAM_CHUNK_SIZE,
    PredictionService,
    RequestRejected,
    UnparsableRecord,
    UseCaseNotFound,
    UseCaseNotReady,
)

MAX_RECORDS_PER_REQUEST = 10_000
SPOOL_BYTES = 8 * 1024 * 1024
DASHBOARD_HTML = files("predictor.core").joinpath("dashboard.html").read_text(encoding="utf-8")


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid")
    record_id: str | None = None
    features: dict[str, Any]


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["1.0"] = SCHEMA_VERSION
    records: list[Record] = Field(min_length=1, max_length=MAX_RECORDS_PER_REQUEST)


def create_app(service: PredictionService) -> FastAPI:
    app = FastAPI(
        title="Applied AI prediction backend",
        version="0.1.0",
        description=(
            "Model-agnostic prediction API. Every use case shares this API; "
            "the feature schema per use case is available at `GET /models/{use_case_id}`."
        ),
    )

    @app.exception_handler(UseCaseNotFound)
    async def _not_found(_: Request, exc: UseCaseNotFound) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": f"unknown use case '{exc}'"})

    @app.exception_handler(UseCaseNotReady)
    async def _not_ready(_: Request, exc: UseCaseNotReady) -> JSONResponse:
        return JSONResponse(
            status_code=409,
            content={"detail": f"use case '{exc}' has no trained model yet; run `predictor train`"},
        )

    @app.exception_handler(RequestRejected)
    async def _rejected(_: Request, exc: RequestRejected) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": "invalid records", "errors": exc.errors})

    @app.get("/health")
    def health() -> dict[str, Any]:
        use_cases = service.list_use_cases()
        return {"status": "ok", "use_cases_ready": sum(uc["status"] == "ready" for uc in use_cases)}

    @app.get("/use-cases")
    def list_use_cases() -> list[dict[str, Any]]:
        return service.list_use_cases()

    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse("/dashboard")

    @app.get("/dashboard", include_in_schema=False)
    def dashboard() -> HTMLResponse:
        return HTMLResponse(DASHBOARD_HTML)

    @app.get("/models/{use_case_id}")
    def describe(use_case_id: str) -> dict[str, Any]:
        return service.describe(use_case_id)

    @app.get("/models/{use_case_id}/evaluation")
    def evaluation(use_case_id: str) -> dict[str, Any]:
        return service.evaluation(use_case_id)

    @app.post("/models/{use_case_id}/predict")
    def predict(use_case_id: str, body: PredictRequest) -> dict[str, Any]:
        records = [r.model_dump() for r in body.records]
        return service.predict(use_case_id, records)

    @app.post(
        "/models/{use_case_id}/predict/batch",
        summary="Batch prediction (JSONL in, JSONL out)",
        description=(
            "Send one record per line (`{\"record_id\": ..., \"features\": {...}}`) with "
            "`Content-Type: application/x-ndjson`. Records are processed in bounded chunks; "
            "invalid lines produce an error line instead of failing the batch."
        ),
    )
    async def predict_batch(use_case_id: str, request: Request) -> StreamingResponse:
        service.describe(use_case_id)  # fail fast with 404/409 before reading the body

        # Spool the upload (memory up to SPOOL_BYTES, then disk), then predict in bounded
        # chunks while streaming the results back, so memory does not grow with the file.
        spool = tempfile.SpooledTemporaryFile(max_size=SPOOL_BYTES)
        async for chunk in request.stream():
            spool.write(chunk)
        spool.seek(0)

        def encode(chunk: list[Any], offset: int) -> str:
            # One string per chunk: Starlette hops to a thread for every yielded item.
            return "".join(json.dumps(line) + "\n" for line in service.predict_stream(use_case_id, chunk, offset))

        def results() -> Iterator[str]:
            with spool:
                chunk: list[Any] = []
                offset = 0
                for raw in spool:
                    if raw.strip():
                        chunk.append(_parse_line(raw))
                    if len(chunk) >= STREAM_CHUNK_SIZE:
                        yield encode(chunk, offset)
                        offset += len(chunk)
                        chunk = []
                if chunk:
                    yield encode(chunk, offset)

        return StreamingResponse(results(), media_type="application/x-ndjson")

    @app.get("/metrics")
    def metrics() -> dict[str, Any]:
        return service.monitor.snapshot()

    return app


def _parse_line(line: bytes) -> Any:
    try:
        return json.loads(line)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        return UnparsableRecord(f"invalid JSON: {exc}")
