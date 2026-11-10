"""Training side of the joins: read the lookup tables, check them, apply the shared engine to the base table.

The join itself is done by ``predictor.core.joins`` (numpy only), the same code the service runs for a
request, so a feature has the same value when the model is trained and when it is used. This module only
does what needs pandas: reading files, adding the new columns, dropping rows that found no match.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from predictor.adapters.sources import read_table
from predictor.core.join_checks import JoinReport, check_join
from predictor.core.joins import OBSERVED, JoinEngine
from predictor.core.manifest import JoinConfig, Manifest
from predictor.trainer.errors import TrainingError


@dataclass
class AppliedJoins:
    df: pd.DataFrame  # base table plus the new columns, rows without a match removed
    engine: JoinEngine | None  # all joins that worked, with their lookup tables (stored with the model)
    reports: list[JoinReport]
    summary: list[dict[str, Any]]  # per join, for the descriptor and the report
    notes: list[str] = field(default_factory=list)
    consumed: list[str] = field(default_factory=list)  # base columns only needed to place the windows (not features)

    @property
    def failed(self) -> bool:
        return any(not r.ok for r in self.reports)


def arrays(df: pd.DataFrame, columns: list[str] | None = None) -> dict[str, np.ndarray]:
    return {c: df[c].to_numpy() for c in (columns if columns is not None else df.columns)}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_lookup(manifest: Manifest, join: JoinConfig) -> tuple[Path, pd.DataFrame]:
    if not join.source.path:
        raise TrainingError(f"{manifest.use_case_id}: join {join.name!r} has no source.path")
    path = Path(join.source.path)
    if not path.is_file():
        raise TrainingError(f"{manifest.use_case_id}: lookup table {path} of join {join.name!r} not found")
    return path, read_table(join.source.adapter, path)


def apply_joins(manifest: Manifest, df: pd.DataFrame, *, raise_on_error: bool = True) -> AppliedJoins:
    """Apply every join of the manifest in order. With ``raise_on_error=False`` problems are only reported."""
    reports: list[JoinReport] = []
    summary: list[dict[str, Any]] = []
    notes: list[str] = []
    lookups = {}
    working = df
    base_rows = len(df)
    for join in manifest.joins:
        path, table = read_lookup(manifest, join)
        # the clash check needs to see which new columns the base table already has
        present = [c for c in [*join.base_inputs, *join.new_columns] if c in working.columns]
        report = check_join(join, arrays(working, present), arrays(table), rows=len(working))
        reports.append(report)
        entry: dict[str, Any] = {
            "name": join.name, "source": str(path), "sha256": file_sha256(path), "lookup_rows": int(len(table)),
            "keys": dict(join.keys), "time": join.time.model_dump() if join.time else None,
            "window": join.window.model_dump() if join.window else None,
            "columns": join.new_columns, "fallback": join.fallback, "min_coverage": join.min_coverage,
            "min_match": join.min_match, "errors": report.errors, "warnings": report.warnings, "info": report.info,
        }
        summary.append(entry)
        if not report.ok:
            if raise_on_error:
                raise TrainingError(f"{manifest.use_case_id}: join {join.name!r} failed its checks:\n  - "
                                    + "\n  - ".join(report.errors))
            continue
        engine = JoinEngine([join], {join.name: report.lookup})
        result = engine.apply(arrays(working, join.base_inputs), fallback=False)
        keep = result.status[join.name] == OBSERVED
        working = working.copy()
        for name in join.new_columns:
            working[name] = result.columns[name]
        dropped = int((~keep).sum())
        entry.update(matched=float(keep.mean()), dropped_rows=dropped)
        if dropped:
            working = working[keep].reset_index(drop=True)
            notes.append(f"join {join.name!r}: dropped {dropped} of {len(keep)} rows without a match in {path.name}")
        lookups[join.name] = report.lookup

    applied = [j for j in manifest.joins if j.name in lookups]
    engine = JoinEngine(applied, lookups) if applied else None
    cfg = manifest.training
    keys = {c for j in manifest.joins for c in j.keys}
    consumed = [c for c in dict.fromkeys(c for j in manifest.joins for c in j.base_inputs)
                if c not in keys and c != cfg.time_column and c not in cfg.drop_columns]
    if applied:
        notes.insert(0, f"joined {len(applied)} lookup table(s) onto {base_rows} rows: "
                        + ", ".join(f"{j.name} -> {', '.join(j.new_columns)}" for j in applied))
    return AppliedJoins(working, engine, reports, summary, notes, consumed)


def format_reports(reports: list[JoinReport]) -> list[str]:
    """Readable lines for the CLI."""
    lines: list[str] = []
    for report in reports:
        lines.append(f"join {report.join}: {'OK' if report.ok else 'FAILED'}")
        stats = report.stats
        if "matched" in stats:
            lines.append(f"  - {stats['matched']:.1%} of {stats['rows']} rows match a lookup row "
                         f"({stats['lookup_rows']} lookup rows)")
        lines += [f"  - error: {m}" for m in report.errors]
        lines += [f"  - warning: {m}" for m in report.warnings]
        lines += [f"  - note: {m}" for m in report.info]
    return lines
