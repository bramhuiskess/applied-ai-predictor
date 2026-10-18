"""Training orchestration: manifest + dataset -> versioned model in the registry."""

from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime, timezone
from importlib.metadata import version as pkg_version
from pathlib import Path
from typing import Any

from predictor.adapters.fs_registry import FilesystemRegistry
from predictor.adapters.sources import adapter_for, read_table
from predictor.core.manifest import MANIFEST_FILENAME, Manifest, load_manifest
from predictor.trainer import models
from predictor.trainer.export import check_parity, make_thresholds_float32_safe, to_onnx
from predictor.trainer.errors import TrainingError
from predictor.trainer.profile import build_feature_schema, detect_task, prepare

ARTIFACT_NAME = "model.onnx"


def train_use_case(manifest_path: Path, registry: FilesystemRegistry, data_path: Path | None = None) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    cfg = manifest.training
    source = data_path or manifest.data_source.path
    if not source:
        raise TrainingError(f"{manifest.use_case_id}: no dataset given (use --data or data_source.path)")
    source = Path(source)
    if not source.is_file():
        raise TrainingError(f"{manifest.use_case_id}: dataset {source} not found")
    adapter = manifest.data_source.adapter if data_path is None else adapter_for(source)
    df = read_table(adapter, source)

    data = prepare(df, cfg, manifest.policy)
    task = data.task
    split = models.make_split(len(data.X), cfg.test_fraction, data.chronological,
                              data.y if task == "classification" else None)
    X_train, X_val, X_test = (data.X.iloc[idx] for idx in (split.train, split.val, split.test))
    y_train = y_val = y_test = None
    if data.y is not None:
        y_train, y_val, y_test = (data.y.iloc[idx] for idx in (split.train, split.val, split.test))
    label_val = label_test = None
    if data.evaluation_label is not None:
        label_val, label_test = data.evaluation_label.iloc[split.val], data.evaluation_label.iloc[split.test]

    results: dict[str, dict[str, Any]] = {}
    best_name, best_val = None, None
    for name, pipeline in models.candidates(task, cfg.anomaly_share).items():
        pipeline.fit(X_train.to_numpy(dtype="float32"), y_train)
        make_thresholds_float32_safe(pipeline)
        val_metrics = models.evaluate(task, pipeline, X_val, y_val, label_val)
        test_metrics = models.evaluate(task, pipeline, X_test, y_test, label_test)
        results[name] = {"model": pipeline, "validation": val_metrics, "test": test_metrics}
        if task == "anomaly" or models.is_better(task, val_metrics, best_val):
            best_name, best_val = name, val_metrics
    chosen = results[best_name]
    suspects = _leakage_suspects(data, X_train, y_train, cfg.target if cfg.horizon_steps else None)
    if suspects:
        data.notes.append(
            f"WARNING: possible target leakage in {suspects}: they (almost) determine the target on their own. "
            "If they are derived from the target or only known afterwards, add them to drop_columns"
        )

    reference = None
    if data.persistence is not None:
        # "Tomorrow is like today": the honest baseline for forecasting.
        persistence = data.persistence.iloc[split.test]
        reference = {"persistence_mae": float((persistence - y_test).abs().mean())}

    feature_names = [f.name for f in data.features]
    schema = build_feature_schema(manifest.use_case_id, data.features, X_train, cfg.units)
    descriptor: dict[str, Any] = {
        "use_case_id": manifest.use_case_id,
        "task_type": task,
        "algorithm": best_name,
        "runner": "onnx",
        "target_name": data.target_name,
        "target_unit": cfg.units.get(cfg.target) if cfg.target else None,
        "classes": data.classes,
        "features": feature_names,
        "metrics": chosen["test"],
        "candidates": {n: {"validation": r["validation"], "test": r["test"]} for n, r in results.items()},
        "reference": reference,
        "feature_importance": models.feature_importance(chosen["model"], feature_names),
        "split": {"chronological": data.chronological, "train": len(split.train), "validation": len(split.val),
                  "test": len(split.test)},
        "notes": data.notes,
        "dataset": {"path": str(source), "rows": int(len(df)), "sha256": _sha256(source)},
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "environment": _environment(),
    }

    artifact = to_onnx(chosen["model"], len(feature_names), task)
    parity = check_parity(chosen["model"], artifact, X_test, descriptor)
    descriptor["parity"] = parity
    if not parity["passed"]:
        raise TrainingError(f"ONNX export does not match the original model: {parity}")

    version = registry.publish(
        manifest.use_case_id,
        descriptor=descriptor,
        feature_schema=schema,
        artifact_name=ARTIFACT_NAME,
        artifact=artifact,
        extra_files={"manifest.snapshot.json": manifest.model_dump_json(indent=2)},
    )
    report_path = registry.root / manifest.use_case_id / version / "report.md"
    report_path.write_text(render_report(manifest, {**descriptor, "model_version": version}), encoding="utf-8")
    return {**descriptor, "model_version": version, "report": str(report_path)}


def init_use_case(
    use_case_id: str,
    data_path: Path,
    usecases_dir: Path,
    *,
    target: str | None = None,
    time_column: str | None = None,
    horizon_steps: int | None = None,
    description: str = "",
) -> tuple[Path, list[str]]:
    """Scaffold a manifest for a new dataset by profiling it. Returns (path, remarks)."""
    data_path = Path(data_path)
    adapter = adapter_for(data_path)
    df = read_table(adapter, data_path)
    remarks: list[str] = [f"{len(df)} rows, {len(df.columns)} columns: {list(df.columns)}"]
    if target and target not in df.columns:
        raise TrainingError(f"target {target!r} not in dataset columns {list(df.columns)}")

    drop = []
    for name in df.columns:
        lowered = str(name).lower()
        looks_like_id = lowered in {"id", "uid", "udi", "index"} or lowered.endswith(("_id", " id"))
        if name != target and looks_like_id and df[name].nunique() > 0.9 * len(df):
            drop.append(name)
    if drop:
        remarks.append(f"proposed to drop identifier column(s) {drop}")

    sensitive_words = ("diagnos", "illness", "ziekte", "treatment", "behandel", "bsn", "name", "naam", "email")
    sensitive = [c for c in df.columns if any(w in str(c).lower() for w in sensitive_words)]
    if sensitive:
        remarks.append(f"WARNING: column(s) {sensitive} may contain personal data; review the policy section")

    task = "auto"
    if target:
        task = detect_task(df[target])
        remarks.append(f"target {target!r} -> task '{task}'")
    else:
        task = "anomaly"
        remarks.append("no target -> anomaly detection")

    manifest = {
        "manifest_version": "1.0",
        "use_case_id": use_case_id,
        "description": description,
        "data_source": {"adapter": adapter, "path": str(data_path)},
        "training": {
            "target": target,
            "task": task,
            "time_column": time_column,
            "horizon_steps": horizon_steps,
            "drop_columns": drop,
            "rename_columns": {},
            "evaluation_label": None,
            "anomaly_share": 0.05,
            "test_fraction": 0.2,
            "units": {},
        },
        "model": {"runner": "onnx", "version": "latest"},
        "policy": {
            "contains_personal_data": bool(sensitive),
            "special_category_data": False,
            "reject_unknown_fields": True,
            "log_feature_values": not sensitive,
            "forbidden_fields": sensitive,
            "min_group_size": None,
        },
    }
    Manifest.model_validate(manifest)  # never write an invalid manifest
    directory = Path(usecases_dir) / use_case_id
    path = directory / MANIFEST_FILENAME
    if path.exists():
        raise TrainingError(f"{path} already exists")
    directory.mkdir(parents=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path, remarks


def render_report(manifest: Manifest, d: dict[str, Any]) -> str:
    lines = [
        f"# Training report: {manifest.use_case_id} v{d['model_version']}",
        "",
        manifest.description,
        "",
        f"- Task: **{d['task_type']}**" + (f", target `{d['target_name']}`" if d.get("target_name") else ""),
        f"- Chosen model: **{models.describe_algorithm(d['algorithm'])}** (selected on the validation split)",
        f"- Data: {d['dataset']['rows']} rows from `{d['dataset']['path']}`; split train/validation/test = "
        f"{d['split']['train']}/{d['split']['validation']}/{d['split']['test']}"
        + (" (chronological)" if d["split"]["chronological"] else " (random, seed 42)"),
        f"- Trained at: {d['trained_at']}",
        "",
        "## Candidates (test split)",
        "",
    ]
    metric_names = sorted({m for r in d["candidates"].values() for m in r["test"]})
    lines.append("| model | " + " | ".join(metric_names) + " |")
    lines.append("|---" * (len(metric_names) + 1) + "|")
    for name, r in d["candidates"].items():
        marker = " (chosen)" if name == d["algorithm"] else ""
        lines.append(f"| {name}{marker} | " + " | ".join(f"{r['test'].get(m, float('nan')):.4f}" for m in metric_names) + " |")
    if d.get("reference"):
        lines += ["", f"Reference, persistence forecast (value stays the same): MAE = {d['reference']['persistence_mae']:.4f}"]
    if d.get("feature_importance"):
        lines += ["", "## Feature importance (chosen model)", ""]
        lines += [f"- `{name}`: {share:.1%}" for name, share in list(d["feature_importance"].items())[:10]]
    p = d.get("parity")
    if p:
        lines += ["", "## ONNX parity check", "",
                  f"- Rows compared: {p['rows_compared']}, max abs diff: {p['max_abs_diff']:.2e}, "
                  f"label agreement: {p['label_agreement']}, passed: **{p['passed']}**"]
    if d.get("notes"):
        lines += ["", "## Notes", ""] + [f"- {n}" for n in d["notes"]]
    return "\n".join(lines) + "\n"


def _leakage_suspects(data, X_train, y_train, horizon_source: str | None) -> list[str]:
    """Features that on their own (almost) determine the target, a typical sign of leakage."""
    if y_train is None:
        return []
    suspects = []
    for spec in data.features:
        x = X_train[spec.name]
        if spec.name == horizon_source or x.nunique() < 2:
            continue  # the current value of a forecast target is a legitimate feature
        if data.task == "classification" and spec.kind == "binary":
            ones = y_train[x == 1]
            if len(ones) >= 10 and ones.nunique() == 1 and ones.iloc[0] != y_train.mode().iloc[0]:
                suspects.append(spec.name)  # x = 1 always means the rare class
        elif data.task == "regression" and abs(x.corr(y_train)) >= 0.99:
            suspects.append(spec.name)
    return suspects


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _environment() -> dict[str, str]:
    env = {"python": platform.python_version()}
    for pkg in ("scikit-learn", "lightgbm", "skl2onnx", "onnxruntime", "pandas"):
        try:
            env[pkg] = pkg_version(pkg)
        except Exception:  # noqa: BLE001 - informational only
            pass
    return env

