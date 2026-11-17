"""Training orchestration: manifest + dataset -> versioned model in the registry."""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import version as pkg_version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone

from predictor.adapters.fs_registry import FilesystemRegistry, PublishedOutput
from predictor.adapters.sources import adapter_for, read_table
from predictor.core import ranges
from predictor.core.certainty import SupportIndex, build_support, certainty
from predictor.core.manifest import ALGORITHMS, MANIFEST_FILENAME, Manifest, TrainingConfig, load_manifest
from predictor.trainer import models
from predictor.trainer.export import check_parity, make_thresholds_float32_safe, to_onnx
from predictor.trainer.errors import TrainingError
from predictor.trainer.join import apply_joins
from predictor.trainer.profile import SharedFeatures, build_feature_schema, detect_task, prepare, shared_features

EVALUATION_SAMPLES = 500


@dataclass
class _Table:
    """The base table with its joins applied, read once per training run."""

    df: pd.DataFrame
    source: Path
    base_rows: int
    joined: Any  # trainer.join.AppliedJoins | None


@dataclass
class _Trained:
    """One trained model plus what has to be published with it."""

    descriptor: dict[str, Any]
    schema: dict[str, Any]
    artifact: bytes
    support_file: dict[str, str]


def _load_table(manifest: Manifest, data_path: Path | None) -> _Table:
    source = data_path or manifest.data_source.path
    if not source:
        raise TrainingError(f"{manifest.use_case_id}: no dataset given (use --data or data_source.path)")
    source = Path(source)
    if not source.is_file():
        raise TrainingError(f"{manifest.use_case_id}: dataset {source} not found")
    adapter = manifest.data_source.adapter if data_path is None else adapter_for(source)
    df = read_table(adapter, source)
    # Joins run first, on the raw base columns (before rename_columns): lookup tables become ordinary columns.
    base_rows = len(df)
    joined = apply_joins(manifest, df) if manifest.joins else None
    return _Table(joined.df if joined else df, source, base_rows, joined)


def _train_output(manifest: Manifest, cfg: TrainingConfig, table: _Table, allowed: list[str] | None = None,
                  shared: SharedFeatures | None = None) -> _Trained:
    """Train, choose and export the model for the single target of ``cfg`` (profile, split, candidates, ONNX)."""
    df, source, base_rows, joined = table.df, table.source, table.base_rows, table.joined
    if joined and joined.consumed:  # dates that only place a window are join inputs, not model features
        cfg = cfg.model_copy(update={"drop_columns": [*cfg.drop_columns, *joined.consumed]})

    data = prepare(df, cfg, manifest.policy, shared)
    if joined:
        data.notes[:0] = joined.notes
    task = data.task
    feature_names = [f.name for f in data.features]
    monotone = _scenario_constraints(manifest, data, cfg.target)
    bounds = ((cfg.target_min, cfg.target_max)
              if task == "regression" and (cfg.target_min is not None or cfg.target_max is not None) else None)
    test_start = None
    if cfg.test_from:
        test_start = int((data.times < pd.Timestamp(cfg.test_from)).sum())
        if test_start < 50 or len(data.X) - test_start < 10:
            raise TrainingError(f"test_from {cfg.test_from} leaves {test_start} rows to train on and "
                                f"{len(data.X) - test_start} to test on; pick another date")
    test_mask = None
    if cfg.test_group:
        test_mask = data.groups.isin([str(v) for v in cfg.test_group.values]).to_numpy()
        if test_mask.sum() < 10 or (~test_mask).sum() < 50:
            raise TrainingError(f"test_group {cfg.test_group.values} leaves {int((~test_mask).sum())} rows to train on "
                                f"and {int(test_mask.sum())} to test on")
    blocks = None
    if cfg.test_blocks:
        # The same calendar blocks for every site; every n-th one is the test set.
        tb = cfg.test_blocks
        blocks = ((data.times - data.times.min()).dt.days // tb.days).to_numpy()
        test_mask = blocks % tb.every == tb.every - 1
        if test_mask.sum() < 10 or (~test_mask).sum() < 50:
            raise TrainingError(f"test_blocks (every {tb.every}th block of {tb.days} days) leaves "
                                f"{int((~test_mask).sum())} rows to train on and {int(test_mask.sum())} to test on")
    split = models.make_split(len(data.X), cfg.test_fraction, data.chronological,
                              data.y if task == "classification" else None, test_start, test_mask)
    X_train, X_val, X_test = (data.X.iloc[idx] for idx in (split.train, split.val, split.test))
    y_train = y_val = y_test = None
    if data.y is not None:
        y_train, y_val, y_test = (data.y.iloc[idx] for idx in (split.train, split.val, split.test))
    label_val = label_test = None
    if data.evaluation_label is not None:
        label_val, label_test = data.evaluation_label.iloc[split.val], data.evaluation_label.iloc[split.test]

    # Fit on train, choose on validation, then refit on train + validation: with a chronological
    # split the validation part is the most recent data, which the final model must not miss.
    X_fit = pd.concat([X_train, X_val])
    y_fit = pd.concat([y_train, y_val]) if y_train is not None else None
    # When the test is "a group the model never saw", choose the model on that same question:
    # leave each training group out in turn (group cross-validation) instead of the last 20% in time.
    fit_groups, fold_name = None, None
    fit_index = np.concatenate([split.train, split.val])
    if cfg.test_group and data.y is not None:
        fit_groups = data.groups.iloc[fit_index].reset_index(drop=True)
        fold_name = cfg.test_group.column
    elif blocks is not None and data.y is not None:
        # The same question as the test: leave whole blocks out in turn.
        fit_groups = pd.Series(blocks[fit_index] % cfg.test_blocks.every).astype(str)
        fold_name = f"set of {cfg.test_blocks.days}-day blocks"
    if fit_groups is not None and fit_groups.nunique() < 3:
        fit_groups = None
    results: dict[str, dict[str, Any]] = {}
    best_name, best_val = None, None
    excluded: dict[str, list[str]] = {}
    for name, pipeline in models.candidates(task, cfg.anomaly_share, monotone, allowed).items():
        unseen = None   # (rows, true values, predictions) the candidate did not learn from: calibrates the range
        if fit_groups is not None:
            val_metrics, out_of_fold = _group_cross_validation(task, pipeline, X_fit, y_fit, fit_groups, bounds, True)
            unseen = (X_fit, y_fit, out_of_fold)
        else:
            pipeline.fit(X_train.to_numpy(dtype="float32"), y_train)
            make_thresholds_float32_safe(pipeline)
            val_metrics = models.evaluate(task, pipeline, X_val, y_val, label_val, bounds)
            if task == "regression":
                unseen = (X_val, y_val, _bounded(pipeline.predict(X_val.to_numpy(dtype="float32")), bounds))
        candidate = pipeline
        final = clone(pipeline).fit(X_fit.to_numpy(dtype="float32"), y_fit)
        make_thresholds_float32_safe(final)
        test_metrics = models.evaluate(task, final, X_test, y_test, label_test, bounds)
        results[name] = {"model": final, "validation": val_metrics, "test": test_metrics, "unseen": unseen,
                         "pipeline": candidate}
        violations = models.effect_violations(final, feature_names, monotone)
        if violations:
            # A scenario model must respect what the manifest declares, so it cannot win.
            excluded[name] = violations
            data.notes.append(f"{name} was not eligible: it contradicts the declared effects ({'; '.join(violations)})")
            continue
        if task == "anomaly" or models.is_better(task, val_metrics, best_val):
            best_name, best_val = name, val_metrics
    if best_name is None:
        raise TrainingError(f"no candidate respects the declared scenario effects: {excluded}")
    if fit_groups is not None:
        simpler = models.one_standard_error_choice(task, {n: r["validation"] for n, r in results.items()
                                                          if n not in excluded}, best_name)
        if simpler != best_name:
            data.notes.append(f"{simpler} is within one standard error of {best_name} in the cross-validation, so "
                              "the simpler model is used (smoother effects, less risk on new situations)")
            best_name = simpler
    chosen = results[best_name]
    # With test_blocks the held-out blocks only measure: the published model learns from every row.
    new_group = None
    published, X_published = chosen["model"], X_fit
    if cfg.test_blocks:
        pipeline = models.candidates(task, cfg.anomaly_share, monotone, [best_name])[best_name]
        if data.groups is not None and data.groups.nunique() >= 3:
            # The harder question next to it: a site the model has never seen (each site left out in turn).
            new_group = {"column": cfg.test_blocks.group, "groups": int(data.groups.nunique()),
                         "metrics": None}
            new_group["metrics"], new_predictions = _group_cross_validation(task, pipeline, data.X, data.y, data.groups,
                                                                            bounds, True)
            if new_predictions is not None:
                chosen["new_group_errors"] = np.abs(data.y.to_numpy(dtype=float) - new_predictions).tolist()
        published = clone(pipeline).fit(data.X.to_numpy(dtype="float32"), data.y)
        make_thresholds_float32_safe(published)
        X_published = data.X
    pinned = models.pinned_effects(published, feature_names, monotone)
    if pinned:
        data.notes.append(f"the data shows no effect of {pinned} in the declared direction, so the linear model "
                          "holds it at 0 (more data on those levers would help)")
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
        label = cfg.baseline.label if cfg.baseline else "it stays the same"
        reference = {"persistence_mae": float((persistence - y_test).abs().mean()), "label": label}

    schema = build_feature_schema(manifest.use_case_id, data.features, X_train, cfg.units)
    descriptor: dict[str, Any] = {
        "use_case_id": manifest.use_case_id,
        "output": cfg.target or "anomaly",
        "task_type": task,
        "algorithm": best_name,
        "runner": "onnx",
        "target_name": data.target_name,
        "target_unit": cfg.units.get(cfg.target) if cfg.target else None,
        "target_bounds": ([cfg.target_min, cfg.target_max]
                          if task == "regression" and (cfg.target_min is not None or cfg.target_max is not None)
                          else None),
        "classes": data.classes,
        "features": feature_names,
        "metrics": chosen["test"],
        "candidates": {n: {"validation": r["validation"], "test": r["test"]} for n, r in results.items()},
        "reference": reference,
        "feature_importance": models.feature_importance(published, feature_names),
        "scenario_constraints": ({"effects": {n: ("increasing" if d > 0 else "decreasing")
                                              for n, d in zip(feature_names, monotone) if d},
                                  "excluded_candidates": excluded, "held_at_zero": pinned} if monotone else None),
        "split": {"chronological": data.chronological, "test_from": cfg.test_from.isoformat() if cfg.test_from else None,
                  "test_group": cfg.test_group.model_dump() if cfg.test_group else None,
                  "test_blocks": cfg.test_blocks.model_dump() if cfg.test_blocks else None,
                  "train": len(split.train), "validation": len(split.val),
                  "test": len(split.test),
                  "published_on": "all rows" if cfg.test_blocks else "train + validation",
                  "validation_method": (f"leave one {fold_name} out ({fit_groups.nunique()} folds)"
                                        if fit_groups is not None else "hold-out")},
        "new_group": new_group,
        "evaluation": _evaluation_sample(data, chosen["model"], X_train, X_test, y_test, label_test,
                                         split.test, manifest.policy.evaluation_rows_allowed),
        "notes": data.notes,
        "dataset": {"path": str(source), "rows": base_rows, "sha256": _sha256(source),
                    **({"rows_after_joins": int(len(df))} if joined else {})},
        "joins": joined.summary if joined else [],
        **({"joins_example": _joins_example(joined, df)} if joined and joined.engine and manifest.policy.evaluation_rows_allowed else {}),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "environment": _environment(),
    }

    # Map of the training data for "comparable cases" in scenarios. Only where it is needed
    # (a scenario section) and allowed (it holds transformed training rows: no personal data by default).
    support_file: dict[str, str] = {}
    support = None
    if manifest.scenario is not None:
        if manifest.policy.evaluation_rows_allowed:
            support = build_support(X_published.to_numpy(dtype="float64"), [f.kind for f in data.features], feature_names,
                                    [f.categories for f in data.features], descriptor["feature_importance"])
            support_file["support.json"] = json.dumps(support)  # stored in this output's own folder
            descriptor["certainty"] = {"available": True, "training_rows": support["rows"],
                                       "radius": round(support["radius"], 6)}
        else:
            descriptor["certainty"] = {"available": False,
                                       "reason": "not stored for use cases with personal data"}
    if task == "regression" and fit_groups is None and len(X_fit) >= 100:
        # One validation period says little about how much periods differ (a good or a bad weather year). So the
        # range is measured on five held-out parts of the training data: consecutive periods for data in time order.
        part = (np.arange(len(X_fit)) * 5 // len(X_fit) if data.chronological
                else np.random.default_rng(models.SEED).permutation(len(X_fit)) % 5)
        _, out_of_fold = _group_cross_validation(task, chosen["pipeline"], X_fit, y_fit, pd.Series(part), bounds, True)
        chosen["unseen"] = (X_fit, y_fit, out_of_fold)
    if task == "regression" and chosen["unseen"] is not None:
        descriptor["range"] = _range(chosen, support, X_test, y_test, bounds, test_in_support=bool(cfg.test_blocks))

    artifact = to_onnx(published, len(feature_names), task)
    parity = check_parity(published, artifact, X_test, descriptor)
    descriptor["parity"] = parity
    if not parity["passed"]:
        raise TrainingError(f"ONNX export does not match the original model: {parity}")
    return _Trained(descriptor, schema, artifact, support_file)


def train_use_case(manifest_path: Path, registry: FilesystemRegistry, data_path: Path | None = None) -> dict[str, Any]:
    """Train every output of a use case on the same joined data and publish them as one version.

    The result has the flat keys of the first output (what single-output callers expect) plus ``outputs``:
    name -> descriptor of every output.
    """
    manifest = load_manifest(manifest_path)
    table = _load_table(manifest, data_path)
    configs = manifest.training.output_configs()
    allowed = _check_algorithms(manifest, table, configs)  # before anything is trained
    # With several outputs the inputs are profiled once on all rows, so every output encodes them the same way.
    shared = _shared_features(manifest, table) if len(configs) > 1 else None
    trained: dict[str, _Trained] = {}
    for name in configs or ["anomaly"]:  # without a target the use case is anomaly detection
        view = manifest.training.for_output(name) if name in configs else manifest.training
        trained[name] = _train_output(manifest, view, table, allowed.get(name), shared)
    first_name, first = next(iter(trained.items()))
    for name, other in trained.items():
        if _encoding(other.schema) != _encoding(first.schema):
            raise TrainingError(
                f"{manifest.use_case_id}: output {name!r} ended up with other inputs than {first_name!r} "
                f"({sorted(set(_encoding(other.schema).items()) ^ set(_encoding(first.schema).items()))[:4]}); "
                "all outputs of a use case share one set of inputs, so fix the data or split the use case")
    joined = table.joined
    version = registry.publish(
        manifest.use_case_id,
        outputs={name: PublishedOutput(t.descriptor, t.artifact, t.support_file) for name, t in trained.items()},
        feature_schema=first.schema,
        extra_files={"manifest.snapshot.json": manifest.model_dump_json(indent=2),
                     **(joined.engine.snapshot() if joined and joined.engine else {})},
    )
    described = {name: {**t.descriptor, "model_version": version} for name, t in trained.items()}
    report_path = registry.root / manifest.use_case_id / version / "report.md"
    report_path.write_text(render_report(manifest, described[first_name], described), encoding="utf-8")
    return {**described[first_name], "report": str(report_path), "outputs": described}


def _encoding(schema: dict[str, Any]) -> dict[str, Any]:
    """What a model needs to read an input: its name, kind and category order (not the observed range)."""
    return {name: (prop.get("x-kind"), tuple(prop.get("enum") or ())) for name, prop in schema["properties"].items()}


def _shared_features(manifest: Manifest, table: _Table) -> SharedFeatures:
    cfg = manifest.training
    if table.joined and table.joined.consumed:  # same dates-that-only-place-a-window rule as for every output
        cfg = cfg.model_copy(update={"drop_columns": [*cfg.drop_columns, *table.joined.consumed]})
    return shared_features(table.df, cfg, manifest.policy)


def _check_algorithms(manifest: Manifest, table: _Table, configs: dict) -> dict[str, list[str] | None]:
    """The models allowed per output, checked for the real task of each output before the first one is trained.

    An ``algorithms`` list on an output must all exist for its task; the default (``model.algorithms``) is cut down to
    the names that exist for the task of that output.
    """
    default = manifest.model.algorithms
    if not configs:  # anomaly detection
        return {"anomaly": [n for n in default if n in ALGORITHMS["anomaly"]] or None} if default else {}
    allowed: dict[str, list[str] | None] = {}
    for name, config in configs.items():
        task = config.task
        if task == "auto" and name in table.df.columns:
            task = detect_task(table.df[name])
        if task not in ALGORITHMS:  # the column is missing: prepare() reports that
            allowed[name] = config.algorithms or default
            continue
        if config.algorithms:
            unknown = [n for n in config.algorithms if n not in ALGORITHMS[task]]
            if unknown:
                raise TrainingError(f"{manifest.use_case_id}: output {name!r} is a '{task}' output, so the model(s) {unknown} "
                                    f"are not available for it (available: {', '.join(ALGORITHMS[task])}); fix 'algorithms'")
            allowed[name] = config.algorithms
        elif default:
            fitting = [n for n in default if n in ALGORITHMS[task]]
            if not fitting:
                raise TrainingError(f"{manifest.use_case_id}: model.algorithms {default} has no model for output {name!r} "
                                    f"(a '{task}' output; available: {', '.join(ALGORITHMS[task])}); set 'algorithms' on it")
            allowed[name] = fitting
        else:
            allowed[name] = None
    return allowed


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


def render_report(manifest: Manifest, d: dict[str, Any], outputs: dict[str, dict[str, Any]] | None = None) -> str:
    """The training report. With several ``outputs`` (name -> descriptor) there is one section per output."""
    lines = [f"# Training report: {manifest.use_case_id} v{d['model_version']}", "", manifest.description, ""]
    if not outputs or len(outputs) == 1:
        return "\n".join(lines + _output_report(manifest, d)) + "\n"
    lines += [f"One training run, {len(outputs)} outputs, each with its own model: " + ", ".join(f"`{n}`" for n in outputs), ""]
    for i, (name, descriptor) in enumerate(outputs.items()):
        shown = {**descriptor, "model_version": d["model_version"], **({"joins": []} if i else {})}  # joins: once
        section = _output_report(manifest, shown)
        lines += [f"## Output: {name}", ""] + [("#" + line if line.startswith("## ") else line) for line in section]
    return "\n".join(lines) + "\n"


def _output_report(manifest: Manifest, d: dict[str, Any]) -> list[str]:
    lines = [
        f"- Task: **{d['task_type']}**" + (f", target `{d['target_name']}`" if d.get("target_name") else ""),
        f"- Chosen model: **{models.describe_algorithm(d['algorithm'])}** (selected on "
        + (f"{d['split']['validation_method']} cross-validation" if d["split"].get("validation_method", "hold-out") != "hold-out"
           else "the validation split") + f", then retrained on {d['split'].get('published_on', 'train + validation')})",
        f"- Data: {d['dataset']['rows']} rows from `{d['dataset']['path']}`; split train/validation/test = "
        f"{d['split']['train']}/{d['split']['validation']}/{d['split']['test']}"
        + (f" (test on unseen {d['split']['test_group']['column']}: {', '.join(d['split']['test_group']['values'])})"
           if d["split"].get("test_group")
           else f" (test on held-out blocks: every {d['split']['test_blocks']['every']}th block of "
                f"{d['split']['test_blocks']['days']} days)" if d["split"].get("test_blocks")
           else f" (backtest: test from {d['split']['test_from']})" if d["split"].get("test_from")
           else " (chronological)" if d["split"]["chronological"] else " (random, seed 42)"),
        f"- Trained at: {d['trained_at']}",
        *_new_group_lines(d),
        *_range_lines(d),
        "",
        *_joins_section(d.get("joins")),
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
        ref = d["reference"]
        better = d["metrics"]["mae"] < ref["persistence_mae"]
        lines += ["", f"Reference, simple guess ({ref.get('label', 'it stays the same')}): MAE = {ref['persistence_mae']:.4f}"
                  + ("" if better else " -- the simple guess is still better than the model")]
    if d.get("feature_importance"):
        lines += ["", "## Feature importance (chosen model)", ""]
        lines += [f"- `{name}`: {share:.1%}" for name, share in list(d["feature_importance"].items())[:10]]
    if manifest.scenario:
        target = d.get("output") or manifest.training.target
        lines += ["", "## Scenario levers", "", "| lever | range | step | declared effect on " + f"`{target}` |", "|---|---|---|---|"]
        for name, lever in manifest.scenario.levers.items():
            effect = lever.effects.get(target, "none (free)")
            lines.append(f"| `{name}` | {lever.min:g} - {lever.max:g} | {lever.step:g} | {effect} |")
        constraints = d.get("scenario_constraints")
        if constraints:
            lines += ["", "Declared effects are hard constraints: monotone constraints for LightGBM and sign "
                      "constraints for the linear model (a coefficient the data gives the wrong sign becomes 0). "
                      "A model that still contradicts them is not eligible."]
            if constraints.get("held_at_zero"):
                lines.append(f"- held at 0 (no effect in the declared direction in the data): "
                             f"{', '.join(f'`{n}`' for n in constraints['held_at_zero'])}")
            for name, why in constraints["excluded_candidates"].items():
                lines.append(f"- `{name}` excluded: {'; '.join(why)}")
    p = d.get("parity")
    if p:
        lines += ["", "## ONNX parity check", "",
                  f"- Rows compared: {p['rows_compared']}, max abs diff: {p['max_abs_diff']:.2e}, "
                  f"label agreement: {p['label_agreement']}, passed: **{p['passed']}**"]
    if d.get("notes"):
        lines += ["", "## Notes", ""] + [f"- {n}" for n in d["notes"]]
    return lines




def _range_lines(d: dict[str, Any]) -> list[str]:
    r = d.get("range")
    if not r:
        return []
    unit = f" {d['target_unit']}" if d.get("target_unit") else ""
    widths = (", ".join(f"{name} certainty ±{w:.4g}" for name, w in r["by_certainty"].items()) if r.get("by_certainty")
              else f"±{r['half_width']:.4g}")
    lines = [f"- {r['level']:.0%} range around a prediction: {widths}{unit} (measured on {r['calibration_rows']} rows the model "
             f"did not learn from); on the test rows **{r['test_coverage']:.0%}** fell inside"]
    if r.get("new_group_half_width") is not None:
        lines.append(f"- Low certainty is at least as wide as the error on a {d['new_group']['column']} the model never saw "
                     f"(±{r['new_group_half_width']:.4g}{unit}); outside the training range of an input no range is given")
    return lines


def _new_group_lines(d: dict[str, Any]) -> list[str]:
    """Two honest numbers: a site with its own history (the test) and a site the model never saw."""
    new = d.get("new_group")
    if not new:
        return []
    metric = models.PRIMARY_METRIC[d["task_type"]][0]
    return [f"- With its own history (the test above): {metric} **{d['metrics'][metric]:.4f}**; for a "
            f"{new['column']} the model has never seen (each of {new['groups']} left out in turn): "
            f"{metric} **{new['metrics'][metric]:.4f}**"]


def _group_cross_validation(task: str, pipeline, X: pd.DataFrame, y: pd.Series, groups: pd.Series,
                            bounds: tuple | None, with_predictions: bool = False):
    """Average metrics over folds that each leave one group (greenhouse, farm) out.

    With `with_predictions` also the prediction for every row from the model that did not see its group.
    """
    folds = []
    out_of_fold = np.full(len(X), np.nan)
    for group in groups.unique():
        held = (groups == group).to_numpy()
        model = clone(pipeline).fit(X[~held].to_numpy(dtype="float32"), y[~held])
        folds.append(models.evaluate(task, model, X[held], y[held], None, bounds))
        if with_predictions and task == "regression":
            out_of_fold[held] = _bounded(model.predict(X[held].to_numpy(dtype="float32")), bounds)
    metrics = {k: float(np.mean([f[k] for f in folds])) for k in folds[0]}
    metrics.update({f"{k}_se": float(np.std([f[k] for f in folds], ddof=1) / np.sqrt(len(folds))) for k in folds[0]})
    return (metrics, out_of_fold if task == "regression" else None) if with_predictions else metrics


def _bounded(values: np.ndarray, bounds: tuple | None) -> np.ndarray:
    if not bounds:
        return np.asarray(values, dtype=float)
    return np.clip(values, bounds[0] if bounds[0] is not None else -np.inf, bounds[1] if bounds[1] is not None else np.inf)


def _range(chosen: dict[str, Any], support: dict[str, Any] | None, X_test: pd.DataFrame, y_test: pd.Series,
           bounds: tuple | None, test_in_support: bool) -> dict[str, Any] | None:
    """Calibrate the 80% range on rows the chosen model did not learn from, then check it on the test rows."""
    X_unseen, y_unseen, predicted = chosen["unseen"]
    errors = np.abs(np.asarray(y_unseen, dtype=float) - predicted)
    index = SupportIndex(support) if support else None

    def levels(X: pd.DataFrame, own_rows: bool) -> list[str] | None:
        if index is None:
            return None
        # A row that is itself in the map must not count as its own comparable case.
        own = 1 if own_rows and index.scale == 1.0 else 0
        return [certainty(max(c - own, 0))["level"] for c in index.comparable(X.to_numpy(dtype="float64"))]

    spec = ranges.build(errors.tolist(), levels(X_unseen, own_rows=True), new_group_errors=chosen.get("new_group_errors"))
    if spec is None:
        return None
    test_errors = np.abs(np.asarray(y_test, dtype=float) - _bounded(chosen["model"].predict(X_test.to_numpy(dtype="float32")), bounds))
    test_levels = levels(X_test, own_rows=test_in_support)
    widths = np.array([ranges.half_width(spec, lv) for lv in (test_levels or [None] * len(test_errors))])
    spec["test_coverage"] = round(float(np.mean(test_errors <= widths)), 4)
    if test_levels and "by_certainty" in spec:
        spec["test_coverage_by_certainty"] = {
            name: {"rows": int(sum(lv == name for lv in test_levels)),
                   "coverage": round(float(np.mean(test_errors[[lv == name for lv in test_levels]] <= ranges.half_width(spec, name))), 4)}
            for name in ("low", "medium", "high") if any(lv == name for lv in test_levels)}
    return spec


def _joins_section(joins: list[dict[str, Any]] | None) -> list[str]:
    """What was connected to the base table, and how well it matched (empty without joins)."""
    if not joins:
        return []
    lines = ["## Joins", "", "| join | lookup | connected by | new columns | matched | dropped |", "|---|---|---|---|---|---|"]
    for j in joins:
        by = ", ".join(f"{b} = {k}" for b, k in j["keys"].items())
        if j.get("window"):
            w = j["window"]
            edge = lambda e: (e["column"] or j["time"]["base"]) + (f" {e['offset']:+d} {e['unit']}" if e["offset"] else "")
            by = (by + "; " if by else "") + f"window [{edge(w['start'])}, {edge(w['end'])})"
        matched = f"{j['matched']:.1%}" if "matched" in j else "-"
        lines.append(f"| `{j['name']}` | `{j['source']}` ({j['lookup_rows']} rows) | {by} | "
                     f"{', '.join(f'`{c}`' for c in j['columns'])} | {matched} | {j.get('dropped_rows', 0)} |")
    for j in joins:
        lines += [f"- {j['name']}: {m}" for m in j.get("warnings", [])]
    lines += ["", "At prediction time the service derives these columns from the same lookup tables (snapshot stored with "
              "the model); a value sent by the caller wins.", ""]
    return lines


def _evaluation_sample(data, model, X_train, X_test, y_test, label_test, test_index,
                       rows_allowed: bool) -> dict[str, Any]:
    """A small, chart-ready summary of how the chosen model does on the test split.

    Row-level samples (actual vs. predicted, anomaly scores) are withheld unless the
    policy allows them (by default only without personal data); aggregates such as a
    confusion matrix are always kept.
    """
    X = X_test.to_numpy(dtype="float32")
    sample: dict[str, Any] = {"example_input": _typical_input(data.features, X_train), "samples_withheld": None}
    if data.task == "classification":
        pred = model.predict(X)
        n = len(data.classes)
        matrix = [[0] * n for _ in range(n)]
        for actual, predicted in zip(y_test, pred):
            matrix[int(actual)][int(predicted)] += 1
        sample["confusion_matrix"] = {"classes": data.classes, "counts": matrix}  # rows = actual
        return sample
    if not rows_allowed:
        sample["samples_withheld"] = "row-level results are not published for use cases with personal data"
        return sample
    rows = slice(0, EVALUATION_SAMPLES)
    if data.task == "regression":
        sample["regression"] = {
            "actual": [round(float(v), 4) for v in y_test.iloc[rows]],
            "predicted": [round(float(v), 4) for v in model.predict(X[rows])],
            "persistence": ([None if pd.isna(v) else round(float(v), 4) for v in data.persistence.iloc[test_index[rows]]]
                            if data.persistence is not None else None),
            "time": ([t.isoformat() for t in data.times.iloc[test_index[rows]]]
                     if data.times is not None else None),
        }
    else:
        rows = slice(0, EVALUATION_SAMPLES * 10)  # rare labels need more rows to show a distribution
        scores = model.decision_function(X[rows])
        sample["anomaly"] = {
            "scores": [round(float(s), 4) for s in scores],
            "threshold": 0.0,  # IsolationForest: score < 0 is flagged as an anomaly
            "label": [bool(v) for v in models._truthy(label_test.iloc[rows])] if label_test is not None else None,
        }
    return sample


def _joins_example(joined, df: pd.DataFrame) -> dict[str, str]:
    """The join inputs (compartment, start date, ...) of one real, typical row: consistent defaults for a form."""
    row = df.iloc[len(df) // 2]
    return {c: str(row[c]) for c in joined.engine.inputs if c not in joined.engine.derived and c in df.columns}


def _typical_input(features, X_train) -> dict[str, Any]:
    """Median of every numeric feature and the most common value of the others."""
    example: dict[str, Any] = {}
    for spec in features:
        column = X_train[spec.name].dropna()
        if spec.kind == "categorical":
            example[spec.name] = spec.categories[int(column.mode().iloc[0])]
        elif spec.kind == "binary":
            example[spec.name] = int(column.mode().iloc[0])
        elif spec.kind == "integer":
            example[spec.name] = int(round(float(column.median())))
        else:
            example[spec.name] = round(float(column.median()), 3)
    return example


def _scenario_constraints(manifest: Manifest, data, target: str | None) -> list[int] | None:
    """Check the scenario levers against the prepared data and build the monotone constraints."""
    if manifest.scenario is None:
        return None
    kinds = {f.name: f.kind for f in data.features}
    for name in manifest.scenario.levers:
        if name not in kinds:
            raise TrainingError(f"scenario lever {name!r} is not a model input after preparing the data "
                                f"(missing, constant or dropped; inputs: {', '.join(kinds)})")
        if kinds[name] == "categorical":
            raise TrainingError(f"scenario lever {name!r} is categorical; only numeric and yes/no inputs can be levers")
    directions = {name: lever.effects[target] for name, lever in manifest.scenario.levers.items()
                  if target in lever.effects}
    if not directions:
        return None
    if data.task == "classification" and len(data.classes or []) > 2:
        raise TrainingError("scenario effects are only supported for regression and yes/no classification")
    return [{"increasing": 1, "decreasing": -1}.get(directions.get(f.name), 0) for f in data.features]


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

