"""The steps behind ``predictor setup``, shared by the terminal flow and the browser wizard."""

from __future__ import annotations

from typing import Callable

from predictor import bootstrap
from predictor.core.doctor import Finding, has_errors, model_state, run_doctor

Log = Callable[[str], None]


def doctor() -> list[Finding]:
    return run_doctor(bootstrap.home(), bootstrap.usecases_dir(), bootstrap.models_dir(), bootstrap.datasets_dir(),
                      known_sources=set(bootstrap.SOURCES), known_runners=set(bootstrap.RUNNERS))


def trainable() -> list[str]:
    """Use cases without a model whose data is all on disk."""
    ids = []
    for manifest in bootstrap.load_manifests():
        paths = [p for p in [manifest.data_source.path, *(j.source.path for j in manifest.joins)] if p]
        if model_state(bootstrap.models_dir(), manifest.use_case_id)[0] is None and \
                paths and all((bootstrap.home() / p).is_file() for p in paths):
            ids.append(manifest.use_case_id)
    return ids


def trained() -> list[str]:
    """Use cases that have a complete model."""
    ids = []
    for manifest in bootstrap.load_manifests():
        version, problems = model_state(bootstrap.models_dir(), manifest.use_case_id)
        if version is not None and not problems:
            ids.append(manifest.use_case_id)
    return ids


def is_set_up() -> bool:
    """Ready to just start: nothing wrong and at least one model to show."""
    return not has_errors(doctor()) and bool(trained())


def format_result(result: dict) -> list[str]:
    several = len(result["outputs"]) > 1
    lines = []
    for name, d in result["outputs"].items():
        metrics = ", ".join(f"{k}={v:.4f}" for k, v in d["metrics"].items())
        label = f"{result['use_case_id']} [{name}]" if several else result["use_case_id"]
        lines.append(f"{label}: v{result['model_version']} {d['algorithm']} ({metrics}) "
                     f"parity max diff {d['parity']['max_abs_diff']:.1e}")
        lines += [f"  {n}" for n in d["notes"] if n.startswith("WARNING")]
    for join in result.get("joins", []):
        lines.append(f"  join {join['name']}: {join.get('matched', 0):.1%} matched, "
                     f"{join.get('dropped_rows', 0)} rows dropped")
    lines.append(f"  report: {result['report']}")
    return lines


def train_ids(ids: list[str], log: Log = print) -> None:
    from predictor.core.manifest import MANIFEST_FILENAME
    from predictor.trainer.run import train_use_case

    registry = bootstrap.build_registry()
    for use_case_id in ids:
        log(f"training {use_case_id} ...")
        for line in format_result(train_use_case(bootstrap.usecases_dir() / use_case_id / MANIFEST_FILENAME, registry)):
            log(line)

