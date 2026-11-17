"""Command line: ``predictor init | train | check-joins | list | serve``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from predictor import bootstrap
from predictor.core.manifest import MANIFEST_FILENAME, ManifestError
from predictor.trainer.errors import TrainingError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="predictor", description="Modular prediction backend")
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help="profile a new dataset and scaffold its use-case manifest")
    p_init.add_argument("use_case_id")
    p_init.add_argument("--data", required=True, type=Path)
    p_init.add_argument("--target", help="column to predict; omit for anomaly detection")
    p_init.add_argument("--time-column")
    p_init.add_argument("--horizon-steps", type=int)
    p_init.add_argument("--description", default="")

    p_train = sub.add_parser("train", help="train a use case and register a new model version")
    p_train.add_argument("use_cases", nargs="*", help="use-case ids or manifest paths (default: all)")
    p_train.add_argument("--data", type=Path, help="override the dataset path from the manifest")

    p_check = sub.add_parser("check-joins", help="check the joins of use cases (keys, dates, coverage), "
                                                 "or suggest how two files could be joined")
    p_check.add_argument("use_cases", nargs="*", help="use-case ids or manifest paths (default: all with joins)")
    p_check.add_argument("--suggest", nargs=2, type=Path, metavar=("BASE", "LOOKUP"),
                         help="propose key and date columns to join two data files")

    sub.add_parser("list", help="list use cases and their model versions")

    p_serve = sub.add_parser("serve", help="start the HTTP API")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)

    args = parser.parse_args(argv)
    try:
        return {"init": _init, "train": _train, "check-joins": _check_joins, "list": _list,
                "serve": _serve}[args.command](args)
    except ManifestError as exc:
        print(f"manifest error: {exc}", file=sys.stderr)
        return 2
    except TrainingError as exc:
        print(f"training error: {exc}", file=sys.stderr)
        return 2


def _init(args: argparse.Namespace) -> int:
    from predictor.trainer.run import init_use_case

    path, remarks = init_use_case(
        args.use_case_id, args.data, bootstrap.usecases_dir(), target=args.target,
        time_column=args.time_column, horizon_steps=args.horizon_steps, description=args.description,
    )
    for remark in remarks:
        print(f"  - {remark}")
    print(f"wrote {path}\nreview it, then run: predictor train {args.use_case_id}")
    return 0


def _train(args: argparse.Namespace) -> int:
    from predictor.trainer.run import train_use_case

    manifests = bootstrap.load_manifests()
    parents = {alias: m.use_case_id for m in manifests for alias in m.aliases}  # training an alias trains its use case
    targets = list(dict.fromkeys(parents.get(t, t) for t in args.use_cases)) or [m.use_case_id for m in manifests]
    if args.data and len(targets) != 1:
        print("--data can only be used with a single use case", file=sys.stderr)
        return 2
    registry = bootstrap.build_registry()
    for target in targets:
        path = Path(target)
        if not path.exists():
            path = bootstrap.usecases_dir() / target / MANIFEST_FILENAME
        result = train_use_case(path, registry, args.data)
        several = len(result["outputs"]) > 1
        for name, d in result["outputs"].items():
            metrics = ", ".join(f"{k}={v:.4f}" for k, v in d["metrics"].items())
            label = f"{result['use_case_id']} [{name}]" if several else result["use_case_id"]
            print(f"{label}: v{result['model_version']} {d['algorithm']} ({metrics}) "
                  f"parity max diff {d['parity']['max_abs_diff']:.1e}")
            for note in d["notes"]:
                if note.startswith("WARNING"):
                    print(f"  {note}")
        for join in result.get("joins", []):
            print(f"  join {join['name']}: {join.get('matched', 0):.1%} matched, {join.get('dropped_rows', 0)} rows dropped")
        print(f"  report: {result['report']}")
    return 0


def _check_joins(args: argparse.Namespace) -> int:
    from predictor.adapters.sources import adapter_for, read_table
    from predictor.core.join_checks import suggest_joins
    from predictor.core.manifest import load_manifest
    from predictor.trainer.join import apply_joins, arrays, format_reports

    if args.suggest:
        if args.use_cases:
            print("give either use cases or --suggest, not both", file=sys.stderr)
            return 2
        try:
            base, lookup = (read_table(adapter_for(p), p) for p in args.suggest)
        except (OSError, ValueError) as exc:
            print(f"cannot read the files: {exc}", file=sys.stderr)
            return 2
        found = suggest_joins(arrays(base), arrays(lookup))
        if not found:
            print("no shared key or date columns found")
        for s in found:
            print(f"{s['kind']:<5} base.{s['base']} = lookup.{s['lookup']}  ({s['detail']})")
        return 0
    if args.use_cases:
        paths = []
        parents = {alias: m.use_case_id for m in bootstrap.load_manifests() for alias in m.aliases}
        for target in args.use_cases:
            target = parents.get(target, target)  # an alias is checked through its use case
            path = Path(target)
            paths.append(path if path.exists() else bootstrap.usecases_dir() / target / MANIFEST_FILENAME)
        manifests = [load_manifest(p) for p in paths]
    else:
        manifests = [m for m in bootstrap.load_manifests() if m.joins]
    failed = False
    for manifest in manifests:
        if not manifest.joins:
            print(f"{manifest.use_case_id}: no joins")
            continue
        if not manifest.data_source.path:
            print(f"{manifest.use_case_id}: no data_source.path to check", file=sys.stderr)
            failed = True
            continue
        print(f"{manifest.use_case_id}:")
        try:
            base = read_table(manifest.data_source.adapter, Path(manifest.data_source.path))
            applied = apply_joins(manifest, base, raise_on_error=False)
        except (OSError, TrainingError) as exc:
            print(f"  cannot check: {exc}", file=sys.stderr)
            failed = True
            continue
        print("\n".join(format_reports(applied.reports)))
        failed = failed or applied.failed
    return 1 if failed else 0


def _list(_: argparse.Namespace) -> int:
    registry = bootstrap.build_registry()
    for manifest in bootstrap.load_manifests():
        versions = registry.versions(manifest.use_case_id)
        state = f"versions {', '.join(versions)}" if versions else "not trained"
        outputs = f" [{', '.join(manifest.training.outputs)}]" if len(manifest.training.outputs) > 1 else ""
        print(f"{manifest.use_case_id:<24} {state:<20} {manifest.description}{outputs}")
        for alias, output in manifest.aliases.items():
            print(f"  {alias:<22} alias of {manifest.use_case_id} / {output}")
    return 0


def _serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run(bootstrap.build_app(), host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
