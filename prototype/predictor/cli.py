"""Command line: ``predictor init | train | list | serve``."""

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

    sub.add_parser("list", help="list use cases and their model versions")

    p_serve = sub.add_parser("serve", help="start the HTTP API")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)

    args = parser.parse_args(argv)
    try:
        return {"init": _init, "train": _train, "list": _list, "serve": _serve}[args.command](args)
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

    targets = args.use_cases or [m.use_case_id for m in bootstrap.load_manifests()]
    if args.data and len(targets) != 1:
        print("--data can only be used with a single use case", file=sys.stderr)
        return 2
    registry = bootstrap.build_registry()
    for target in targets:
        path = Path(target)
        if not path.exists():
            path = bootstrap.usecases_dir() / target / MANIFEST_FILENAME
        result = train_use_case(path, registry, args.data)
        metrics = ", ".join(f"{k}={v:.4f}" for k, v in result["metrics"].items())
        print(f"{result['use_case_id']}: v{result['model_version']} {result['algorithm']} ({metrics}) "
              f"parity max diff {result['parity']['max_abs_diff']:.1e}")
        for note in result["notes"]:
            if note.startswith("WARNING"):
                print(f"  {note}")
        print(f"  report: {result['report']}")
    return 0


def _list(_: argparse.Namespace) -> int:
    registry = bootstrap.build_registry()
    for manifest in bootstrap.load_manifests():
        versions = registry.versions(manifest.use_case_id)
        state = f"versions {', '.join(versions)}" if versions else "not trained"
        print(f"{manifest.use_case_id:<24} {state:<20} {manifest.description}")
    return 0


def _serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run(bootstrap.build_app(), host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
