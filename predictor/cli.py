"""Command line: ``predictor setup | doctor | data | package | run | init | train | check-joins | list | serve``."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from predictor import bootstrap, serving, workflow
from predictor.core.manifest import MANIFEST_FILENAME, ManifestError
from predictor.trainer.errors import TrainingError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="predictor", description="Modular prediction backend")
    parser.add_argument("--home", type=Path, default=None,
                        help="project or package folder to work in (default: PREDICTOR_HOME, else the current folder)")
    sub = parser.add_subparsers(dest="command", required=True)

    p_setup = sub.add_parser("setup", help="guided first-time setup: in the browser (default) or in the terminal")
    p_setup.add_argument("--cli", "--no-browser", dest="cli", action="store_true",
                         help="do the setup in the terminal instead of the browser")
    p_setup.add_argument("--with-data", action="store_true", help="install the demo data (default: serve-only)")
    p_setup.add_argument("--yes", action="store_true", help="never ask; answer the questions with their defaults")
    p_setup.add_argument("--reconfigure", action="store_true",
                         help="run the setup again even if the project is already set up (default: just start it)")
    p_setup.add_argument("--host", default="127.0.0.1")
    p_setup.add_argument("--port", type=int, default=8000)

    p_doctor = sub.add_parser("doctor", help="check that the project is ready to train and to serve (read-only)")
    p_doctor.add_argument("--json", action="store_true", help="machine-readable output")

    p_data = sub.add_parser("data", help="the demo datasets")
    data_sub = p_data.add_subparsers(dest="data_command", required=True)
    p_fetch = data_sub.add_parser("fetch", help="generate/download the demo datasets and write their dataset.json")
    p_fetch.add_argument("datasets", nargs="*", help="dataset ids (default: all)")
    p_fetch.add_argument("--force", action="store_true", help="run the generators again even if the files exist")
    data_sub.add_parser("list", help="the demo datasets and whether they are on disk")

    p_package = sub.add_parser("package", help="build or verify the application package")
    package_sub = p_package.add_subparsers(dest="package_command", required=True)
    p_build = package_sub.add_parser("build", help="bundle use cases and models (and with --with-data the data)")
    p_build.add_argument("--out", required=True, type=Path, help="a folder, or a .tar.gz archive")
    p_build.add_argument("--with-data", action="store_true", help="also include the datasets (default: serve-only)")
    p_build.add_argument("--name")
    p_verify = package_sub.add_parser("verify", help="check a package folder or archive offline")
    p_verify.add_argument("path", type=Path, nargs="?", default=Path("."))
    p_verify.add_argument("--json", action="store_true")

    p_run = sub.add_parser("run", help="check the project, then start the HTTP API and dashboard")
    p_run.add_argument("--host", default="127.0.0.1")
    p_run.add_argument("--port", type=int, default=8000)
    p_run.add_argument("--open", action="store_true", help="open the dashboard in the browser once it is up")

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
    home = args.home or (Path(os.environ["PREDICTOR_HOME"]) if os.environ.get("PREDICTOR_HOME") else None)
    if home is not None:
        if not home.is_dir():
            print(f"--home {home}: not a folder", file=sys.stderr)
            return 2
        os.chdir(home)  # manifests use paths relative to the project folder
    try:
        return {"setup": _setup, "doctor": _doctor, "data": _data, "package": _package, "run": _run,
                "init": _init, "train": _train, "check-joins": _check_joins, "list": _list,
                "serve": _serve}[args.command](args)
    except ManifestError as exc:
        print(f"manifest error: {exc}", file=sys.stderr)
        return 2
    except TrainingError as exc:
        print(f"training error: {exc}", file=sys.stderr)
        return 2


_MARK = {"ok": "ok  ", "warn": "warn", "error": "ERR "}


def _print_findings(findings) -> None:
    for f in findings:
        print(f"[{_MARK[f.level]}] {f.check}: {f.message}")
        if f.hint and f.level != "ok":
            print(f"       -> {f.hint}")


def _doctor(args: argparse.Namespace) -> int:
    from predictor.core.doctor import has_errors

    findings = workflow.doctor()
    if args.json:
        print(json.dumps([f.as_dict() for f in findings], indent=2))
    else:
        _print_findings(findings)
    return 1 if has_errors(findings) else 0


def _data(args: argparse.Namespace) -> int:
    from predictor import data_fetch

    if args.data_command == "list":
        for d in data_fetch.state(bootstrap.home()):
            print(f"{d['id']:<22} {'on disk' if d['present'] else 'not fetched':<12} {d['description']}")
        return 0
    from predictor.core.dataset import DatasetError

    try:
        done = data_fetch.fetch(bootstrap.home(), args.datasets, force=args.force)
    except (data_fetch.FetchError, DatasetError, ImportError) as exc:
        print(f"data fetch failed: {exc}", file=sys.stderr)
        return 2
    print(f"described: {', '.join(done)} (datasets/<id>/dataset.json)")
    return 0


def _package(args: argparse.Namespace) -> int:
    from predictor.core import package

    try:
        if args.package_command == "build":
            manifest = package.build_package(bootstrap.home(), bootstrap.usecases_dir(), bootstrap.models_dir(),
                                             bootstrap.datasets_dir(), args.out, with_data=args.with_data,
                                             name=args.name)
            kind = "with data" if manifest.with_data else "serve-only"
            print(f"wrote {args.out}: {len(manifest.files)} files ({kind})")
            return 0
        report = package.verify_package(args.path)
    except package.PackageError as exc:
        print(f"package error: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps({"ok": report.ok, "errors": report.errors, "warnings": report.warnings}, indent=2))
    else:
        for line in report.errors:
            print(f"[ERR ] {line}")
        for line in report.warnings:
            print(f"[warn] {line}")
        n = len(report.manifest.files) if report.manifest else 0
        print(f"package {'ok' if report.ok else 'FAILED'} ({n} files checked)")
    return 0 if report.ok else 1


def _run(args: argparse.Namespace) -> int:
    from predictor.core.doctor import has_errors

    findings = workflow.doctor()
    _print_findings([f for f in findings if f.level != "ok"])
    if has_errors(findings):
        print("not starting: fix the errors above (see `predictor doctor`)", file=sys.stderr)
        return 1
    return serving.start(args.host, args.port, open_browser=args.open)


def _ask(question: str, default: bool) -> bool:
    if not sys.stdin.isatty():
        return default
    answer = input(f"{question} [{'Y/n' if default else 'y/N'}] ").strip().lower()
    return default if not answer else answer.startswith("y")


def _setup(args: argparse.Namespace) -> int:
    if args.cli:
        return _setup_terminal(args)
    if not args.reconfigure and workflow.is_set_up():
        print("Already set up, starting. Use `predictor setup --reconfigure` to go through the setup again.")
        return serving.start(args.host, args.port, open_browser=True)
    return _setup_browser(args)


def _setup_terminal(args: argparse.Namespace) -> int:
    from predictor import data_fetch
    from predictor.core.doctor import has_errors, installed_train_extra

    findings = workflow.doctor()
    _print_findings(findings)
    if has_errors(findings):
        print("fix the errors above first", file=sys.stderr)
        return 1
    install = args.with_data or (not args.yes and _ask("Install the demo data (needed to train the demo use cases)?", False))
    if install:
        if not installed_train_extra():
            print("the training libraries are not installed: pip install -e '.[train,datasets]'", file=sys.stderr)
            return 2
        try:
            data_fetch.fetch(bootstrap.home())
        except (data_fetch.FetchError, ImportError) as exc:
            print(f"data fetch failed: {exc}", file=sys.stderr)
            return 2
    ids = workflow.trainable()
    if ids and installed_train_extra() and (args.with_data or args.yes or _ask(f"Train {len(ids)} use case(s) now?", True)):
        workflow.train_ids(ids)
    print("\nready. Start it with: predictor run --open")
    return 1 if has_errors(workflow.doctor()) else 0


def _setup_browser(args: argparse.Namespace) -> int:
    import threading
    import webbrowser

    import uvicorn

    from predictor.wizard import create_wizard_app

    finished = threading.Event()

    def finish() -> None:
        finished.set()
        server.should_exit = True  # stop the wizard; the normal server takes over below

    server = uvicorn.Server(uvicorn.Config(create_wizard_app(finish), host=args.host, port=args.port,
                                           log_level="warning"))
    url = f"http://{args.host}:{args.port}/setup"
    print(f"setup wizard on {url}  (Ctrl-C to stop; `predictor setup --cli` works without a browser)")
    threading.Timer(1.0, webbrowser.open, args=(url,)).start()
    server.run()
    if not finished.is_set():
        return 0
    return serving.start(args.host, args.port)  # the wizard page opens the dashboard itself


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
        print("\n".join(workflow.format_result(result)))
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
