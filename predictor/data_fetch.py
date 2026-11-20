"""Fetch the demo datasets: run the existing generator scripts and describe the result in ``dataset.json``.

One table (``DEMO_DATASETS``) names the generators; adding a dataset later means adding a row. The scripts
themselves are unchanged and still work on their own. Needs the source checkout (``scripts/``) and the
``train`` extra (pandas); ``greenhouse-challenge`` also needs the ``datasets`` extra (py7zr).
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from predictor.core.dataset import DatasetManifest, describe_file, write_dataset_manifest



def scripts_dir() -> Path:
    """The generator scripts: next to the package in a source checkout, else in ./scripts (e.g. a Docker build)."""
    for candidate in (Path(__file__).resolve().parents[1] / "scripts", Path.cwd() / "scripts"):
        if (candidate / "make_datasets.py").is_file():
            return candidate
    return Path(__file__).resolve().parents[1] / "scripts"


Log = Callable[[str], None]


class FetchError(RuntimeError):
    pass


@dataclass(frozen=True)
class DemoDataset:
    id: str
    script: str
    files: tuple[str, ...]  # relative to the project home
    description: str
    source: str


DEMO_DATASETS: tuple[DemoDataset, ...] = (
    DemoDataset("basics", "make_datasets.py",
                ("data/greenhouse.csv", "data/absence.csv", "data/absence_with_diagnosis.csv", "data/ai4i2020.csv"),
                "Simulated greenhouse climate and absence, plus the real AI4I 2020 machine data",
                "simulated (seeded); AI4I 2020 from archive.ics.uci.edu, CC BY 4.0"),
    DemoDataset("nursery", "make_nursery.py",
                ("data/nursery.csv", "data/nursery_base.csv", "data/compartments.csv", "data/weather_rotterdam.csv"),
                "Simulated nursery batches on real KNMI weather", "simulator + KNMI daily weather"),
    DemoDataset("arable", "make_arable.py",
                ("data/arable.csv", "data/arable_base.csv", "data/weather_provinces.csv"),
                "Arable yields per province with season weather", "CBS table 85636NED + KNMI weather"),
    DemoDataset("tomato-season", "make_tomato_season.py", ("data/tomato_season.csv",),
                "Simulated tomato seasons (crop model on KNMI weather)", "crop model + KNMI daily weather"),
    DemoDataset("greenhouse-challenge", "make_greenhouse_challenge.py",
                ("data/greenhouse_days.csv", "data/greenhouse_harvests.csv"),
                "Real greenhouse growing data, daily and harvests", "WUR Autonomous Greenhouse Challenge, data.4tu.nl"),
)


def known_ids() -> list[str]:
    return [d.id for d in DEMO_DATASETS]


def state(home: Path) -> list[dict]:
    """Per demo dataset: is it on disk and described?"""
    home = Path(home)
    return [{"id": d.id, "description": d.description, "files": list(d.files),
             "present": all((home / f).is_file() for f in d.files),
             "described": (home / "datasets" / d.id / "dataset.json").is_file()} for d in DEMO_DATASETS]


def fetch(home: Path, ids: list[str] | None = None, *, force: bool = False, log: Log = print) -> list[str]:
    """Generate the datasets (all when ``ids`` is empty) and write their dataset.json. Returns the fetched ids."""
    home = Path(home)
    wanted = ids or known_ids()
    unknown = [i for i in wanted if i not in known_ids()]
    if unknown:
        raise FetchError(f"unknown dataset(s): {', '.join(unknown)} (known: {', '.join(known_ids())})")
    done = []
    for dataset in DEMO_DATASETS:
        if dataset.id not in wanted:
            continue
        if not force and all((home / f).is_file() for f in dataset.files):
            log(f"{dataset.id}: already there")
        else:
            _run(dataset, home, log)
        files = [describe_file(home, f) for f in dataset.files]
        write_dataset_manifest(home / "datasets", DatasetManifest(
            dataset_version="1.0", id=dataset.id, description=dataset.description, files=files,
            source=dataset.source))
        done.append(dataset.id)
    return done


def _run(dataset: DemoDataset, home: Path, log: Log) -> None:
    script = scripts_dir() / dataset.script
    if not script.is_file():
        raise FetchError(f"{script} not found: `data fetch` needs the source checkout")
    log(f"{dataset.id}: running {dataset.script} (may download data)")
    (Path(home) / "data").mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen([sys.executable, str(script)], cwd=home, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    tail: list[str] = []
    for line in proc.stdout or []:
        line = line.rstrip()
        if line:
            log(f"  {line}")
            tail = (tail + [line])[-5:]
    if proc.wait() != 0:
        raise FetchError(f"{dataset.id}: {dataset.script} failed ({'; '.join(tail)})")
    missing = [f for f in dataset.files if not (Path(home) / f).is_file()]
    if missing:
        raise FetchError(f"{dataset.id}: expected file(s) not written: {', '.join(missing)}")
