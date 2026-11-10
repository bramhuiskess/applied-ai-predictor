"""Data-source adapters: read a dataset file into a pandas DataFrame.

pandas is only needed for training, so it is imported lazily; a serving-only
installation can still list the available adapter names.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    import pandas as pd


def _read_csv(path: Path) -> pd.DataFrame:
    import pandas as pd

    return pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")  # sniffs "," or ";", strips a BOM


def _read_jsonl(path: Path) -> pd.DataFrame:
    import pandas as pd

    return pd.read_json(path, lines=True, convert_dates=False)

def _read_json(path: Path) -> pd.DataFrame:
    import json

    import pandas as pd

    with open(path, encoding="utf-8-sig") as handle:
        content = json.load(handle)
    if isinstance(content, dict) and isinstance(content.get("records"), list):
        content = content["records"]
    if not isinstance(content, list) or not all(isinstance(row, dict) for row in content):
        raise ValueError(f"{path}: expected a JSON list of records or an object with a 'records' list")
    return pd.DataFrame(content)


def _read_parquet(path: Path) -> pd.DataFrame:
    import pandas as pd

    return pd.read_parquet(path)


SOURCES: dict[str, Callable[[Path], "pd.DataFrame"]] = {
    "csv-file": _read_csv,
    "jsonl-file": _read_jsonl,
    "json-file": _read_json,
    "parquet-file": _read_parquet,
}

EXTENSIONS = {".csv": "csv-file", ".json": "json-file", ".jsonl": "jsonl-file", ".ndjson": "jsonl-file", ".parquet": "parquet-file"}


def read_table(adapter: str, path: Path) -> pd.DataFrame:
    try:
        reader = SOURCES[adapter]
    except KeyError:
        raise ValueError(f"unknown data-source adapter {adapter!r}") from None
    path = Path(path)
    if path.stat().st_size == 0:
        raise ValueError(f"{path}: the file is empty")
    df = reader(path)
    if df.empty:
        raise ValueError(f"{path}: the file contains no records")
    return df


def adapter_for(path: Path) -> str:
    try:
        return EXTENSIONS[Path(path).suffix.lower()]
    except KeyError:
        raise ValueError(f"cannot guess the format of {path}; use .csv, .json, .jsonl or .parquet") from None