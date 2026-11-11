"""Data-source adapters: read a dataset file into a pandas DataFrame.

pandas is only needed for training, so it is imported lazily; a serving-only
installation can still list the available adapter names.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterable

if TYPE_CHECKING:
    import pandas as pd


def _read_csv(path: Path) -> pd.DataFrame:
    import pandas as pd

    return pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")  # sniffs "," or ";", strips a BOM


def read_jsonl_stream(stream: Iterable[str], name: str = "<stream>") -> pd.DataFrame:
    """Read JSON Lines (one JSON object per line) from any text stream: a file, ``sys.stdin`` or a socket file.

    The stream is consumed line by line, so a sensor feed does not have to be a file on disk. Nothing is
    corrected silently: a line that is not a JSON object is refused with its line number, and values keep
    their JSON type (the text "1" stays text). Blank lines and a leading byte-order mark are ignored, and nested
    objects are flattened into `parent_child` columns, because sensors often send grouped readings.
    """
    records = []
    for number, line in enumerate(stream, start=1):
        line = line.lstrip("\ufeff").strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except ValueError as exc:
            raise ValueError(f"{name}: line {number} is not valid JSON ({exc.msg})") from None
        if not isinstance(record, dict):
            raise ValueError(f"{name}: line {number} is not a JSON object")
        records.append(record)
    return _to_table(records)


def _to_table(records: list[dict]) -> pd.DataFrame:
    import pandas as pd

    if any(isinstance(value, dict) for record in records for value in record.values()):
        return pd.json_normalize(records, sep="_") 
    return pd.DataFrame(records)


def _read_jsonl(path: Path) -> pd.DataFrame:
    with open(path, encoding="utf-8") as handle:
        return read_jsonl_stream(handle, str(path))


def _read_json(path: Path) -> pd.DataFrame:
    with open(path, encoding="utf-8-sig") as handle:
        content = json.load(handle)
    if isinstance(content, dict) and isinstance(content.get("records"), list):
        content = content["records"]
    if not isinstance(content, list) or not all(isinstance(row, dict) for row in content):
        raise ValueError(f"{path}: expected a JSON list of records or an object with a 'records' list")
    return _to_table(content)


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