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


def _reject_constant(constant: str):
    """json calls this for NaN, Infinity and -Infinity, which JSON does not allow: a missing value is null."""
    raise ValueError(f"{constant} is not allowed, use null for a missing value")


def read_jsonl_stream(stream: Iterable[str | bytes], name: str = "<stream>") -> pd.DataFrame:
    """Read JSON Lines (one JSON object per line) from any text or byte stream: a file, ``sys.stdin`` or a socket.

    The stream is consumed line by line, so a sensor feed does not have to be a file on disk. Nothing is
    corrected silently: a line that is not a JSON object, or that holds NaN or Infinity, is refused with its line
    number, and values keep their JSON type (the text "1" stays text). Blank lines and a leading byte-order mark
    are ignored, lines of bytes are decoded as UTF-8, and nested objects are flattened into ``parent_child``
    columns, because sensors often send grouped readings.
    """
    records = []
    for number, line in enumerate(stream, start=1):
        if isinstance(line, bytes):
            try:
                line = line.decode("utf-8")
            except UnicodeDecodeError:
                raise ValueError(f"{name}: line {number} is not valid UTF-8") from None
        line = line.lstrip("\ufeff").strip()
        if not line:
            continue
        try:
            record = json.loads(line, parse_constant=_reject_constant)
        except ValueError as exc:
            raise ValueError(f"{name}: line {number} is not valid JSON ({getattr(exc, 'msg', exc)})") from None
        if not isinstance(record, dict):
            raise ValueError(f"{name}: line {number} is not a JSON object")
        records.append(record)
    return _to_table(records)


def _to_table(records: list[dict]) -> pd.DataFrame:
    import pandas as pd

    sources: dict[str, tuple[str, ...]] = {} 
    object_paths: set[tuple[str, ...]] = set()

    def find_objects(obj: dict, path: tuple[str, ...] = ()) -> None:
        for key, value in obj.items():
            if isinstance(value, dict):
                object_paths.add(path + (key,))
                find_objects(value, path + (key,))

    def flatten(obj: dict, path: tuple[str, ...] = ()) -> dict:
        flat = {}
        for key, value in obj.items():
            here = path + (key,)
            if isinstance(value, dict):
                flat.update(flatten(value, here))
                continue
            if value is None and here in object_paths:
                continue
            column = "_".join(here)
            if sources.setdefault(column, here) != here or column in flat:
                other = sources[column]
                raise ValueError(
                    f"column {column!r} is made from more than one key: "
                    f"{'.'.join(other)!r} and {'.'.join(here)!r}; rename one of them"
                )
            flat[column] = value
        return flat

    for record in records:
        find_objects(record)
    return pd.DataFrame([flatten(record) for record in records])


def _read_jsonl(path: Path) -> pd.DataFrame:
    with open(path, encoding="utf-8") as handle:
        return read_jsonl_stream(handle, str(path))


def _read_json(path: Path) -> pd.DataFrame:
    with open(path, encoding="utf-8-sig") as handle:
        try:
            content = json.load(handle, parse_constant=_reject_constant)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}: not valid JSON (line {exc.lineno}, column {exc.colno}: {exc.msg})") from None
        except ValueError as exc:  # NaN or Infinity
            raise ValueError(f"{path}: {exc}") from None
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


def read_table(adapter: str, path: Path, *, allow_empty: bool = False) -> pd.DataFrame:
    try:
        reader = SOURCES[adapter]
    except KeyError:
        raise ValueError(f"unknown data-source adapter {adapter!r}") from None
    path = Path(path)
    if path.stat().st_size == 0:
        raise ValueError(f"{path}: the file is empty")
    df = reader(path)
    if df.empty and not allow_empty:
        raise ValueError(f"{path}: the file contains no records")
    return df


def adapter_for(path: Path) -> str:
    try:
        return EXTENSIONS[Path(path).suffix.lower()]
    except KeyError:
        raise ValueError(f"cannot guess the format of {path}; use .csv, .json, .jsonl or .parquet") from None