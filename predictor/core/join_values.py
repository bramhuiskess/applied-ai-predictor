"""Small helpers shared by the join engine and its checks: strict dates, key text, "did you mean"."""

from __future__ import annotations

import datetime
import difflib
import re
import unicodedata
from typing import Any, Sequence

import numpy as np

_DATE = re.compile(r"\d{4}-\d{2}-\d{2}(?:[T ].*)?")
MIN_YEAR, MAX_YEAR = 1900, 2200  # anything else is a typo or an integer that is not a date (20240304, 5, 9999-12-31)


def as_dates(values: Any, what: str = "date") -> np.ndarray:
    """Convert to ``datetime64[D]``; raises ValueError with a readable message for anything that is not a date.

    Accepts dates, datetimes and text that starts with YYYY-MM-DD. Numbers, bare years, lists and years
    outside 1900-2200 are rejected instead of being turned into a wrong date.
    """
    arr = np.asarray(values, dtype=object) if not isinstance(values, np.ndarray) else values
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.ndim != 1:
        raise ValueError(f"{what}: expected one date per row, got a nested value")
    if arr.dtype.kind == "M":
        out = arr.astype("datetime64[D]")
    else:
        items = arr.tolist()
        bad = next((v for v in items if not isinstance(v, (str, datetime.date, np.datetime64)) or
                    (isinstance(v, str) and not _DATE.fullmatch(v.strip()))), None)
        if bad is not None or not items:
            if not items:
                return np.array([], dtype="datetime64[D]")
            raise ValueError(f"{what}: {bad!r} is not a date (use YYYY-MM-DD)")
        try:
            out = np.array([np.datetime64(v.strip()[:10] if isinstance(v, str) else v, "D") for v in items])
        except ValueError as exc:
            bad = next(v for v in items if not _is_date(v))
            raise ValueError(f"{what}: {bad!r} is not a date (use YYYY-MM-DD)") from exc
    if np.isnat(out).any():
        raise ValueError(f"{what}: missing or invalid date (use YYYY-MM-DD)")
    years = out.astype("datetime64[Y]").astype(np.int64) + 1970
    if len(out) and (years.min() < MIN_YEAR or years.max() > MAX_YEAR):
        odd = out[(years < MIN_YEAR) | (years > MAX_YEAR)][0]
        raise ValueError(f"{what}: {odd} is outside {MIN_YEAR}-{MAX_YEAR}; check the date")
    return out



def _is_date(value: Any) -> bool:
    try:
        return not np.isnat(np.datetime64(str(value).strip()[:10], "D"))
    except (ValueError, TypeError):
        return False



def key_str(value: Any) -> str:
    """One text form for keys, so 344, 344.0 and "344" match."""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)



def key_strings(values: Any) -> np.ndarray:
    if isinstance(values, np.ndarray) and values.dtype.kind == "f" and values.dtype.itemsize < 8:
        values = np.array([float(x) for x in values.astype(str)])  # float32 0.1 is the key 0.1, not 0.10000000149
    items = values.tolist() if hasattr(values, "tolist") else list(values)  # keep each value's own type
    return np.array([key_str(v) for v in items], dtype=str)



def _fold(text: str) -> str:
    """Case, accent and whitespace insensitive form, to hint at near-identical keys."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c)).strip().casefold()



def is_numeric(values: np.ndarray) -> bool:
    return values.dtype.kind in "iufb"



def did_you_mean(value: str, known: Sequence[str]) -> str:
    folded = {_fold(k): k for k in known}
    if _fold(value) in folded:
        return f"; it differs only in case, accents or spaces from {folded[_fold(value)]!r}"
    close = difflib.get_close_matches(value, known, n=3, cutoff=0.6) or [
        folded[f] for f in difflib.get_close_matches(_fold(value), list(folded), n=3, cutoff=0.6)]
    return f"; did you mean {' or '.join(repr(c) for c in close)}?" if close else ""
