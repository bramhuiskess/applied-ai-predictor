"""Joining lookup tables onto records: ONE engine for training and serving.

A join (see ``JoinConfig`` in the manifest) connects a lookup table to the base table by key, by a time
window, or both, and creates new columns (``aggregate``). The trainer applies it to a whole table, the
service applies it to the record of a request, with this same code, so a feature is computed identically
when the model is trained and when it is used (no train/serve skew).

Only numpy and the standard library are used: the core must not depend on pandas (the serving image has
none), and lookup tables travel with the model version as small ``.npz`` snapshots. The key check lives in
``join_checks.py``, strict dates and key helpers in ``join_values.py``.
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Collection, Mapping, Sequence

import numpy as np

from predictor.core.join_values import as_dates, did_you_mean, is_numeric, key_str, key_strings
from predictor.core.manifest import JoinAggregate, JoinConfig
from predictor.core.validation import FieldError

OBSERVED, CLIMATOLOGY, NO_DATA, UNKNOWN_KEY = 0, 1, 2, 3
SNAPSHOT_VERSION = 1


@dataclass
class Lookup:
    """A lookup table sorted by (keys, date), with the rows of every key group findable in O(1)."""

    key_columns: list[str]
    time_column: str | None
    keys: list[np.ndarray]
    dates: np.ndarray | None
    values: dict[str, np.ndarray]
    groups: dict[tuple[str, ...], tuple[int, int]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = len(next(iter(self.values.values()))) if self.values else (len(self.dates) if self.dates is not None else 0)
        if not self.keys:
            self.groups = {(): (0, n)} if n else {}
            return
        change = np.zeros(n, dtype=bool)
        if n:
            change[0] = True
        for column in self.keys:
            change[1:] |= column[1:] != column[:-1]
        starts = np.flatnonzero(change)
        ends = np.append(starts[1:], n)
        self.groups = {tuple(str(column[s]) for column in self.keys): (int(s), int(e)) for s, e in zip(starts, ends)}

    @classmethod
    def build(cls, table: Mapping[str, Any], *, key_columns: Sequence[str], time_column: str | None,
              value_columns: Sequence[str]) -> Lookup:
        missing = [c for c in [*key_columns, *([time_column] if time_column else []), *value_columns] if c not in table]
        if missing:
            raise ValueError(f"lookup table has no column(s) {missing} (columns: {list(table)})")
        keys = [key_strings(table[c]) for c in key_columns]
        dates = as_dates(table[time_column], f"lookup column {time_column!r}") if time_column else None
        values: dict[str, np.ndarray] = {}
        for name in dict.fromkeys(value_columns):
            raw = np.asarray(table[name])
            if time_column is not None:
                try:
                    values[name] = raw.astype(np.float64)
                except (ValueError, TypeError) as exc:
                    raise ValueError(f"lookup column {name!r} must be numeric to aggregate over a time window") from exc
            else:
                values[name] = raw if is_numeric(raw) else raw.astype(str)
        order_keys = ([dates.astype(np.int64)] if dates is not None else []) + list(reversed(keys))
        if order_keys:
            order = np.lexsort(tuple(order_keys))
            keys = [k[order] for k in keys]
            dates = dates[order] if dates is not None else None
            values = {k: v[order] for k, v in values.items()}
        return cls(list(key_columns), time_column, keys, dates, values)

    def __len__(self) -> int:
        return len(next(iter(self.values.values()))) if self.values else 0

    def known_values(self, index: int) -> list[str]:
        return sorted({group[index] for group in self.groups})

    def to_npz(self) -> bytes:
        names = list(self.values)
        arrays: dict[str, np.ndarray] = {"meta": np.array(json.dumps(
            {"key_columns": self.key_columns, "time_column": self.time_column, "values": names}))}
        for i, column in enumerate(self.keys):
            arrays[f"key_{i}"] = column
        if self.dates is not None:
            arrays["dates"] = self.dates.astype("datetime64[D]").astype(np.int64)
        for i, name in enumerate(names):
            arrays[f"val_{i}"] = self.values[name]
        buffer = io.BytesIO()
        np.savez_compressed(buffer, **arrays)
        return buffer.getvalue()

    @classmethod
    def from_npz(cls, data: bytes) -> Lookup:
        with np.load(io.BytesIO(data), allow_pickle=False) as z:
            meta = json.loads(str(z["meta"]))
            keys = [z[f"key_{i}"] for i in range(len(meta["key_columns"]))]
            dates = z["dates"].astype("datetime64[D]") if "dates" in z.files else None
            values = {name: z[f"val_{i}"] for i, name in enumerate(meta["values"])}
        return cls(meta["key_columns"], meta["time_column"], keys, dates, values)



def _shift_years(starts: np.ndarray, years: np.ndarray) -> np.ndarray:
    months = starts.astype("datetime64[M]")
    day = (starts - months.astype("datetime64[D]")).astype(np.int64)
    return (months + (years * 12).astype("timedelta64[M]")).astype("datetime64[D]") + day.astype("timedelta64[D]")



def _year(dates: np.ndarray) -> np.ndarray:
    return dates.astype("datetime64[Y]").astype(np.int64) + 1970



def _aggregate(dates: np.ndarray, columns: Mapping[str, np.ndarray], aggregates: Mapping[str, JoinAggregate],
               starts: np.ndarray, ends: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Aggregate the rows of one key group over [start, end) for many windows at once (raw, unscaled).

    Returns the values per new column and the share of the window's days that exist in the lookup.
    """
    lo = np.searchsorted(dates, starts, side="left")
    hi = np.maximum(np.searchsorted(dates, ends, side="left"), lo)
    expected = np.maximum((ends - starts).astype("timedelta64[D]").astype(np.int64), 0)
    # A day only counts when the value is there: rows full of empty cells are a gap, not data.
    prefix: dict[str, np.ndarray] = {}  # cumulative count of non-missing values per column (exact)
    for agg in aggregates.values():
        if agg.column not in prefix:
            prefix[agg.column] = np.concatenate([[0], np.cumsum(~np.isnan(columns[agg.column]))])
    present = np.min([prefix[a.column][hi] - prefix[a.column][lo] for a in aggregates.values()], axis=0)
    coverage = np.where(expected > 0, np.minimum(present / np.maximum(expected, 1), 1.0), 0.0)
    out: dict[str, np.ndarray] = {}
    for name, agg in aggregates.items():
        v = columns[agg.column]
        res = np.full(len(starts), np.nan)
        if agg.op == "count":
            res = (prefix[agg.column][hi] - prefix[agg.column][lo]).astype(np.float64)
        else:
            # Per-window slices (not differences of running totals): a sum has the same rounding as summing the
            # window directly, so a feature does not change in the last digit when it moves into a join.
            pick = {"sum": np.sum, "mean": np.mean, "min": np.min, "max": np.max,
                    "first": lambda s: s[0], "last": lambda s: s[-1]}[agg.op]
            for i in range(len(starts)):
                segment = v[lo[i]:hi[i]]
                segment = segment[~np.isnan(segment)]
                if segment.size:
                    res[i] = pick(segment)
        out[name] = res
    return out, coverage



def _finish(values: np.ndarray, agg: JoinAggregate) -> np.ndarray:
    if agg.scale != 1.0:
        inverse = 1.0 / agg.scale
        # 0.01 is not exact in binary: dividing by 100 gives the digit people expect where x * 0.01 can be off by one
        divisor = round(inverse)
        values = values / divisor if divisor != 0 and abs(inverse - divisor) < 1e-9 else values * agg.scale
    return np.round(values, agg.round) if agg.round is not None else values



@dataclass
class JoinResult:
    columns: dict[str, np.ndarray]
    status: dict[str, np.ndarray]  # join name -> OBSERVED | CLIMATOLOGY | NO_DATA | UNKNOWN_KEY per row
    coverage: dict[str, np.ndarray]  # join name -> share of window days found (time joins; 1.0 for key joins)
    years: dict[str, np.ndarray]  # join name -> number of years averaged when CLIMATOLOGY
    windows: dict[str, tuple[np.ndarray, np.ndarray]]  # join name -> (start, end) per row (time joins)




class JoinEngine:
    def __init__(self, joins: Sequence[JoinConfig], lookups: Mapping[str, Lookup]) -> None:
        self.joins = list(joins)
        self.lookups = dict(lookups)
        for join in self.joins:
            lookup = self.lookups.get(join.name)
            if lookup is None:
                raise ValueError(f"join {join.name!r}: no lookup table loaded")
            for agg in join.aggregate.values():
                if agg.column not in lookup.values:
                    raise ValueError(f"join {join.name!r}: lookup has no column {agg.column!r}")
        self.derived: dict[str, str] = {c: j.name for j in self.joins for c in j.new_columns}
        self.inputs: list[str] = list(dict.fromkeys(c for j in self.joins for c in j.base_inputs))

    # -- batch (training, checks)

    def apply(self, base: Mapping[str, Any], *, fallback: bool = False, only: Collection[str] | None = None) -> JoinResult:
        n = len(next(iter(base.values()))) if base else 0
        base = dict(base)
        result = JoinResult({}, {}, {}, {}, {})
        for join in self.joins:
            if only is not None and join.name not in only:
                continue
            missing = [c for c in join.base_inputs if c not in base]
            if missing:
                raise ValueError(f"join {join.name!r}: the base table has no column(s) {missing}")
            self._apply_one(join, base, n, fallback, result)
            base.update(result.columns)  # a later join may be keyed on a column an earlier join created
        return result

    def _apply_one(self, join: JoinConfig, base: Mapping[str, Any], n: int, fallback: bool, result: JoinResult) -> None:
        lookup = self.lookups[join.name]
        key_arrays = [key_strings(base[c]) for c in join.keys]
        rows_by_group: dict[tuple[str, ...], list[int]] = {}
        for i, key in enumerate(zip(*key_arrays) if key_arrays else [()] * n):
            rows_by_group.setdefault(tuple(key), []).append(i)

        status = np.full(n, UNKNOWN_KEY, dtype=np.int8)
        coverage = np.ones(n) if join.time is None else np.zeros(n)
        years = np.zeros(n, dtype=np.int32)
        columns: dict[str, np.ndarray] = {}
        for name, agg in join.aggregate.items():
            source = lookup.values[agg.column]
            columns[name] = np.full(n, np.nan) if join.time is not None or is_numeric(source) else np.full(n, None, dtype=object)

        if join.time is None:
            for key, rows in rows_by_group.items():
                span = lookup.groups.get(key)
                if span is None:
                    continue
                idx = np.array(rows)
                status[idx] = OBSERVED
                for name, agg in join.aggregate.items():
                    value = lookup.values[agg.column][span[0]]
                    columns[name][idx] = _finish(np.asarray(value, dtype=np.float64), agg) if is_numeric(lookup.values[agg.column]) else value
        else:
            starts = self._bound(join, join.window.start, base)
            ends = self._bound(join, join.window.end, base)
            result.windows[join.name] = (starts, ends)
            for key, rows in rows_by_group.items():
                span = lookup.groups.get(key)
                if span is None:
                    continue
                idx = np.array(rows)
                dates = lookup.dates[span[0]:span[1]]
                cols = {agg.column: lookup.values[agg.column][span[0]:span[1]] for agg in join.aggregate.values()}
                values, cover = _aggregate(dates, cols, join.aggregate, starts[idx], ends[idx])
                coverage[idx] = cover
                ok = (cover >= join.min_coverage) & ~np.any([np.isnan(v) for v in values.values()], axis=0)
                status[idx] = np.where(ok, OBSERVED, NO_DATA)
                for name, agg in join.aggregate.items():
                    columns[name][idx[ok]] = _finish(values[name][ok], agg)
                if fallback and join.fallback == "climatology" and (~ok).any():
                    bad = idx[~ok]
                    cvalues, ny = self._climatology(join, dates, cols, starts[bad], ends[bad])
                    got = ny > 0
                    status[bad[got]] = CLIMATOLOGY
                    years[bad[got]] = ny[got]
                    for name, agg in join.aggregate.items():
                        columns[name][bad[got]] = _finish(cvalues[name][got], agg)
        result.columns.update(columns)
        result.status[join.name] = status
        result.coverage[join.name] = coverage
        result.years[join.name] = years

    @staticmethod
    def _bound(join: JoinConfig, bound: Any, base: Mapping[str, Any]) -> np.ndarray:
        column = bound.column or join.time.base
        return as_dates(base[column], f"column {column!r}") + np.timedelta64(bound.offset_days, "D")

    @staticmethod
    def _climatology(join: JoinConfig, dates: np.ndarray, cols: Mapping[str, np.ndarray], starts: np.ndarray,
                     ends: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray]:
        """Average of the same calendar window in every year of the lookup where it is fully (>= min_coverage) there."""
        sums = {name: np.zeros(len(starts)) for name in join.aggregate}
        count = np.zeros(len(starts), dtype=np.int32)
        if len(dates) == 0:
            return {name: np.full(len(starts), np.nan) for name in join.aggregate}, count
        own_year = _year(starts)
        for year in range(int(_year(dates[:1])[0]), int(_year(dates[-1:])[0]) + 1):
            shifted = _shift_years(starts, year - own_year)
            values, cover = _aggregate(dates, cols, join.aggregate, shifted, shifted + (ends - starts))
            ok = (cover >= join.min_coverage) & ~np.any([np.isnan(v) for v in values.values()], axis=0)
            for name in sums:
                sums[name][ok] += values[name][ok]
            count += ok
        return {name: np.where(count > 0, s / np.maximum(count, 1), np.nan) for name, s in sums.items()}, count

    # -- single record (serving)

    def enrich_record(self, features: Mapping[str, Any], feature_names: Collection[str]) -> tuple[dict[str, Any], list[FieldError], list[str]]:
        """Fill the joined columns of one request record, returning (features, errors, warnings).

        Columns the caller sent themselves (not null) are kept. Join inputs that are not model features
        (e.g. ``start_date``) are removed afterwards, so the schema check still sees only features. Joins run in
        manifest order, so a join may be keyed on a column an earlier join created.
        """
        out = dict(features)
        errors: list[FieldError] = []
        warnings: list[str] = []
        read_by_a_join = {c for j in self.joins for c in j.base_inputs}
        for join in self.joins:
            # Only what the model (or a later join) uses, and only what the caller did not send itself.
            wanted = [c for c in join.new_columns if (c in feature_names or c in read_by_a_join) and out.get(c) is None]
            if not wanted:
                continue
            missing = [c for c in join.base_inputs if out.get(c) is None]
            if missing:
                errors.append(FieldError(missing[0], f"{missing[0]} is required to look up {wanted} from '{join.name}' "
                                                     f"(or send {'it' if len(wanted) == 1 else 'them'} yourself)"))
                continue
            try:
                res = self.apply({c: [out[c]] for c in join.base_inputs}, fallback=True, only={join.name})
            except (ValueError, TypeError, OverflowError) as exc:
                errors.append(FieldError(join.base_inputs[0] if len(join.base_inputs) == 1 else "features", str(exc)))
                continue
            code = int(res.status[join.name][0])
            if code == UNKNOWN_KEY:
                errors.append(self._unknown_key_error(join, out))
            elif code == NO_DATA:
                start, end = res.windows[join.name]
                errors.append(FieldError(join.time.base or join.base_inputs[0],
                                         f"no data in '{join.name}' for {start[0]} to {end[0]} "
                                         f"(coverage {res.coverage[join.name][0]:.0%}); send {wanted} yourself"))
            else:
                for name in wanted:
                    value = res.columns[name][0]
                    out[name] = value.item() if isinstance(value, np.generic) else value
                if code == CLIMATOLOGY:
                    start, end = res.windows[join.name]
                    warnings.append(f"{', '.join(wanted)} are not observed for {start[0]} to {end[0]} (no data in '{join.name}'); "
                                    f"filled with the average of the same period in {int(res.years[join.name][0])} other year(s)")
        for name in self.inputs:
            if name not in feature_names:
                out.pop(name, None)
        return out, errors, warnings

    def _unknown_key_error(self, join: JoinConfig, features: Mapping[str, Any]) -> FieldError:
        lookup = self.lookups[join.name]
        if len(lookup) == 0:
            return FieldError(join.time.base if join.time and join.time.base else join.base_inputs[0],
                              f"the lookup table of join '{join.name}' has no rows")
        for i, base_column in enumerate(join.keys):
            value = key_str(features[base_column])
            known = lookup.known_values(i)
            if value not in known:
                return FieldError(base_column, f"{value!r} is not known in '{join.name}'" + did_you_mean(value, known))
        return FieldError(next(iter(join.keys), join.base_inputs[0]),
                          f"this combination of {list(join.keys) or join.base_inputs} is not known in '{join.name}'")

    # -- snapshot (stored with the model version)

    def snapshot(self) -> dict[str, str | bytes]:
        files: dict[str, str | bytes] = {"joins.json": json.dumps(
            {"snapshot_version": SNAPSHOT_VERSION, "joins": [j.model_dump(mode="json") for j in self.joins]}, indent=2)}
        for join in self.joins:
            files[f"lookups/{join.name}.npz"] = self.lookups[join.name].to_npz()
        return files

    @classmethod
    def from_snapshot(cls, read: Callable[[str], bytes]) -> JoinEngine:
        spec = json.loads(read("joins.json").decode("utf-8"))
        joins = [JoinConfig.model_validate(j) for j in spec["joins"]]
        return cls(joins, {j.name: Lookup.from_npz(read(f"lookups/{j.name}.npz")) for j in joins})
