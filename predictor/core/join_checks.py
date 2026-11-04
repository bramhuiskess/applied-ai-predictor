"""The automatic key check for joins, and suggestions for new joins (numpy only, used by training and the CLI)."""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from predictor.core.join_values import as_dates, did_you_mean, key_strings
from predictor.core.joins import OBSERVED, UNKNOWN_KEY, JoinEngine, Lookup
from predictor.core.manifest import JoinConfig


@dataclass
class JoinReport:
    join: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    info: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)
    lookup: Lookup | None = None

    @property
    def ok(self) -> bool:
        return not self.errors



def check_join(join: JoinConfig, base: Mapping[str, Any], table: Mapping[str, Any], rows: int | None = None) -> JoinReport:
    """Check that a join will work, before and while training: keys, dates, duplicates, coverage.

    ``base`` needs at least the columns the join reads; pass ``rows`` when it holds only those.
    """
    report = JoinReport(join.name)
    n = rows if rows is not None else (len(next(iter(base.values()))) if base else 0)
    if n == 0:
        report.errors.append("the base table has no rows")
        return report
    clash = [c for c in join.new_columns if c in base]
    if clash:
        report.errors.append(f"new column(s) {clash} already exist in the base table; the join would silently replace "
                             "them (rename the new column, or drop the old one from the base table)")
    lookup_columns = [*join.keys.values(), *([join.time.lookup] if join.time else []), *(a.column for a in join.aggregate.values())]
    missing_base = [c for c in join.base_inputs if c not in base]
    missing_lookup = sorted({c for c in lookup_columns if c not in table})
    if missing_base:
        report.errors.append(f"base table has no column(s) {missing_base} (columns: {list(base)})")
    if missing_lookup:
        report.errors.append(f"lookup table has no column(s) {missing_lookup} (columns: {list(table)})")
    if not any(len(np.asarray(table[c])) for c in lookup_columns):
        report.errors.append("the lookup table has no rows")
    if report.errors:
        return report

    # key types: only informative, 344 and "344" are matched by their text form
    for base_column, lookup_column in join.keys.items():
        kinds = (np.asarray(base[base_column]).dtype.kind in "iuf", np.asarray(table[lookup_column]).dtype.kind in "iuf")
        if kinds[0] != kinds[1]:
            report.info.append(f"key {base_column!r} is {'numeric' if kinds[0] else 'text'} in the base table but "
                               f"{'numeric' if kinds[1] else 'text'} in the lookup; compared as text")

    try:
        lookup = Lookup.build(table, key_columns=list(join.keys.values()), time_column=join.time.lookup if join.time else None,
                              value_columns=[a.column for a in join.aggregate.values()])
        base_dates = {}
        if join.time:
            for column in [c for c in join.base_inputs if c not in join.keys]:
                base_dates[column] = as_dates(base[column], f"base column {column!r}")
    except ValueError as exc:
        report.errors.append(str(exc))
        return report
    report.lookup = lookup

    _check_duplicates(join, lookup, report)
    _check_keys(join, base, lookup, n, report)
    if report.errors:
        return report

    engine = JoinEngine([join], {join.name: lookup})
    result = engine.apply(base, fallback=False)
    status, coverage = result.status[join.name], result.coverage[join.name]
    observed = float(np.mean(status == OBSERVED))
    report.stats.update(rows=n, matched=observed, lookup_rows=len(lookup))
    if join.time:
        starts, ends = result.windows[join.name]
        report.info.append(f"base windows {starts.min()} .. {ends.max()} vs lookup {lookup.dates.min()} .. {lookup.dates.max()}")
        keyed = status != UNKNOWN_KEY
        partial = keyed & (coverage > 0) & (coverage < join.min_coverage)
        empty = keyed & (coverage == 0)
        if partial.any():
            report.warnings.append(f"{int(partial.sum())} row(s) have only part of their window in the lookup "
                                   f"(less than {join.min_coverage:.0%} of the days)")
        if empty.any():
            report.warnings.append(f"{int(empty.sum())} row(s) have no data at all in their window "
                                   f"(first: {starts[empty][0]} .. {ends[empty][0]})")
        gappy = keyed & (coverage >= join.min_coverage) & (coverage < 1)
        if gappy.any():
            report.warnings.append(f"{int(gappy.sum())} row(s) miss some days inside their window (still used)")
    if observed < join.min_match:
        report.errors.append(f"only {observed:.1%} of the {n} rows found a match (at least {join.min_match:.0%} is required)")
    elif observed < 1:
        report.warnings.append(f"{(1 - observed) * 100:.1f}% of the rows found no match and will be dropped from training")
    for name in join.new_columns:
        column = result.columns[name][status == OBSERVED]
        if len(column) > 1 and len(set(column.tolist())) == 1:
            report.warnings.append(f"new column {name!r} is constant ({column[0]!r}): it carries no information")
    return report



def _check_duplicates(join: JoinConfig, lookup: Lookup, report: JoinReport) -> None:
    if join.time is None:
        dup = [(key, e - s) for key, (s, e) in lookup.groups.items() if e - s > 1]
        if dup:
            report.errors.append(f"the lookup has {len(dup)} key(s) with more than one row (that would multiply rows), "
                                 f"e.g. {dup[0][0]} x{dup[0][1]}; a key join needs one row per key")
        return
    same = lookup.dates[1:] == lookup.dates[:-1]
    for column in lookup.keys:
        same &= column[1:] == column[:-1]
    if same.any():
        i = int(np.flatnonzero(same)[0]) + 1
        where = tuple(str(c[i]) for c in lookup.keys)
        report.errors.append(f"the lookup has {int(same.sum())} duplicate (key, date) row(s), e.g. {where} on {lookup.dates[i]}")



def _check_keys(join: JoinConfig, base: Mapping[str, Any], lookup: Lookup, n: int, report: JoinReport) -> None:
    if not join.keys:
        return
    base_keys = [tuple(str(v) for v in key) for key in zip(*[key_strings(base[c]) for c in join.keys])]
    known = set(lookup.groups)
    missing = [k for k in base_keys if k not in known]
    share = 1 - len(missing) / n
    report.stats["key_match"] = share
    if not missing:
        return
    unique = sorted(set(missing))
    examples = []
    for key in unique[:5]:
        hint = ""
        for i, base_column in enumerate(join.keys):
            values = lookup.known_values(i)
            if key[i] not in values:
                hint = did_you_mean(key[i], values)
                break
        examples.append(f"{key[0] if len(key) == 1 else key!r} ({missing.count(key)} rows){hint}")
    message = f"{len(unique)} key value(s) are not in the lookup ({share:.1%} of the rows match): " + "; ".join(examples)
    (report.errors if share < join.min_match else report.warnings).append(message)



def suggest_joins(base: Mapping[str, Any], other: Mapping[str, Any], limit: int = 8) -> list[dict[str, Any]]:
    """Propose how two tables could be connected: shared keys (by value overlap) and date columns."""
    suggestions: list[dict[str, Any]] = []
    base_dates = {c: v for c, v in base.items() if _looks_like_dates(v)}
    other_dates = {c: v for c, v in other.items() if _looks_like_dates(v)}
    for b, bv in base_dates.items():
        for o, ov in other_dates.items():
            try:
                lo, hi = as_dates(ov, o).min(), as_dates(ov, o).max()
                bd = as_dates(bv, b)
            except ValueError:
                continue
            inside = float(np.mean((bd >= lo) & (bd <= hi)))
            suggestions.append({"kind": "time", "base": b, "lookup": o, "score": round(inside, 3),
                                "detail": f"{inside:.0%} of the base dates fall inside {lo} .. {hi}"})
    for b, bv in base.items():
        if b in base_dates or np.asarray(bv).dtype.kind == "f":
            continue
        base_set = set(key_strings(bv).tolist())
        for o, ov in other.items():
            if o in other_dates or np.asarray(ov).dtype.kind == "f":
                continue
            other_keys = key_strings(ov)
            overlap = len(base_set & set(other_keys.tolist())) / max(len(base_set), 1)
            similarity = difflib.SequenceMatcher(None, b.lower(), o.lower()).ratio()
            if overlap >= 0.5 or (overlap > 0 and similarity >= 0.8):
                unique = len(set(other_keys.tolist())) == len(other_keys)
                suggestions.append({"kind": "key", "base": b, "lookup": o, "score": round(overlap + 0.1 * similarity, 3),
                                    "detail": f"{overlap:.0%} of the base values exist in the lookup; "
                                              f"{'one row per key (a key join)' if unique else 'several rows per key (use with a time window)'}"})
    return sorted(suggestions, key=lambda s: -s["score"])[:limit]



def _looks_like_dates(values: Any) -> bool:
    arr = np.asarray(values)
    if arr.dtype.kind == "M":
        return True
    if arr.dtype.kind not in "OU" or len(arr) == 0:
        return False
    sample = [str(v) for v in arr[: min(len(arr), 20)].tolist()]
    return all(len(s) >= 8 and s[:4].isdigit() and s[4] in "-/" for s in sample)
