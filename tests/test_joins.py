"""Joins: manifest rules, the numpy engine (windows, keys, climatology, snapshot) and the automatic key check."""

from __future__ import annotations

import numpy as np
import pytest

from predictor.core.join_checks import check_join, suggest_joins
from predictor.core.join_values import as_dates
from predictor.core.joins import CLIMATOLOGY, NO_DATA, OBSERVED, UNKNOWN_KEY, JoinEngine, Lookup
from predictor.core.manifest import JoinConfig, Manifest

DATES = np.arange("2020-01-01", "2023-01-01", dtype="datetime64[D]")
# temp = the day number, so every window mean/sum can be checked by hand
WEATHER = {"date": DATES, "temp": np.arange(len(DATES), dtype=float), "rain": np.ones(len(DATES))}


def join_config(**over) -> JoinConfig:
    spec = {
        "name": "weather", "source": {"adapter": "csv-file", "path": "w.csv"},
        "time": {"base": "start", "lookup": "date"},
        "window": {"start": {"offset": 0}, "end": {"offset": 10, "unit": "days"}},
        "aggregate": {"t_mean": {"column": "temp", "op": "mean"}},
    }
    spec.update(over)
    return JoinConfig.model_validate(spec)


def engine_for(join: JoinConfig, table) -> JoinEngine:
    lookup = Lookup.build(table, key_columns=list(join.keys.values()), time_column=join.time.lookup if join.time else None,
                          value_columns=[a.column for a in join.aggregate.values()])
    return JoinEngine([join], {join.name: lookup})


def day(text: str) -> int:
    return int((np.datetime64(text) - DATES[0]).astype(int))


# --------------------------------------------------------------------------- manifest rules

MANIFEST = {"manifest_version": "1.0", "use_case_id": "demo", "data_source": {"adapter": "csv-file"},
            "training": {"target": "y", "time_column": "start"}}
WEATHER_JOIN = {"name": "weather", "source": {"adapter": "csv-file", "path": "w.csv"},
                "time": {"base": "start", "lookup": "date"}, "window": {"end": {"offset": 10}},
                "aggregate": {"t_mean": {"column": "temp", "op": "mean"}}}


def manifest(**over):
    return Manifest.model_validate({**MANIFEST, **over})


def test_a_manifest_without_joins_still_loads():
    assert manifest().joins == []


def test_a_valid_join_parses_with_defaults():
    m = manifest(joins=[WEATHER_JOIN])
    join = m.joins[0]
    assert join.base_inputs == ["start"] and join.new_columns == ["t_mean"]
    assert join.window.start.offset_days == 0 and join.window.end.offset_days == 10
    assert join.fallback == "error" and join.min_coverage == 0.9 and join.min_match == 0.95


@pytest.mark.parametrize(
    "change, message",
    [
        ({"time": None}, "needs 'keys', 'time' or both"),
        ({"window": None}, "needs a 'window'"),
        ({"keys": {"a": "b"}, "time": None, "window": {"end": {"offset": 1}}}, "'window' needs 'time'"),
        ({"keys": {"a": "b"}, "time": None, "window": None}, "use op 'first'"),
        ({"keys": {"a": "b"}, "time": None, "window": None, "aggregate": {"x": {"column": "c", "op": "first"}},
          "fallback": "climatology"}, "only applies to a time join"),
        ({"name": "Weather Data"}, "lowercase slug"),
        ({"aggregate": {}}, "at least 1 item"),
        ({"aggregate": {"x": {"column": "c", "op": "median"}}}, "Input should be"),
        ({"aggregate": {"x": {"column": "c", "scale": 0}}}, "scale may not be 0"),
        ({"min_coverage": 0}, "greater than 0"),
        ({"time": {"lookup": "date"}}, "name a 'column' or set 'time.base'"),
        ({"unknown_key": 1}, "Extra inputs"),
    ],
)
def test_invalid_joins_are_rejected(change, message):
    with pytest.raises(ValueError, match=message):
        manifest(joins=[{**WEATHER_JOIN, **change}])


def test_window_edges_may_come_from_base_columns():
    join = manifest(joins=[{**WEATHER_JOIN, "time": {"lookup": "date"}, "window": {
        "start": {"column": "season_start"}, "end": {"column": "season_end", "offset": 1}}}]).joins[0]
    assert join.base_inputs == ["season_start", "season_end"]


def test_join_names_and_new_columns_must_be_unique_and_may_not_clash():
    with pytest.raises(ValueError, match="names must be unique"):
        manifest(joins=[WEATHER_JOIN, WEATHER_JOIN])
    other = {**WEATHER_JOIN, "name": "other"}
    with pytest.raises(ValueError, match="already created by join 'weather'"):
        manifest(joins=[WEATHER_JOIN, other])
    with pytest.raises(ValueError, match="is the target"):
        manifest(joins=[{**WEATHER_JOIN, "aggregate": {"y": {"column": "temp"}}}])
    with pytest.raises(ValueError, match="is the drop_columns"):
        manifest(joins=[WEATHER_JOIN], training={**MANIFEST["training"], "drop_columns": ["t_mean"]})


def test_a_joined_column_cannot_be_a_scenario_lever():
    scenario = {"levers": {"t_mean": {"min": 0, "max": 10, "step": 1}}}
    with pytest.raises(ValueError, match="derived from join 'weather'"):
        manifest(joins=[WEATHER_JOIN], scenario=scenario)


def test_join_inputs_and_columns_obey_the_forbidden_fields_policy():
    with pytest.raises(ValueError, match="forbidden field in the policy"):
        manifest(joins=[WEATHER_JOIN], policy={"forbidden_fields": ["Start"]})


# --------------------------------------------------------------------------- windows and operations

def test_window_is_half_open_and_in_days():
    engine = engine_for(join_config(), WEATHER)
    result = engine.apply({"start": ["2020-01-11"]})
    assert result.columns["t_mean"][0] == pytest.approx(np.mean(np.arange(10, 20)))  # days 10..19, day 20 is excluded
    assert result.status["weather"][0] == OBSERVED


def test_weeks_unit_and_offset_start():
    join = join_config(window={"start": {"offset": 1, "unit": "weeks"}, "end": {"offset": 3, "unit": "weeks"}})
    result = engine_for(join, WEATHER).apply({"start": ["2020-01-01"]})
    assert result.columns["t_mean"][0] == pytest.approx(np.mean(np.arange(7, 21)))  # [7, 21)


def test_every_operation():
    aggregate = {f"t_{op}": {"column": "temp", "op": op} for op in ("mean", "sum", "min", "max", "count", "first", "last")}
    result = engine_for(join_config(aggregate=aggregate), WEATHER).apply({"start": ["2020-01-11"]})
    got = {k: float(v[0]) for k, v in result.columns.items()}
    assert got == {"t_mean": 14.5, "t_sum": 145.0, "t_min": 10.0, "t_max": 19.0, "t_count": 10.0, "t_first": 10.0, "t_last": 19.0}


def test_scale_and_round():
    aggregate = {"rain_mm": {"column": "rain", "op": "sum", "scale": 0.123456, "round": 2}}
    result = engine_for(join_config(aggregate=aggregate), WEATHER).apply({"start": ["2020-01-11"]})
    assert result.columns["rain_mm"][0] == 1.23


def test_nan_days_are_skipped_in_the_aggregate_and_count_as_gaps():
    table = {**WEATHER, "temp": WEATHER["temp"].copy()}
    table["temp"][12:15] = np.nan
    expected = np.mean([v for v in range(10, 20) if v not in (12, 13, 14)])
    lenient = engine_for(join_config(min_coverage=0.7), table).apply({"start": ["2020-01-11"]})
    assert lenient.columns["t_mean"][0] == pytest.approx(expected)
    assert lenient.status["weather"][0] == OBSERVED and lenient.coverage["weather"][0] == pytest.approx(0.7)
    strict = engine_for(join_config(), table).apply({"start": ["2020-01-11"]})  # 90% needed
    assert strict.status["weather"][0] == NO_DATA


def test_a_sensor_that_is_mostly_empty_does_not_pass_as_covered():
    """Rows exist for every day, but 9 of 10 cells are empty: a sum over that window would be far too low."""
    table = {**WEATHER, "rain": np.where(np.arange(len(DATES)) % 10 < 9, np.nan, 1.0)}
    join = join_config(aggregate={"r_sum": {"column": "rain", "op": "sum"}})
    result = engine_for(join, table).apply({"start": ["2020-01-21"]})
    assert result.status["weather"][0] == NO_DATA and result.coverage["weather"][0] == pytest.approx(0.1)


def test_window_edges_from_base_columns_with_inclusive_end():
    join = join_config(time={"lookup": "date"}, window={
        "start": {"column": "s"}, "end": {"column": "e", "offset": 1}})
    result = engine_for(join, WEATHER).apply({"s": ["2020-01-11"], "e": ["2020-01-20"]})  # both days included
    assert result.columns["t_mean"][0] == pytest.approx(np.mean(np.arange(10, 20)))


def test_batch_rows_are_independent_and_keep_their_order():
    starts = ["2021-06-01", "2020-01-11", "2022-03-03"]
    result = engine_for(join_config(), WEATHER).apply({"start": starts})
    expected = [np.mean(np.arange(day(s), day(s) + 10)) for s in starts]
    assert result.columns["t_mean"].tolist() == pytest.approx(expected)


# --------------------------------------------------------------------------- keys

PROVINCES = {"province": np.array(["Fryslân", "Drenthe"]), "station": np.array([270, 279]), "kind": np.array(["coast", "inland"])}


def key_join(**over) -> JoinConfig:
    spec = {"name": "station", "source": {"adapter": "csv-file", "path": "p.csv"}, "keys": {"prov": "province"},
            "aggregate": {"station_id": {"column": "station", "op": "first"}, "kind": {"column": "kind", "op": "first"}}}
    spec.update(over)
    return JoinConfig.model_validate(spec)


def test_key_join_copies_numbers_and_text():
    result = engine_for(key_join(), PROVINCES).apply({"prov": ["Drenthe", "Fryslân", "Drenthe"]})
    assert result.columns["station_id"].tolist() == [279.0, 270.0, 279.0]
    assert result.columns["kind"].tolist() == ["inland", "coast", "inland"]
    assert result.status["station"].tolist() == [OBSERVED] * 3


def test_key_join_marks_unknown_keys_and_leaves_them_empty():
    result = engine_for(key_join(), PROVINCES).apply({"prov": ["Drenthe", "Atlantis"]})
    assert result.status["station"].tolist() == [OBSERVED, UNKNOWN_KEY]
    assert result.columns["kind"][1] is None and np.isnan(result.columns["station_id"][1])


def test_keys_match_on_their_text_form():
    table = {"province": np.array([344, 280]), "station": np.array([1.0, 2.0]), "kind": np.array(["a", "b"])}
    result = engine_for(key_join(), table).apply({"prov": ["344", 280.0]})
    assert result.columns["station_id"].tolist() == [1.0, 2.0]


def test_key_and_time_join_use_the_weather_of_the_right_group():
    per_province = {"province": np.repeat(["A", "B"], len(DATES)), "date": np.tile(DATES, 2),
                    "temp": np.concatenate([WEATHER["temp"], WEATHER["temp"] + 1000])}
    join = join_config(keys={"prov": "province"})
    result = engine_for(join, per_province).apply({"prov": ["B", "A"], "start": ["2020-01-11", "2020-01-11"]})
    assert result.columns["t_mean"].tolist() == pytest.approx([1014.5, 14.5])


# --------------------------------------------------------------------------- coverage and the climatology fallback

def test_a_window_that_is_not_in_the_lookup_has_no_data_by_default():
    result = engine_for(join_config(), WEATHER).apply({"start": ["2022-12-28", "2030-01-01"]}, fallback=True)
    assert result.status["weather"].tolist() == [NO_DATA, NO_DATA]  # fallback is 'error' unless the join says otherwise
    assert np.isnan(result.columns["t_mean"]).all()


def test_min_coverage_decides_when_a_partial_window_is_accepted():
    strict = engine_for(join_config(), WEATHER).apply({"start": ["2022-12-25"]})  # 7 of 10 days
    assert strict.status["weather"][0] == NO_DATA and strict.coverage["weather"][0] == pytest.approx(0.7)
    lenient = engine_for(join_config(min_coverage=0.6), WEATHER).apply({"start": ["2022-12-25"]})
    assert lenient.status["weather"][0] == OBSERVED
    assert lenient.columns["t_mean"][0] == pytest.approx(np.mean(np.arange(day("2022-12-25"), day("2022-12-31") + 1)))


def test_climatology_averages_the_same_window_in_earlier_years():
    engine = engine_for(join_config(fallback="climatology"), WEATHER)
    result = engine.apply({"start": ["2023-01-05"]}, fallback=True)
    expected = np.mean([np.mean(np.arange(day(f"{y}-01-05"), day(f"{y}-01-05") + 10)) for y in (2020, 2021, 2022)])
    assert result.status["weather"][0] == CLIMATOLOGY and result.years["weather"][0] == 3
    assert result.columns["t_mean"][0] == pytest.approx(expected)
    # training never uses the fallback: the same row has no data
    assert engine.apply({"start": ["2023-01-05"]}, fallback=False).status["weather"][0] == NO_DATA


def test_observed_windows_are_never_replaced_by_climatology():
    engine = engine_for(join_config(fallback="climatology"), WEATHER)
    result = engine.apply({"start": ["2021-03-01"]}, fallback=True)
    assert result.status["weather"][0] == OBSERVED


def test_climatology_needs_at_least_one_covered_year():
    short = {k: v[:100] for k, v in WEATHER.items()}  # only Jan-Apr 2020
    result = engine_for(join_config(fallback="climatology"), short).apply({"start": ["2023-09-01"]}, fallback=True)
    assert result.status["weather"][0] == NO_DATA


# --------------------------------------------------------------------------- single records (serving)

FEATURES = {"area", "t_mean"}


def record_engine(**over) -> JoinEngine:
    return engine_for(join_config(fallback="climatology", **over), WEATHER)


def test_enrich_record_fills_the_derived_column_and_removes_the_join_input():
    features, errors, warnings = record_engine().enrich_record({"area": 5, "start": "2021-03-01"}, FEATURES)
    assert not errors and not warnings
    assert features == {"area": 5, "t_mean": pytest.approx(np.mean(np.arange(day("2021-03-01"), day("2021-03-01") + 10)))}


def test_explicit_values_win_and_need_no_join_input():
    features, errors, _ = record_engine().enrich_record({"area": 5, "t_mean": 3.5}, FEATURES)
    assert not errors and features == {"area": 5, "t_mean": 3.5}
    features, errors, _ = record_engine().enrich_record({"area": 5, "t_mean": 3.5, "start": "2021-03-01"}, FEATURES)
    assert features == {"area": 5, "t_mean": 3.5}  # the join input is still removed, it is not a feature


def test_a_join_input_that_is_also_a_feature_is_kept():
    features, _, _ = record_engine().enrich_record({"area": 5, "start": "2021-03-01"}, FEATURES | {"start"})
    assert features["start"] == "2021-03-01" and "t_mean" in features


def test_missing_and_invalid_join_inputs_give_readable_errors():
    _, errors, _ = record_engine().enrich_record({"area": 5}, FEATURES)
    assert [e.field for e in errors] == ["start"] and "required to look up ['t_mean']" in errors[0].message
    _, errors, _ = record_engine().enrich_record({"area": 5, "start": "next week"}, FEATURES)
    assert errors and "not a date" in errors[0].message


def test_future_dates_get_climatology_with_a_warning():
    features, errors, warnings = record_engine().enrich_record({"area": 5, "start": "2023-01-05"}, FEATURES)
    assert not errors and "t_mean" in features
    assert len(warnings) == 1 and "3 other year(s)" in warnings[0] and "not observed" in warnings[0]


def test_without_fallback_a_future_date_is_an_error():
    engine = engine_for(join_config(), WEATHER)
    _, errors, _ = engine.enrich_record({"area": 5, "start": "2023-01-05"}, FEATURES)
    assert errors and "no data in 'weather'" in errors[0].message and "send ['t_mean'] yourself" in errors[0].message


def test_unknown_keys_get_a_suggestion():
    engine = engine_for(key_join(), PROVINCES)
    _, errors, _ = engine.enrich_record({"prov": "Friesland"}, {"station_id", "kind", "prov"})
    assert [e.field for e in errors] == ["prov"] and "did you mean 'Fryslân'" in errors[0].message
    _, errors, _ = engine.enrich_record({"prov": "fryslan"}, {"station_id", "kind"})
    assert "differs only in case, accents or spaces from 'Fryslân'" in errors[0].message


def test_batch_and_single_record_agree_for_every_kind_of_row():
    """Train/serve parity: the trainer uses apply() on the table, the service enrich_record() per record."""
    engine = engine_for(join_config(fallback="climatology", aggregate={
        "t_mean": {"column": "temp", "op": "mean", "round": 3}, "r_sum": {"column": "rain", "op": "sum"}}), WEATHER)
    starts = ["2020-01-01", "2021-07-15", "2022-12-25", "2023-02-03", "2024-11-30"]
    batch = engine.apply({"start": starts}, fallback=True)
    for i, start in enumerate(starts):
        features, errors, _ = engine.enrich_record({"start": start}, {"t_mean", "r_sum"})
        assert not errors
        assert features["t_mean"] == pytest.approx(batch.columns["t_mean"][i]) and features["r_sum"] == pytest.approx(batch.columns["r_sum"][i])


def test_snapshot_round_trip_gives_identical_results():
    engine = JoinEngine([join_config(fallback="climatology"), key_join()],
                        {"weather": engine_for(join_config(), WEATHER).lookups["weather"],
                         "station": engine_for(key_join(), PROVINCES).lookups["station"]})
    files = engine.snapshot()
    assert set(files) == {"joins.json", "lookups/weather.npz", "lookups/station.npz"}
    loaded = JoinEngine.from_snapshot(lambda name: files[name].encode() if isinstance(files[name], str) else files[name])
    base = {"start": ["2021-03-01", "2023-01-05"], "prov": ["Drenthe", "Fryslân"]}
    a, b = engine.apply(base, fallback=True), loaded.apply(base, fallback=True)
    for name in a.columns:
        assert a.columns[name].tolist() == b.columns[name].tolist() or np.allclose(a.columns[name].astype(float), b.columns[name].astype(float), equal_nan=True)


def test_dates_are_parsed_strictly():
    assert as_dates(["2024-03-04", "2024-03-04 10:00:00"]).tolist() == [np.datetime64("2024-03-04")] * 2
    with pytest.raises(ValueError, match="'31-12-2024' is not a date"):
        as_dates(["2024-01-01", "31-12-2024"], "start")


# --------------------------------------------------------------------------- the automatic key check

def messages(report) -> str:
    return " | ".join(report.errors + report.warnings)


def test_a_clean_join_passes_and_reports_its_coverage():
    report = check_join(join_config(), {"start": ["2020-02-01", "2021-02-01"]}, WEATHER)
    assert report.ok and not report.warnings and report.stats["matched"] == 1.0 and report.lookup is not None


def test_a_misspelled_key_is_an_error_with_a_suggestion():
    base = {"prov": np.array(["Friesland"] * 3 + ["Drenthe"] * 7)}
    report = check_join(key_join(), base, PROVINCES)
    assert not report.ok and "70.0% of the rows match" in messages(report) and "did you mean 'Fryslân'" in messages(report)


def test_a_small_share_of_unknown_keys_is_only_a_warning_above_min_match():
    base = {"prov": np.array(["Atlantis"] + ["Drenthe"] * 49)}
    report = check_join(key_join(min_match=0.9), base, PROVINCES)
    assert report.ok and "Atlantis" in messages(report) and "dropped from training" in messages(report)


def test_keys_without_any_overlap_fail():
    report = check_join(key_join(), {"prov": ["x", "y"]}, PROVINCES)
    assert not report.ok and "0.0% of the rows match" in messages(report)


def test_duplicate_lookup_keys_would_multiply_rows():
    table = {"province": np.array(["A", "A"]), "station": np.array([1, 2]), "kind": np.array(["x", "y"])}
    report = check_join(key_join(), {"prov": ["A"]}, table)
    assert not report.ok and "more than one row" in messages(report)


def test_duplicate_key_and_date_rows_are_an_error():
    table = {"date": np.append(DATES, DATES[5]), "temp": np.append(WEATHER["temp"], 1.0)}
    report = check_join(join_config(), {"start": ["2020-02-01"]}, table)
    assert not report.ok and "duplicate (key, date) row" in messages(report)


def test_time_coverage_problems_are_counted():
    base = {"start": ["2020-02-01", "2022-12-28", "2019-01-01", "2020-03-01"]}
    report = check_join(join_config(min_match=0.5), base, WEATHER)
    text = messages(report)
    assert report.ok and "have only part of their window" in text and "no data at all in their window" in text
    assert report.stats["matched"] == 0.5


def test_gaps_inside_a_window_are_a_warning():
    table = {k: np.delete(v, [40]) for k, v in WEATHER.items()}  # 9 of 10 days = exactly min_coverage
    report = check_join(join_config(), {"start": ["2020-02-05", "2021-02-05"]}, table)
    assert report.ok and "miss some days inside their window" in messages(report)


def test_missing_columns_and_bad_dates_are_reported_not_raised():
    assert "has no column(s) ['start']" in messages(check_join(join_config(), {"other": [1]}, WEATHER))
    assert "has no column(s) ['temp']" in messages(check_join(join_config(), {"start": ["2020-02-01"]}, {"date": DATES}))
    assert "is not a date" in messages(check_join(join_config(), {"start": ["soon"]}, WEATHER))


def test_a_constant_new_column_is_flagged():
    table = {**WEATHER, "temp": np.full(len(DATES), 5.0)}
    report = check_join(join_config(), {"start": ["2020-02-01", "2021-02-01"]}, table)
    assert "is constant" in messages(report)


def test_key_type_differences_are_noted_but_still_match():
    table = {"province": np.array([344, 280]), "station": np.array([1.0, 2.0]), "kind": np.array(["a", "b"])}
    report = check_join(key_join(), {"prov": np.array(["344", "280"])}, table)
    assert report.ok and "compared as text" in " ".join(report.info)


def test_suggest_joins_proposes_keys_and_dates():
    base = {"province": np.array(["Fryslân", "Drenthe", "Drenthe"]), "day": np.array(["2020-02-01", "2020-03-01", "2021-01-01"]),
            "yield": np.array([1.5, 2.5, 3.5])}
    other = {"Province": np.array(["Fryslân", "Drenthe"]), "date": DATES[:3].astype(str), "temp": np.array([1.0, 2.0, 3.0])}
    found = suggest_joins(base, {**other, "Province": other["Province"]})
    kinds = {(s["kind"], s["base"], s["lookup"]) for s in found}
    assert ("key", "province", "Province") in kinds and ("time", "day", "date") in kinds
    assert not any(s["base"] == "yield" for s in found)


# --------------------------------------------------------------------------- review findings: strict dates, odd inputs

@pytest.mark.parametrize("value", [20240304, 5, "2024", "2024-03", "0001-01-01", "9999-12-31", "99999-01-01", ["2024-01-01"],
                                   {"d": 1}, True, 1.5, None, "soon"])
def test_only_real_dates_are_accepted(value):
    with pytest.raises(ValueError):
        as_dates([value], "start")


def test_dates_with_a_time_part_and_date_objects_are_accepted():
    import datetime

    parsed = as_dates(["2024-03-04T10:00:00", datetime.date(2024, 3, 4), np.datetime64("2024-03-04")])
    assert parsed.tolist() == [datetime.date(2024, 3, 4)] * 3


def test_odd_request_values_become_field_errors_not_crashes():
    engine = record_engine()
    for value in (["2021-01-01", "2021-01-02"], {"a": 1}, 20240101, "9999-12-31", 12.5):
        _, errors, _ = engine.enrich_record({"area": 5, "start": value}, FEATURES)
        assert errors and errors[0].field == "start", value


def test_a_null_derived_column_counts_as_not_sent():
    features, errors, _ = record_engine().enrich_record({"area": 5, "t_mean": None, "start": "2021-03-01"}, FEATURES)
    assert not errors and features["t_mean"] is not None


def test_huge_and_tiny_scales_do_not_divide_by_zero():
    for scale, expected in ((1e10, 14.5e10), (1e-10, 14.5e-10), (0.01, 0.145)):
        aggregate = {"v": {"column": "temp", "op": "mean", "scale": scale}}
        result = engine_for(join_config(aggregate=aggregate), WEATHER).apply({"start": ["2020-01-11"]})
        assert result.columns["v"][0] == pytest.approx(expected)


def test_float32_keys_match_their_float64_twins():
    table = {"province": np.array([0.1, 0.2]), "station": np.array([1, 2]), "kind": np.array(["a", "b"])}
    result = engine_for(key_join(), table).apply({"prov": np.array([0.1, 0.2], dtype=np.float32)})
    assert result.status["station"].tolist() == [OBSERVED, OBSERVED]


# --------------------------------------------------------------------------- empty lookups

def test_an_empty_lookup_is_reported_not_crashed_on():
    empty = {"date": np.array([], dtype="datetime64[D]"), "temp": np.array([])}
    report = check_join(join_config(min_match=0.01), {"start": ["2022-01-01"]}, empty)
    assert not report.ok and "lookup table has no rows" in messages(report)
    engine = engine_for(join_config(fallback="climatology"), empty)
    _, errors, _ = engine.enrich_record({"start": "2022-01-01"}, {"t_mean"})
    assert errors and "has no rows" in errors[0].message


# --------------------------------------------------------------------------- chained joins and clashes

PLOTS = {"plot": np.array(["p1", "p2"]), "zone": np.array(["z1", "z2"])}
ZONES = {"zone": np.array(["z1", "z2"]), "ph": np.array([5.5, 7.0])}


def chained() -> JoinEngine:
    plots = JoinConfig.model_validate({"name": "plots", "source": {"adapter": "csv-file", "path": "a.csv"}, "keys": {"plot": "plot"},
                                       "aggregate": {"zone": {"column": "zone", "op": "first"}}})
    zones = JoinConfig.model_validate({"name": "zones", "source": {"adapter": "csv-file", "path": "b.csv"}, "keys": {"zone": "zone"},
                                       "aggregate": {"ph": {"column": "ph", "op": "first"}}})
    return JoinEngine([plots, zones], {"plots": engine_for(plots, PLOTS).lookups["plots"],
                                       "zones": engine_for(zones, ZONES).lookups["zones"]})


def test_a_join_can_be_keyed_on_a_column_an_earlier_join_created():
    engine = chained()
    assert engine.apply({"plot": ["p2", "p1"]}).columns["ph"].tolist() == [7.0, 5.5]
    features, errors, _ = engine.enrich_record({"plot": "p2"}, {"ph"})
    assert not errors and features == {"ph": 7.0}  # the intermediate zone is a join input, not a feature
    features, errors, _ = engine.enrich_record({"zone": "z1"}, {"ph"})  # the caller may skip the first step
    assert not errors and features == {"ph": 5.5}


def test_a_new_column_may_not_silently_replace_a_base_column():
    report = check_join(join_config(), {"start": ["2020-02-01"], "t_mean": [1.0]}, WEATHER)
    assert not report.ok and "already exist in the base table" in messages(report)


# --------------------------------------------------------------------------- review findings: manifest rules

@pytest.mark.parametrize(
    "join_change, training_change, message",
    [
        ({"window": {"start": {"offset": 5}, "end": {"offset": 1}}}, {}, "window.end must be after window.start"),
        ({"window": {"end": {"offset": 0}}}, {}, "window.end must be after window.start"),
        ({"aggregate": {"": {"column": "temp"}}}, {}, "non-empty names"),
        ({"aggregate": {"x": {"column": ""}}}, {}, "at least 1 character"),
        ({"keys": {"": "x"}}, {}, "non-empty column names"),
        ({}, {"horizon_steps": 3}, "not supported together with horizon_steps"),
        ({}, {"rename_columns": {"start": "begin"}}, "'start' is in rename_columns"),
        ({}, {"rename_columns": {"t_mean": "temperature"}}, "'t_mean' is in rename_columns"),
    ],
)
def test_more_nonsense_is_rejected(join_change, training_change, message):
    with pytest.raises(ValueError, match=message):
        manifest(joins=[{**WEATHER_JOIN, **join_change}], training={**MANIFEST["training"], **training_change})


def test_windows_on_different_columns_may_have_any_offsets():
    m = manifest(joins=[{**WEATHER_JOIN, "time": {"lookup": "date"},
                         "window": {"start": {"column": "a", "offset": 5}, "end": {"column": "b", "offset": 0}}}])
    assert m.joins[0].base_inputs == ["a", "b"]
