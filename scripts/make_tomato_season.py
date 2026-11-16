"""Harvest months ahead: tomato seasons from a crop model calibrated on real greenhouses, on real weather.

    python scripts/make_tomato_season.py                       # writes data/tomato_season.csv
    python scripts/make_tomato_season.py --weather synthetic   # offline
    python scripts/make_tomato_season.py --check               # after `predictor train tomato-season`:
                                                               # do the scenarios match the crop model?

The grower's question: "if I change my settings now, what is my harvest in 5 months?" No open dataset
can answer that from data alone (the real greenhouses ended within 5% of each other), so the seasons
here come from the crop model in `tomato_crop.py`, which is fitted on those real greenhouses and tested
on them (season total 3.7% off). Every season runs on real KNMI weather of one of ten years.

One row = one decision: a greenhouse some weeks into its season, the settings so far, and the plan from
now on. The outputs are the harvest in the next 1, 3 and 5 months. The weather of those months is not
an input, because nobody knows it in advance; that is what makes the error of the trained model an
honest margin for a forecast months ahead.

The simulator is also the answer key: for every row it can tell what another plan would have given in
the same weather, which a real greenhouse never can.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import knmi
import tomato_crop as crop

STATION = 344                       # Rotterdam, the nearest KNMI station to Bleiswijk
START, END = "2015-09-01", "2025-12-31"
SEASON_DAYS = 300
HORIZONS = {"harvest_1_month_kg_m2": 30, "harvest_3_months_kg_m2": 91, "harvest_5_months_kg_m2": 152}
NORMAL_TEMP_C, NORMAL_LAMP_HOURS = 22.0, 14.0
SCENARIOS = {"1 degree warmer": {"temp_plan_c": 1.0}, "1 degree colder": {"temp_plan_c": -1.0},
             "4 lamp hours more": {"lamp_hours_plan": 4.0}, "4 lamp hours less": {"lamp_hours_plan": -4.0}}


def lamp_hours(radiation: np.ndarray, hours_on_dark_days: float) -> np.ndarray:
    """Growers light on dark days and switch off as the sun takes over (as in the real data: 16 h in winter, 0 in May)."""
    week = pd.Series(radiation).rolling(7, min_periods=1).mean().to_numpy()
    return hours_on_dark_days * np.clip((1500.0 - week) / 900.0, 0.0, 1.0)


def season(weather: pd.DataFrame, decision_day: int, before: tuple[float, float], after: tuple[float, float],
           noise: np.ndarray | None = None) -> dict[str, float]:
    """Run one season with a plan before and after the decision day; return the harvest after the decision."""
    days = np.arange(len(weather))
    temp_plan = np.where(days < decision_day, before[0], after[0])
    hours = np.where(days < decision_day, before[1], after[1])
    radiation = weather["radiation_j_cm2"].to_numpy()
    lamps = lamp_hours(radiation, 1.0) * hours
    # The greenhouse follows the plan; on warm days it cannot stay below the outside temperature plus sun warmth.
    temp = np.maximum(temp_plan + (noise if noise is not None else 0.0), weather["temp_mean_c"].to_numpy() + 4.0)
    harvest = crop.simulate(temp, crop.light_at_crop(radiation, lamps))
    return {name: float(harvest[decision_day: decision_day + horizon].sum()) for name, horizon in HORIZONS.items()}


def climatology(daily: pd.DataFrame) -> pd.DataFrame:
    """The average year: per day of the year the mean temperature and radiation (what is known in advance)."""
    doy = daily["date"].dt.day_of_year.clip(upper=365)
    return daily.groupby(doy)[["temp_mean_c", "radiation_j_cm2"]].mean()


def _window(daily: pd.DataFrame, planting: pd.Timestamp) -> pd.DataFrame:
    start = int(daily["date"].searchsorted(planting))
    return daily.iloc[start: start + SEASON_DAYS].reset_index(drop=True)


def generate(daily: pd.DataFrame, seasons: int = 4000, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    daily = daily.dropna(subset=["temp_mean_c", "radiation_j_cm2"]).reset_index(drop=True)
    normal = climatology(daily)
    last_planting = daily["date"].max() - pd.Timedelta(days=SEASON_DAYS)
    years = range(daily["date"].min().year, last_planting.year + 1)
    rows = []
    for i in range(seasons):
        # Planted between October and January, like the real crop (December).
        planting = pd.Timestamp(year=int(rng.choice(list(years))), month=10, day=1) + pd.Timedelta(days=int(rng.integers(0, 120)))
        if planting > last_planting or planting < daily["date"].min():
            continue
        weather = _window(daily, planting)
        decision_day = int(rng.choice([0, 28, 56, 84, 112, 140]))
        before = (float(np.clip(rng.normal(NORMAL_TEMP_C, 0.9), 19.5, 24.5)), float(rng.uniform(6, 18)))
        if decision_day == 0 or rng.random() < 0.5:
            after = before                                                  # most growers keep doing what they did
        else:                                                               # some change course
            after = (float(np.clip(before[0] + rng.uniform(-2, 2), 19.5, 24.5)), float(np.clip(before[1] + rng.uniform(-6, 6), 0, 18)))
        noise = rng.normal(0, 0.4, SEASON_DAYS)
        outcome = season(weather, decision_day, before, after, noise)
        # The plain guess: a normal plan in an average year, from the same point in the season.
        doy = weather["date"].dt.day_of_year.clip(upper=365).to_numpy()
        average_year = pd.DataFrame({"temp_mean_c": normal["temp_mean_c"].reindex(doy).to_numpy(),
                                     "radiation_j_cm2": normal["radiation_j_cm2"].reindex(doy).to_numpy()})
        usual = season(average_year, decision_day, (NORMAL_TEMP_C, NORMAL_LAMP_HOURS), (NORMAL_TEMP_C, NORMAL_LAMP_HOURS))
        decision = planting + pd.Timedelta(days=decision_day)
        rows.append({
            "season_id": f"S{i:05d}", "decision_date": decision.strftime("%Y-%m-%d"), "planting_date": planting.strftime("%Y-%m-%d"),
            "decision_week": int(decision.isocalendar().week), "days_since_planting": decision_day,
            "temp_so_far_c": round(before[0], 2), "lamp_hours_so_far": round(before[1], 1),
            "temp_plan_c": round(after[0], 2), "lamp_hours_plan": round(after[1], 1),
            **{f"usual_{name}": round(value, 3) for name, value in usual.items()},
            **{name: round(value, 3) for name, value in outcome.items()},
        })
    return pd.DataFrame(rows).sort_values(["decision_date", "season_id"]).reset_index(drop=True)


def counterfactual(row: pd.Series, daily: pd.DataFrame, **changes: float) -> dict[str, float]:
    """The answer key: the same greenhouse in the same weather with another plan (without the daily noise)."""
    weather = _window(daily, pd.Timestamp(row["planting_date"]))
    before = (float(row["temp_so_far_c"]), float(row["lamp_hours_so_far"]))
    after = (float(row["temp_plan_c"]) + changes.get("temp_plan_c", 0.0),
             float(np.clip(float(row["lamp_hours_plan"]) + changes.get("lamp_hours_plan", 0.0), 0, 24)))
    return season(weather, int(row["days_since_planting"]), before, after)


def check(data: Path, daily: pd.DataFrame, use_case: str = "tomato-season", rows: int = 300,
          test_from: str = "2023-07-01") -> pd.DataFrame:
    """Scenarios through the real engine against the crop model, on decisions from the test period."""
    import logging
    import warnings

    from fastapi.testclient import TestClient

    from predictor import bootstrap

    warnings.filterwarnings("ignore")
    logging.disable(logging.INFO)
    client = TestClient(bootstrap.build_app(Path("usecases"), Path("models")))
    order = client.get(f"/models/{use_case}").json()["feature_schema"]["x-feature-order"]
    daily = daily.dropna(subset=["temp_mean_c", "radiation_j_cm2"]).reset_index(drop=True)
    df = pd.read_csv(data)
    test = df[(df["decision_date"] >= test_from) & (df["days_since_planting"] > 0)]   # all seasons of the test years
    sample = test.sample(min(rows, len(test)), random_state=1)
    records = []
    for _, row in sample.iterrows():
        features = {k: float(row[k]) for k in order}
        base_true = counterfactual(row, daily)
        asked = [{"name": name, "changes": {k: features[k] + v for k, v in change.items()}} for name, change in SCENARIOS.items()
                 if all(0 <= features[k] + v <= 26 for k, v in change.items())]
        body = client.post(f"/models/{use_case}/scenario", json={"base": {"features": features}, "scenarios": asked}).json()
        for scenario in body.get("scenarios", []):
            truth = counterfactual(row, daily, **SCENARIOS[scenario["name"]])
            for output in HORIZONS:
                records.append({"scenario": scenario["name"], "output": output, "level": base_true[output],
                                "true": truth[output] - base_true[output],
                                "predicted": scenario["difference"][output]["change"]})
    r = pd.DataFrame(records)
    r["same_direction"] = np.sign(r["true"].round(3)) == np.sign(r["predicted"].round(3))
    grouped = r.groupby(["scenario", "output"], sort=False)
    return pd.DataFrame({
        "cases": grouped.size(),
        "true_change_pct": grouped.apply(lambda g: g["true"].sum() / g["level"].sum() * 100),
        "predicted_change_pct": grouped.apply(lambda g: g["predicted"].sum() / g["level"].sum() * 100),
        "direction_right_pct": grouped["same_direction"].mean() * 100,
        "per_case_corr": grouped.apply(lambda g: g["true"].corr(g["predicted"])),
    }).round(2).reset_index()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("data/tomato_season.csv"))
    parser.add_argument("--seasons", type=int, default=4000)
    parser.add_argument("--weather", choices=["knmi", "synthetic"], default="knmi")
    parser.add_argument("--check", action="store_true", help="compare the trained model's scenarios with the crop model")
    args = parser.parse_args()
    daily = knmi.load_daily(STATION, START, END) if args.weather == "knmi" else knmi.synthetic_daily(START, END)
    if args.check:
        print(check(args.out, daily).to_string(index=False))
        return
    df = generate(daily, args.seasons)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"wrote {args.out}: {len(df)} decisions, {df['decision_date'].min()} .. {df['decision_date'].max()}; "
          f"harvest in 5 months {df['harvest_5_months_kg_m2'].mean():.1f} kg/m2 on average "
          f"({df['harvest_5_months_kg_m2'].quantile(0.05):.1f} - {df['harvest_5_months_kg_m2'].quantile(0.95):.1f})")


if __name__ == "__main__":
    main()
