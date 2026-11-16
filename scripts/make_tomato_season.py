"""Harvest months ahead: tomato seasons from a crop model calibrated on real greenhouses, on real weather.

    python scripts/make_tomato_season.py                       # writes data/tomato_season.csv
    python scripts/make_tomato_season.py --weather synthetic   # offline
    python scripts/make_tomato_season.py --check               # after `predictor train tomato-season`:
                                                               # do the scenarios match the crop model?
    python scripts/make_tomato_season.py --check-finder        # does the best-plan finder's advice make money?

The grower's question: "if I change my settings now, what is my harvest in 5 months?" No open dataset
can answer that from data alone (the real greenhouses ended within 5% of each other), so the seasons
here come from the crop model in `tomato_crop.py`, which is fitted on those real greenhouses and tested
on them (season total 3.7% off). Every season runs on real KNMI weather of one of ten years.

One row = one decision: a greenhouse some weeks into its season, the settings so far, and the plan from
now on. The outputs are the harvest in the next 1, 3 and 5 months, and what those 5 months cost: heating
energy and lamp electricity. With the grower's prices as inputs, one scenario gives the net result in euros.
The weather of those months is not an input, because nobody knows it in advance; that is what makes the
error of the trained model an honest margin for a forecast months ahead.

Three pieces, each fitted on the real greenhouses:

* harvest: the crop model in `tomato_crop.py`;
* heating: the heat balance in `greenhouse_heat.py`, times the greenhouse's own factor. Growers differ a
  factor 0.5 to 1.7 at the same temperature, so every greenhouse here has its own, drifting a little over
  the season, and the row holds its heating use of the last four weeks (what a real grower would know);
* electricity: lamp hours times the measured 0.131 kWh/m2 per lamp hour.

The simulator is also the answer key: for every row it can tell what another plan would have given in
the same weather, which a real greenhouse never can.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import greenhouse_heat as heating
import knmi
import tomato_crop as crop

STATION = 344                       # Rotterdam, the nearest KNMI station to Bleiswijk
START, END = "2015-08-01", "2025-12-31"
SEASON_DAYS = 300
HISTORY_DAYS = 28                   # the grower's own heating use before the decision
HORIZONS = {"harvest_1_month_kg_m2": 30, "harvest_3_months_kg_m2": 91, "harvest_5_months_kg_m2": 152}
COSTS = ["heating_5_months_mj_m2", "electricity_5_months_kwh_m2"]
OUTPUTS = [*HORIZONS, *COSTS]
# Tomato price per month (EUR/kg): January-June from the challenge's price table (mean of its two quality
# classes), July-December an assumption that closes the year. A grower enters his own price.
TOMATO_PRICE = {1: 4.1, 2: 3.8, 3: 3.0, 4: 2.3, 5: 2.0, 6: 1.8, 7: 1.8, 8: 1.9, 9: 2.2, 10: 2.6, 11: 3.2, 12: 3.8}
NORMAL_TEMP_C, NORMAL_LAMP_HOURS = 22.0, 14.0
SCENARIOS = {"1 degree warmer": {"temp_plan_c": 1.0}, "1 degree colder": {"temp_plan_c": -1.0},
             "4 lamp hours more": {"lamp_hours_plan": 4.0}, "4 lamp hours less": {"lamp_hours_plan": -4.0}}


def lamp_hours(radiation: np.ndarray, hours_on_dark_days: float) -> np.ndarray:
    """Growers light on dark days and switch off as the sun takes over (as in the real data: 16 h in winter, 0 in May)."""
    week = pd.Series(radiation).rolling(7, min_periods=1).mean().to_numpy()
    return hours_on_dark_days * np.clip((1500.0 - week) / 900.0, 0.0, 1.0)


def season(weather: pd.DataFrame, decision_day: int, before: tuple[float, float], after: tuple[float, float],
           noise: np.ndarray | None = None, factor: np.ndarray | None = None) -> dict[str, float]:
    """Run one season with a plan before and after the decision day.

    `weather` starts HISTORY_DAYS before planting (the greenhouse was in use then too). Returns the harvest and
    the costs after the decision, and the heating use of the four weeks before it.
    """
    days = np.arange(len(weather)) - HISTORY_DAYS                   # day 0 = planting
    temp_plan = np.where(days < decision_day, before[0], after[0])
    hours = np.where(days < decision_day, before[1], after[1])
    radiation, outside = weather["radiation_j_cm2"].to_numpy(), weather["temp_mean_c"].to_numpy()
    lamps = lamp_hours(radiation, 1.0) * hours
    # The greenhouse follows the plan; on warm days it cannot stay below the outside temperature plus sun warmth.
    temp = np.maximum(temp_plan + (noise if noise is not None else 0.0), outside + 4.0)
    crop_days = days >= 0
    harvest = crop.simulate(temp[crop_days], crop.light_at_crop(radiation[crop_days], lamps[crop_days]))
    lamp_kwh = lamps * heating.KWH_PER_LAMP_HOUR
    heat = heating.heat_balance(temp, outside, radiation, lamp_kwh) * (factor if factor is not None else 1.0)
    start = decision_day + HISTORY_DAYS
    result = {name: float(harvest[decision_day: decision_day + horizon].sum()) for name, horizon in HORIZONS.items()}
    result["heating_5_months_mj_m2"] = float(heat[start: start + 152].sum())
    result["electricity_5_months_kwh_m2"] = float(lamp_kwh[start: start + 152].sum())
    result["heat_last_4_weeks_mj_m2"] = float(heat[start - HISTORY_DAYS: start].sum())
    return result


def own_factor_path(seed: int, length: int) -> np.ndarray:
    """A greenhouse's own heating factor over the season: its level, drifting a little every month."""
    rng = np.random.default_rng(seed)
    level = np.exp(rng.normal(0.0, heating.OWN_FACTOR_SPREAD))
    months = np.cumsum(rng.normal(0.0, heating.OWN_FACTOR_DRIFT, length // 30 + 2))
    return np.clip(level * np.exp(months[np.arange(length) // 30] - months[0]), 0.35, 2.5)


def climatology(daily: pd.DataFrame) -> pd.DataFrame:
    """The average year: per day of the year the mean temperature and radiation (what is known in advance)."""
    doy = daily["date"].dt.day_of_year.clip(upper=365)
    return daily.groupby(doy)[["temp_mean_c", "radiation_j_cm2"]].mean()


def _window(daily: pd.DataFrame, planting: pd.Timestamp) -> pd.DataFrame:
    start = int(daily["date"].searchsorted(planting)) - HISTORY_DAYS
    return daily.iloc[start: start + HISTORY_DAYS + SEASON_DAYS].reset_index(drop=True)


def prices(rng: np.random.Generator, decision: pd.Timestamp) -> dict[str, float]:
    """The grower's prices: tomatoes follow the season, energy differs per grower and contract.

    Drawn independently of the year on purpose: a price that marks the year would let a model use it as a
    hidden weather label.
    """
    months = [(decision + pd.DateOffset(months=m)).month for m in range(5)]
    return {"tomato_price_eur_kg": round(float(np.mean([TOMATO_PRICE[m] for m in months]) * rng.normal(1.0, 0.1)), 2),
            "heat_price_eur_mj": round(float(np.clip(np.exp(rng.normal(np.log(0.012), 0.45)), 0.005, 0.04)), 4),
            "electricity_price_eur_kwh": round(float(np.clip(np.exp(rng.normal(np.log(0.10), 0.4)), 0.04, 0.30)), 3)}


def generate(daily: pd.DataFrame, seasons: int = 4000, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    daily = daily.dropna(subset=["temp_mean_c", "radiation_j_cm2"]).reset_index(drop=True)
    normal = climatology(daily)
    first_planting = daily["date"].min() + pd.Timedelta(days=HISTORY_DAYS)
    last_planting = daily["date"].max() - pd.Timedelta(days=SEASON_DAYS)
    years = list(range(daily["date"].min().year, last_planting.year + 1))
    length = HISTORY_DAYS + SEASON_DAYS
    rows = []
    for i in range(seasons):
        # Planted between October and January, like the real crop (December).
        planting = pd.Timestamp(year=int(rng.choice(years)), month=10, day=1) + pd.Timedelta(days=int(rng.integers(0, 120)))
        decision_day = int(rng.choice([0, 28, 56, 84, 112, 140]))
        before = (float(np.clip(rng.normal(NORMAL_TEMP_C, 0.9), 19.5, 24.5)), float(rng.uniform(6, 18)))
        if decision_day == 0 or rng.random() < 0.5:
            after = before                                                  # most growers keep doing what they did
        else:                                                               # some change course
            after = (float(np.clip(before[0] + rng.uniform(-2, 2), 19.5, 24.5)), float(np.clip(before[1] + rng.uniform(-6, 6), 0, 18)))
        noise = rng.normal(0, 0.4, length)
        greenhouse_seed = int(rng.integers(0, 2**31 - 1))
        if planting > last_planting or planting < first_planting:
            continue
        weather = _window(daily, planting)
        outcome = season(weather, decision_day, before, after, noise, own_factor_path(greenhouse_seed, length))
        # The plain guess: a normal plan in an average year and an average greenhouse, from the same point in the season.
        doy = weather["date"].dt.day_of_year.clip(upper=365).to_numpy()
        average_year = pd.DataFrame({"temp_mean_c": normal["temp_mean_c"].reindex(doy).to_numpy(),
                                     "radiation_j_cm2": normal["radiation_j_cm2"].reindex(doy).to_numpy()})
        usual = season(average_year, decision_day, (NORMAL_TEMP_C, NORMAL_LAMP_HOURS), (NORMAL_TEMP_C, NORMAL_LAMP_HOURS))
        decision = planting + pd.Timedelta(days=decision_day)
        rows.append({
            "season_id": f"S{i:05d}", "decision_date": decision.strftime("%Y-%m-%d"), "planting_date": planting.strftime("%Y-%m-%d"),
            "greenhouse_seed": greenhouse_seed,
            "decision_week": int(decision.isocalendar().week), "days_since_planting": decision_day,
            "temp_so_far_c": round(before[0], 2), "lamp_hours_so_far": round(before[1], 1),
            "heat_last_4_weeks_mj_m2": round(outcome["heat_last_4_weeks_mj_m2"], 2),
            "temp_plan_c": round(after[0], 2), "lamp_hours_plan": round(after[1], 1),
            **prices(rng, decision),
            **{f"usual_{name}": round(usual[name], 3) for name in OUTPUTS},
            **{name: round(outcome[name], 3) for name in OUTPUTS},
        })
    return pd.DataFrame(rows).sort_values(["decision_date", "season_id"]).reset_index(drop=True)


def counterfactual(row: pd.Series, daily: pd.DataFrame, **changes: float) -> dict[str, float]:
    """The answer key: the same greenhouse in the same weather with another plan (without the daily noise)."""
    weather = _window(daily, pd.Timestamp(row["planting_date"]))
    before = (float(row["temp_so_far_c"]), float(row["lamp_hours_so_far"]))
    after = (float(row["temp_plan_c"]) + changes.get("temp_plan_c", 0.0),
             float(np.clip(float(row["lamp_hours_plan"]) + changes.get("lamp_hours_plan", 0.0), 0, 24)))
    factor = own_factor_path(int(row["greenhouse_seed"]), len(weather))        # the same greenhouse
    return season(weather, int(row["days_since_planting"]), before, after, factor=factor)


def net_eur(row: pd.Series, outcome: dict[str, float]) -> float:
    """Income minus energy costs over the next 5 months, at the grower's prices (EUR/m2)."""
    return (outcome["harvest_5_months_kg_m2"] * float(row["tomato_price_eur_kg"])
            - outcome["heating_5_months_mj_m2"] * float(row["heat_price_eur_mj"])
            - outcome["electricity_5_months_kwh_m2"] * float(row["electricity_price_eur_kwh"]))


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
            for output in OUTPUTS:
                records.append({"scenario": scenario["name"], "output": output, "level": base_true[output],
                                "true": truth[output] - base_true[output],
                                "predicted": scenario["difference"][output]["change"]})
            records.append({"scenario": scenario["name"], "output": "net result (EUR/m2)", "level": abs(net_eur(row, base_true)),
                            "true": net_eur(row, truth) - net_eur(row, base_true), "predicted": scenario["net_eur"]})
    r = pd.DataFrame(records)
    r["same_direction"] = np.sign(r["true"].round(3)) == np.sign(r["predicted"].round(3))
    grouped = r.groupby(["scenario", "output"], sort=False)
    return pd.DataFrame({
        "cases": grouped.size(),
        "true_change_pct": grouped.apply(lambda g: g["true"].sum() / g["level"].sum() * 100),
        "predicted_change_pct": grouped.apply(lambda g: g["predicted"].sum() / g["level"].sum() * 100),
        "direction_right_pct": grouped["same_direction"].mean() * 100,
        "per_case_corr": grouped.apply(lambda g: g["true"].corr(g["predicted"])),
        "true_change": grouped["true"].mean(), "predicted_change": grouped["predicted"].mean(),
    }).round(2).reset_index()


def check_finder(data: Path, daily: pd.DataFrame, use_case: str = "tomato-season", rows: int = 120,
                 test_from: str = "2023-07-01") -> pd.DataFrame:
    """Does following the best-plan finder make money? Its advice against the crop model's true outcome of every plan."""
    import logging
    import warnings

    from fastapi.testclient import TestClient

    from predictor import bootstrap

    warnings.filterwarnings("ignore")
    logging.disable(logging.INFO)
    client = TestClient(bootstrap.build_app(Path("usecases"), Path("models")))
    model = client.get(f"/models/{use_case}").json()
    order = model["feature_schema"]["x-feature-order"]
    levers = {lever["name"]: lever for lever in model["scenario"]["levers"]}
    grid = [(t, h) for t in np.arange(levers["temp_plan_c"]["min"], levers["temp_plan_c"]["max"] + 1e-9, levers["temp_plan_c"]["step"])
            for h in np.arange(levers["lamp_hours_plan"]["min"], levers["lamp_hours_plan"]["max"] + 1e-9, levers["lamp_hours_plan"]["step"])]
    daily = daily.dropna(subset=["temp_mean_c", "radiation_j_cm2"]).reset_index(drop=True)
    df = pd.read_csv(data)
    test = df[(df["decision_date"] >= test_from) & (df["days_since_planting"] > 0)]
    records = []
    for _, row in test.sample(min(rows, len(test)), random_state=2).iterrows():
        def true_net(temp: float, hours: float) -> float:
            return net_eur(row, counterfactual(row, daily, temp_plan_c=temp - float(row["temp_plan_c"]),
                                                lamp_hours_plan=hours - float(row["lamp_hours_plan"])))
        keep = true_net(float(row["temp_plan_c"]), float(row["lamp_hours_plan"]))
        best_possible = max(true_net(t, h) for t, h in grid) - keep
        features = {k: float(row[k]) for k in order}
        record = {"best_possible": best_possible}
        for name, options in [("advice", {}), ("any_gain", {"min_gain": 0}), ("certain", {"min_certainty": "medium", "min_gain": 0})]:
            body = client.post(f"/models/{use_case}/scenario/best", json={"base": {"features": features}, **options}).json()
            plan = body["best"]["plan"] if body.get("best") else None
            record[name] = true_net(plan["temp_plan_c"], plan["lamp_hours_plan"]) - keep if plan else 0.0
            record[f"{name}_promised"] = body["best"]["net_eur"] if plan else 0.0
            record[f"{name}_given"] = plan is not None
        records.append(record)
    r = pd.DataFrame(records)
    summary = {}
    for name, label in [("advice", "the finder's advice (default)"), ("any_gain", "advising every gain, however small"),
                        ("certain", "only plans with at least medium certainty")]:
        given = r[f"{name}_given"]
        summary[label] = {"advice given %": given.mean() * 100, "promised gain": r[f"{name}_promised"].mean(),
                          "true gain": r[name].mean(), "share of the best possible %": r[name].sum() / r["best_possible"].sum() * 100,
                          "made it worse % (of advice given)": (r.loc[given, name] < -0.05).mean() * 100, "worst case": r[name].min()}
    summary["the best plan in hindsight"] = {"advice given %": 100.0, "promised gain": np.nan, "true gain": r["best_possible"].mean(),
                                             "share of the best possible %": 100.0, "made it worse % (of advice given)": 0.0,
                                             "worst case": r["best_possible"].min()}
    return pd.DataFrame(summary).T.round(2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("data/tomato_season.csv"))
    parser.add_argument("--seasons", type=int, default=4000)
    parser.add_argument("--weather", choices=["knmi", "synthetic"], default="knmi")
    parser.add_argument("--check", action="store_true", help="compare the trained model's scenarios with the crop model")
    parser.add_argument("--check-finder", action="store_true", help="does following the best-plan finder make money?")
    args = parser.parse_args()
    daily = knmi.load_daily(STATION, START, END) if args.weather == "knmi" else knmi.synthetic_daily(START, END)
    if args.check:
        print(check(args.out, daily).to_string(index=False))
        return
    if args.check_finder:
        print("net result per m2 over 5 months, compared with keeping the current plan (EUR):")
        print(check_finder(args.out, daily).to_string())
        return
    df = generate(daily, args.seasons)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"wrote {args.out}: {len(df)} decisions, {df['decision_date'].min()} .. {df['decision_date'].max()}; "
          f"harvest in 5 months {df['harvest_5_months_kg_m2'].mean():.1f} kg/m2 on average "
          f"({df['harvest_5_months_kg_m2'].quantile(0.05):.1f} - {df['harvest_5_months_kg_m2'].quantile(0.95):.1f})")


if __name__ == "__main__":
    main()
