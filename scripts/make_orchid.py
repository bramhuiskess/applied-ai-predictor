"""Pot-orchid batches through three phases, on real weather: when to move a batch on, and at what temperature?

    python scripts/make_orchid.py                      # writes data/orchid_batches.csv
    python scripts/make_orchid.py --weather synthetic  # offline
    python scripts/make_orchid.py --check              # after `predictor train orchid-batch`: scenarios vs the crop model
    python scripts/make_orchid.py --check-finder       # does the best-plan finder's advice make money?

One row = one batch: when it started, how long it stayed in each phase and at what temperature, and what came
out: how many plants can be sold and how many have two spikes (the better-paid class), weeks to sale, and the
heating, cooling electricity and space per plant. With the grower's prices a scenario gives the net result per
plant, and the best-plan finder proposes the schedule.

The plant side comes from `orchid_crop.py`, which is built on published research and not fitted on
measurements: this dataset is a demonstration of a crop with phases, not a validated orchid model. The energy
side uses the heat balance fitted on real greenhouses, on real KNMI weather, so a batch cooled in summer costs
electricity and one raised in winter costs heat.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import knmi
import orchid_crop as crop

STATION = 344
START, END = "2015-08-01", "2025-12-31"
MAX_DAYS = 62 * 7
LEVERS = ["weeks_phase_1", "weeks_phase_2", "temp_warm_c", "weeks_cooling", "temp_cooling_c"]
OUTPUTS = ["spikes_per_plant", "sellable_share", "two_spike_share", "weeks_to_sale", "heat_mj_per_plant",
           "cooling_kwh_per_plant", "space_m2_weeks_per_plant"]
SCENARIOS = {"2 weeks longer warm growth": {"weeks_phase_2": 2.0}, "cooling 2 weeks shorter": {"weeks_cooling": -2.0},
             "cooling 2 degrees warmer": {"temp_cooling_c": 2.0}, "warm phases 1.5 degrees colder": {"temp_warm_c": -1.5}}


def outcome(weather: pd.DataFrame, plan: dict[str, float], light_mol: float) -> dict[str, float]:
    """A batch without batch-to-batch luck: the plant result and what it cost on this weather."""
    batch = crop.grow(light_mol=light_mol, **plan)
    costs = crop.energy(weather["temp_mean_c"].to_numpy(), weather["radiation_j_cm2"].to_numpy(),
                        weeks_finishing=batch.weeks_finishing, **plan)
    return {"spikes_per_plant": batch.spikes_per_plant, "sellable_share": batch.sellable_share,
            "two_spike_share": batch.two_spike_share, "weeks_to_sale": batch.weeks_to_sale, **costs}


def _window(daily: pd.DataFrame, start: pd.Timestamp) -> pd.DataFrame:
    first = int(daily["date"].searchsorted(start))
    return daily.iloc[first: first + MAX_DAYS].reset_index(drop=True)


def _plan(rng: np.random.Generator) -> dict[str, float]:
    """Growers stay near the usual schedule and try things now and then."""
    return {"weeks_phase_1": float(np.clip(round(rng.normal(16, 2)), 10, 20)),
            "weeks_phase_2": float(np.clip(round(rng.normal(12, 2.5)), 8, 18)),
            "temp_warm_c": float(np.clip(round(rng.normal(28, 0.9) * 2) / 2, 26, 31)),
            "weeks_cooling": float(np.clip(round(rng.normal(7.5, 1.5)), 4, 10)),
            "temp_cooling_c": float(np.clip(round(rng.normal(19.5, 1.3) * 2) / 2, 17, 23))}


def generate(daily: pd.DataFrame, batches: int = 5000, seed: int = 21) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    daily = daily.dropna(subset=["temp_mean_c", "radiation_j_cm2"]).reset_index(drop=True)
    doy = daily["date"].dt.day_of_year.clip(upper=365)
    normal = daily.groupby(doy)[["temp_mean_c", "radiation_j_cm2"]].mean()
    first, last = daily["date"].min(), daily["date"].max() - pd.Timedelta(days=MAX_DAYS)
    rows = []
    for i in range(batches):
        start = first + pd.Timedelta(days=int(rng.integers(0, (last - first).days)))
        plan, light = _plan(rng), round(float(rng.uniform(3.0, 6.0)), 1)
        weather = _window(daily, start)
        result = outcome(weather, plan, light)
        # Batch-to-batch luck: cultivar, plant material, a week more or less.
        result["spikes_per_plant"] = max(0.0, result["spikes_per_plant"] * rng.normal(1.0, 0.04))
        result["two_spike_share"] = float(np.clip(result["two_spike_share"] + rng.normal(0, 0.03), 0, 1))
        result["sellable_share"] = float(np.clip(result["sellable_share"] - abs(rng.normal(0, 0.01)), 0, 1))
        result["weeks_to_sale"] += rng.normal(0, 0.5)
        days = weather["date"].dt.day_of_year.clip(upper=365).to_numpy()
        average_year = pd.DataFrame({"temp_mean_c": normal["temp_mean_c"].reindex(days).to_numpy(),
                                     "radiation_j_cm2": normal["radiation_j_cm2"].reindex(days).to_numpy()})
        usual = outcome(average_year, dict(crop.STANDARD), light)          # the usual schedule in an average year
        rows.append({
            "batch_id": f"B{i:05d}", "start_date": start.strftime("%Y-%m-%d"), "start_week": int(start.isocalendar().week),
            "light_mol": light, **plan,
            # A plant is paid by class: a price per sellable plant, and a premium for two or more spikes.
            "price_plant_eur": round(float(rng.normal(2.6, 0.25)), 2),
            "two_spike_premium_eur": round(float(rng.normal(1.3, 0.2)), 2),
            "heat_price_eur_mj": round(float(np.clip(np.exp(rng.normal(np.log(0.012), 0.45)), 0.005, 0.04)), 4),
            "electricity_price_eur_kwh": round(float(np.clip(np.exp(rng.normal(np.log(0.10), 0.4)), 0.04, 0.30)), 3),
            # What a square metre costs per week: building, staff, and the next batch that has to wait.
            # Without it the longest schedule always looks best per plant.
            "space_cost_eur_m2_week": round(float(rng.uniform(0.8, 1.8)), 2),
            **{f"usual_{name}": round(usual[name], 3) for name in ("spikes_per_plant", "weeks_to_sale", "heat_mj_per_plant", "cooling_kwh_per_plant")},
            **{name: round(result[name], 3) for name in OUTPUTS},
        })
    return pd.DataFrame(rows).sort_values(["start_date", "batch_id"]).reset_index(drop=True)


def counterfactual(row: pd.Series, daily: pd.DataFrame, **plan: float) -> dict[str, float]:
    """The answer key: the same batch on the same weather with another schedule (lever values, not changes)."""
    return outcome(_window(daily, pd.Timestamp(row["start_date"])), {name: float(plan.get(name, row[name])) for name in LEVERS},
                   float(row["light_mol"]))


def net_eur(row: pd.Series, result: dict[str, float]) -> float:
    """Income minus energy and space, per plant, at the grower's prices."""
    return (result["sellable_share"] * float(row["price_plant_eur"]) + result["two_spike_share"] * float(row["two_spike_premium_eur"])
            - result["heat_mj_per_plant"] * float(row["heat_price_eur_mj"])
            - result["cooling_kwh_per_plant"] * float(row["electricity_price_eur_kwh"])
            - result["space_m2_weeks_per_plant"] * float(row["space_cost_eur_m2_week"]))


def _client(use_case: str):
    import logging
    import warnings

    from fastapi.testclient import TestClient

    from predictor import bootstrap

    warnings.filterwarnings("ignore")
    logging.disable(logging.INFO)
    client = TestClient(bootstrap.build_app(Path("usecases"), Path("models")))
    return client, client.get(f"/models/{use_case}").json()


def _test_rows(data: Path, rows: int, test_from: str, seed: int) -> pd.DataFrame:
    df = pd.read_csv(data)
    test = df[df["start_date"] >= test_from]
    return test.sample(min(rows, len(test)), random_state=seed)


def check(data: Path, daily: pd.DataFrame, use_case: str = "orchid-batch", rows: int = 300, test_from: str = "2023-01-01") -> pd.DataFrame:
    """Scenarios through the real engine against the crop model, on batches from the test period."""
    client, model = _client(use_case)
    order = model["feature_schema"]["x-feature-order"]
    limits = {lever["name"]: (lever["min"], lever["max"]) for lever in model["scenario"]["levers"]}
    daily = daily.dropna(subset=["temp_mean_c", "radiation_j_cm2"]).reset_index(drop=True)
    records = []
    for _, row in _test_rows(data, rows, test_from, 1).iterrows():
        features = {k: float(row[k]) for k in order}
        base_true = counterfactual(row, daily)
        asked = [{"name": name, "changes": {k: features[k] + v for k, v in change.items()}} for name, change in SCENARIOS.items()
                 if all(limits[k][0] <= features[k] + v <= limits[k][1] for k, v in change.items())]
        body = client.post(f"/models/{use_case}/scenario", json={"base": {"features": features}, "scenarios": asked}).json()
        for scenario in body.get("scenarios", []):
            truth = counterfactual(row, daily, **{k: features[k] + v for k, v in SCENARIOS[scenario["name"]].items()})
            for output in OUTPUTS:
                records.append({"scenario": scenario["name"], "output": output, "true": truth[output] - base_true[output],
                                "predicted": scenario["difference"][output]["change"]})
            records.append({"scenario": scenario["name"], "output": "net result (EUR per plant)",
                            "true": net_eur(row, truth) - net_eur(row, base_true), "predicted": scenario["net_eur"]})
    r = pd.DataFrame(records)
    r["same_direction"] = np.sign(r["true"].round(3)) == np.sign(r["predicted"].round(3))
    grouped = r.groupby(["scenario", "output"], sort=False)
    return pd.DataFrame({"cases": grouped.size(), "true_change": grouped["true"].mean(), "predicted_change": grouped["predicted"].mean(),
                         "direction_right_pct": grouped["same_direction"].mean() * 100,
                         "per_case_corr": grouped.apply(lambda g: g["true"].corr(g["predicted"]))}).round(3).reset_index()


def check_finder(data: Path, daily: pd.DataFrame, use_case: str = "orchid-batch", rows: int = 80, test_from: str = "2023-01-01") -> pd.DataFrame:
    """The finder's advice against the crop model's true result of every plan it could have chosen."""
    from predictor.core import plan as planning
    from predictor.core.manifest import MANIFEST_FILENAME, load_manifest

    client, model = _client(use_case)
    order = model["feature_schema"]["x-feature-order"]
    plans = planning.grid(load_manifest(Path("usecases") / use_case / MANIFEST_FILENAME).scenario, LEVERS)
    daily = daily.dropna(subset=["temp_mean_c", "radiation_j_cm2"]).reset_index(drop=True)
    records = []
    for _, row in _test_rows(data, rows, test_from, 2).iterrows():
        keep = net_eur(row, counterfactual(row, daily))
        record = {"best_possible": max(net_eur(row, counterfactual(row, daily, **plan)) for plan in plans) - keep}
        features = {k: float(row[k]) for k in order}
        for name, options in [("advice", {}), ("any_gain", {"min_gain": 0})]:
            body = client.post(f"/models/{use_case}/scenario/best", json={"base": {"features": features}, **options}).json()
            plan = body["best"]["plan"] if body.get("best") else None
            record[name] = net_eur(row, counterfactual(row, daily, **plan)) - keep if plan else 0.0
            record[f"{name}_promised"] = body["best"]["net_eur"] if plan else 0.0
            record[f"{name}_given"] = plan is not None
        records.append(record)
    r = pd.DataFrame(records)
    summary = {}
    for name, label in [("advice", "the finder's advice (default)"), ("any_gain", "advising every gain, however small")]:
        given = r[f"{name}_given"]
        summary[label] = {"advice given %": given.mean() * 100, "promised gain": r[f"{name}_promised"].mean(), "true gain": r[name].mean(),
                          "share of the best possible %": r[name].sum() / r["best_possible"].sum() * 100,
                          "made it worse % (of advice given)": (r.loc[given, name] < -0.01).mean() * 100, "worst case": r[name].min()}
    summary["the best plan in hindsight"] = {"advice given %": 100.0, "promised gain": np.nan, "true gain": r["best_possible"].mean(),
                                             "share of the best possible %": 100.0, "made it worse % (of advice given)": 0.0,
                                             "worst case": r["best_possible"].min()}
    return pd.DataFrame(summary).T.round(3)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("data/orchid_batches.csv"))
    parser.add_argument("--batches", type=int, default=5000)
    parser.add_argument("--weather", choices=["knmi", "synthetic"], default="knmi")
    parser.add_argument("--check", action="store_true", help="compare the trained model's scenarios with the crop model")
    parser.add_argument("--check-finder", action="store_true", help="does following the best-plan finder make money?")
    args = parser.parse_args()
    daily = knmi.load_daily(STATION, START, END) if args.weather == "knmi" else knmi.synthetic_daily(START, END)
    if args.check:
        print(check(args.out, daily).to_string(index=False))
        return
    if args.check_finder:
        print("net result per plant, compared with keeping the batch's own schedule (EUR):")
        print(check_finder(args.out, daily).to_string())
        return
    df = generate(daily, args.batches)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"wrote {args.out}: {len(df)} batches, {df['start_date'].min()} .. {df['start_date'].max()}; "
          f"{df['spikes_per_plant'].mean():.2f} spikes per plant in {df['weeks_to_sale'].mean():.0f} weeks on average")


if __name__ == "__main__":
    main()
