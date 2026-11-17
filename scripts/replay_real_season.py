"""Replay a real season through the product: what would it have told the growers, and what really happened?

    python scripts/replay_real_season.py                          # against the models in ./models
    python scripts/replay_real_season.py --url http://127.0.0.1:8098   # against a running server

The six real tomato compartments of the Autonomous Greenhouse Challenge (Dec 2019 - May 2020). For each one we
stand at a decision day early in the season, give `tomato-season` only what a grower knew that day (the
settings so far, the plan he then really followed, his heating use of the last weeks, his prices) and compare
the forecast with the harvest, heating and electricity that were really measured afterwards.

This is the hardest honest test we can do without a grower's own records: the forecasting model never saw
these greenhouses. It was trained on simulated seasons; the crop model and heat balance behind that simulator
were fitted on these six compartments (two and four numbers), which is the only way they are connected.

The harvest is compared without the trial's final clean-up harvest (unripe trusses), which the product does not count.

What is approximated, because the data starts at planting:
* at decision day 14 the "heating use of the last four weeks" is the first two weeks doubled;
* the model's lamp input is "hours on dark days"; the real lamp hours are converted with the real radiation.
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

import greenhouse_heat as heating
import knmi
import make_greenhouse_challenge as agc
import make_tomato_season as season

USE_CASE = "tomato-season"
PRICES = {"tomato_price_eur_kg": 3.0, "heat_price_eur_mj": 0.0083, "electricity_price_eur_kwh": 0.065}   # the challenge's own
HORIZONS = {"harvest_1_month_kg_m2": 30, "harvest_3_months_kg_m2": 91, "harvest_5_months_kg_m2": 152,
            "heating_5_months_mj_m2": 152, "electricity_5_months_kwh_m2": 152}


class Api:
    """The product, either a running server or the app in this process."""

    def __init__(self, url: str | None) -> None:
        self.url = url.rstrip("/") if url else None
        if not self.url:
            import logging
            import warnings

            from fastapi.testclient import TestClient

            from predictor import bootstrap

            warnings.filterwarnings("ignore")
            logging.disable(logging.INFO)
            self.client = TestClient(bootstrap.build_app(Path("usecases"), Path("models")))

    def get(self, path: str) -> dict:
        if self.url:
            with urllib.request.urlopen(self.url + path, timeout=60) as r:
                return json.load(r)
        return self.client.get(path).json()

    def post(self, path: str, body: dict) -> dict:
        if self.url:
            request = urllib.request.Request(self.url + path, json.dumps(body).encode(), {"content-type": "application/json"})
            with urllib.request.urlopen(request, timeout=120) as r:
                return json.load(r)
        return self.client.post(path, json=body).json()


def dark_factor(radiation: np.ndarray) -> np.ndarray:
    return season.lamp_hours(radiation, 1.0)


def what_the_grower_knew(days: pd.DataFrame, decision_day: int, usual: dict[str, float], decision: pd.Timestamp,
                         sun_last_4_weeks: float | None = None) -> dict[str, float]:
    """The model's inputs on the decision day, from the real log of one compartment."""
    before, after = days.iloc[:decision_day], days.iloc[decision_day: decision_day + 152]
    if decision_day == 0:
        before = days.iloc[:14]                       # no log before planting: the first two weeks stand in

    def lamp_setting(part: pd.DataFrame) -> float:
        # real lamp hours -> "hours on dark days" (the simulator switches lamps off as the sun takes over)
        dark = dark_factor(part["radiation"].to_numpy()).mean()
        return float(np.clip(part["lamp_hours"].mean() / max(dark, 0.05), 0, 18))

    history = days.iloc[max(0, decision_day - 28): decision_day] if decision_day else days.iloc[:14]
    heat_last_4_weeks = float(history["heat"].sum() * 28 / len(history))
    return {"decision_week": int(decision.isocalendar().week), "days_since_planting": decision_day,
            "temp_so_far_c": round(float(before["temp"].mean()), 2), "lamp_hours_so_far": round(lamp_setting(before), 1),
            "heat_last_4_weeks_mj_m2": round(heat_last_4_weeks, 2),
            "sun_last_4_weeks_j_cm2": round(float(sun_last_4_weeks if sun_last_4_weeks is not None else history["radiation"].mean()), 1),
            "harvest_so_far_kg_m2": round(float(days["kg"].iloc[:decision_day].sum()), 3),
            # "sun_ahead_ratio" is left out on purpose: nobody knows the coming months, so the product assumes a
            # normal year and widens its range to a dark and a sunny one
            "temp_plan_c": round(float(np.clip(after["temp"].mean(), 19.5, 24.5)), 2), "lamp_hours_plan": round(lamp_setting(after), 1),
            **PRICES, **{f"usual_{k}": round(v, 3) for k, v in usual.items() if k in season.OUTPUTS}}


def replay(api: Api, decision_days: tuple[int, ...] = (14, 28, 56, 84)) -> pd.DataFrame:
    real = heating.load_real(agc.download())
    order = api.get(f"/models/{USE_CASE}")["feature_schema"]["x-feature-order"]
    daily = knmi.load_daily(season.STATION, season.START, season.END).dropna(subset=["temp_mean_c", "radiation_j_cm2"]).reset_index(drop=True)
    normal = season.climatology(daily)
    rows = []
    for greenhouse, days in real.items():
        days = days.rename(columns={"lamp_kwh": "electricity"})
        planting = days.index[0]
        # The last harvest of the trial also took the unripe trusses; the product forecasts ripe harvest.
        clean_up = int(np.flatnonzero(days["kg"].to_numpy())[-1])
        window = season._window(daily, planting)
        doy = window["date"].dt.day_of_year.clip(upper=365).to_numpy()
        average_year = pd.DataFrame({"temp_mean_c": normal["temp_mean_c"].reindex(doy).to_numpy(),
                                     "radiation_j_cm2": normal["radiation_j_cm2"].reindex(doy).to_numpy()})
        for decision_day in decision_days:
            usual = season.season(average_year, decision_day, (season.NORMAL_TEMP_C, season.NORMAL_LAMP_HOURS),
                                  (season.NORMAL_TEMP_C, season.NORMAL_LAMP_HOURS))
            radiation = window["radiation_j_cm2"].to_numpy()
            start = decision_day + season.HISTORY_DAYS                 # the weather window starts four weeks before planting
            features = what_the_grower_knew(days, decision_day, usual, planting + pd.Timedelta(days=decision_day),
                                            sun_last_4_weeks=radiation[start - season.HISTORY_DAYS: start].mean())
            # For the analysis only: the same forecast if the sun of the coming months had been known.
            hindsight = dict(features, sun_ahead_ratio=round(float(radiation[start: start + 152].mean() / usual["sun_ahead_j_cm2"]), 3))
            body = api.post(f"/models/{USE_CASE}/predict", {"records": [{"features": {k: row[k] for k in order if k in row}}
                                                                        for row in (features, hindsight)]})
            outputs, with_sun = body["predictions"][0]["outputs"], body["predictions"][1]["outputs"]
            for output, horizon in HORIZONS.items():
                if decision_day + horizon > len(days):          # the season ended before this horizon
                    continue
                column = "kg" if output.startswith("harvest") else "heat" if output.startswith("heating") else "electricity"
                actual = float(days[column].iloc[decision_day: decision_day + horizon].sum())
                if column == "kg" and decision_day + horizon > clean_up:
                    actual -= float(days["kg"].iloc[clean_up])       # ripe trusses only, as the product counts them
                rng = outputs[output].get("range")
                rows.append({"greenhouse": greenhouse, "decision_day": decision_day, "output": output, "actual": actual,
                             "forecast": outputs[output]["prediction"], "plain_guess": usual[output],
                             "forecast_knowing_the_sun": with_sun[output]["prediction"],
                             "inside_range": bool(rng and rng["low"] <= actual <= rng["high"]) if rng else np.nan})
    return pd.DataFrame(rows)


def summary(r: pd.DataFrame) -> pd.DataFrame:
    """Per output: the error as a share of what was really measured (also right when a month had no harvest yet)."""
    rows = {}
    for output, g in r.groupby("output", sort=False):
        total = g["actual"].sum()
        rows[output] = {"cases": len(g), "actual": g["actual"].mean(), "forecast": g["forecast"].mean(),
                        "error_pct": (g["forecast"] - g["actual"]).abs().sum() / total * 100,
                        "bias_pct": (g["forecast"].sum() / total - 1) * 100,
                        "plain_guess_error_pct": (g["plain_guess"] - g["actual"]).abs().sum() / total * 100,
                        "error_knowing_the_sun_pct": (g["forecast_knowing_the_sun"] - g["actual"]).abs().sum() / total * 100,
                        "inside_80_range_pct": g["inside_range"].mean() * 100}
    return pd.DataFrame(rows).T.round(2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", help="a running server, e.g. http://127.0.0.1:8000 (default: the models in ./models)")
    parser.add_argument("--out", type=Path, help="write every case to this CSV")
    args = parser.parse_args()
    r = replay(Api(args.url))
    print("the real season 2019-2020, six compartments: forecast on the decision day against what was measured afterwards\n")
    print(summary(r).to_string())
    if args.out:
        r.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
