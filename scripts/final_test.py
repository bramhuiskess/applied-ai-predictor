"""The final test: real greenhouses of another year that nothing in this project was tuned on.

    python scripts/final_test.py            # build data/greenhouse_final_test.csv and score the published models

Every other number in this project comes from data we looked at while building. This one does not:
the 4th Autonomous Greenhouse Challenge (Wageningen UR, 2024, CC BY 4.0), six compartments growing
dwarf tomato from September to November 2024. Other year, other teams, other crop, other lamps.

The published `greenhouse-heating` and `greenhouse-electricity` models (trained on the 2018 and
2019-2020 editions) predict every greenhouse-day through the real API. For these greenhouses the
models have no history, so this is the hard case: a new greenhouse in a new year.

Decisions fixed before the first run, so the result cannot steer them:

* crop = "tomato" (the models know cucumber and tomato; dwarf tomato is closest);
* rail_min_c = the minimum pipe temperature; grow_min_c = 0 (these compartments have one heating circuit);
* the lamps are dimmable LEDs: lamp_hours = hours at full power (the sum of the dimming level), and
  lamp_power_w_m2 = the power at 100%, taken from the electricity log;
* energy_screen_pct = the mean position of the energy screen, as in the earlier editions;
* days with less than 21 hours of log are left out.

Source: Maree, S. et al. (2025), 4th Autonomous Greenhouse Challenge: Dwarf Tomato Timeseries and
Images, 4TU.ResearchData, https://doi.org/10.4121/fa102772-32db-4b30-bace-12f2016722ce
"""

from __future__ import annotations

import argparse
import hashlib
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

URL = "https://data.4tu.nl/file/fa102772-32db-4b30-bace-12f2016722ce/14a4a8cd-e081-4dbd-94e3-cd7b332ce120"
MD5 = "c4da1b2696c4a1e5ea5744f2482b0f93"
RAW = Path("data/raw/greenhouse_challenge")
TEAMS = {"reference": "Reference", "agrifusion": "Agrifusion", "ideas": "IDEAS", "mugrow": "MuGrow",
         "tomatonuts": "Tomatonuts", "trigger": "Trigger"}
TRAINING_DATA = Path("data/greenhouse_days.csv")   # only for the plain guess "the average day of the training data"
SAMPLE_HOURS = 5 / 60
MIN_SAMPLES = 250
C = "compartment/"


def download(raw: Path = RAW) -> Path:
    extracted = raw / "extracted_2024"
    folder = extracted / "autonomous_greenhouse_challenge4_timeseries" / "timeseries"
    if (folder / "weather.csv").is_file():
        return folder
    raw.mkdir(parents=True, exist_ok=True)
    archive = raw / "edition4_timeseries.zip"
    if not archive.is_file():
        with urllib.request.urlopen(URL, timeout=300) as response:
            archive.write_bytes(response.read())
    if hashlib.md5(archive.read_bytes()).hexdigest() != MD5:
        raise SystemExit(f"{archive}: checksum mismatch, delete it and try again")
    with zipfile.ZipFile(archive) as z:
        z.extractall(extracted)
    return folder


def _read(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    # Local clock time as written (the offset changes with summer time); the day is the local day.
    df["date"] = pd.to_datetime(df["time"].str.slice(0, 19)).dt.floor("D")
    return df


def build(folder: Path) -> pd.DataFrame:
    weather = _read(folder / "weather.csv")
    w = weather.groupby("date").agg(
        outside_temp_c=("weather/air_temperature.outside", "mean"), outside_temp_min_c=("weather/air_temperature.outside", "min"),
        radiation_j_cm2=("weather/radiation_global", lambda s: s.fillna(0).sum() * 300 / 1e4),
        wind_ms=("weather/wind_speed", "mean"), outside_rh_pct=("weather/relative_humidity.outside", "mean"),
        samples=("weather/air_temperature.outside", "count"))
    w = w[w["samples"] >= MIN_SAMPLES].drop(columns="samples")
    frames = []
    for file, team in TEAMS.items():
        t = _read(folder / f"{file}.csv")
        level = t[C + "lamps_activation_percentage"].fillna(0) / 100
        kwh = t["energy/electricity_use.lighting"].where(t["energy/electricity_use.lighting"] > 1e-9, 0.0)
        full_power_w_m2 = float((kwh[level > 0.2] / (level[level > 0.2] * SAMPLE_HOURS)).median() * 1000)
        t = t.assign(level=level, kwh=kwh, mj=t["energy/energy_use.heating"].where(t["energy/energy_use.heating"] > 1e-9, 0.0))
        d = t.groupby("date").agg(
            heat_setpoint_c=(C + "heating_temperature_vip", "mean"), vent_setpoint_c=(C + "ventilation_temperature_lee_side_vip", "mean"),
            lamp_hours=("level", lambda s: s.sum() * SAMPLE_HOURS), energy_screen_pct=(C + "screen_energy/screen_position", "mean"),
            co2_setpoint_ppm=(C + "co2_concentration_vip", "mean"), rail_min_c=(C + "minimum_pipe_temperature_vip", "mean"),
            heat_mj_m2=("mj", "sum"), electricity_kwh_m2=("kwh", "sum"), samples=(C + "heating_temperature_vip", "count"))
        d = d[d["samples"] >= MIN_SAMPLES].drop(columns="samples")
        d["grow_min_c"] = 0.0
        d["lamp_power_w_m2"] = round(full_power_w_m2, 1)
        d["lamp_kwh_m2"] = d["lamp_hours"] * d["lamp_power_w_m2"] / 1000   # as in the training data
        d = d.join(w, how="inner").reset_index()
        d.insert(1, "greenhouse", f"{team} 2024")
        d.insert(2, "crop", "tomato")
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    # The same yardstick as in training: the same greenhouse one week earlier.
    earlier = df[["greenhouse", "date", "heat_mj_m2", "electricity_kwh_m2"]].copy()
    earlier["date"] += pd.Timedelta(days=7)
    df = df.merge(earlier.rename(columns={"heat_mj_m2": "heat_week_ago_mj_m2", "electricity_kwh_m2": "electricity_week_ago_kwh_m2"}),
                  on=["greenhouse", "date"], how="left")
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    for column in df.select_dtypes("float").columns:
        df[column] = df[column].round(3)
    return df.sort_values(["date", "greenhouse"]).reset_index(drop=True)


def score(df: pd.DataFrame, use_case: str, target: str, guess: str) -> tuple[pd.DataFrame, dict]:
    """Every greenhouse-day through the real API: the prediction, its 80% range, and plain guesses."""
    import logging
    import warnings

    from fastapi.testclient import TestClient

    from predictor import bootstrap

    warnings.filterwarnings("ignore")
    logging.disable(logging.INFO)
    client = TestClient(bootstrap.build_app(Path("usecases"), Path("models")))
    model = client.get(f"/models/{use_case}").json()
    order = model["feature_schema"]["x-feature-order"]
    records = [{"features": {k: (row[k] if isinstance(row[k], str) else float(row[k])) for k in order}} for _, row in df.iterrows()]
    body = client.post(f"/models/{use_case}/predict", json={"records": records}).json()
    if "predictions" not in body:
        raise SystemExit(f"{use_case}: the API refused the records: {str(body)[:500]}")
    out = df[["greenhouse", "date", target, guess]].rename(columns={target: "actual", guess: "week_ago"}).copy()
    out["predicted"] = [p["prediction"] for p in body["predictions"]]
    out["low"] = [p.get("range", {}).get("low", np.nan) for p in body["predictions"]]
    out["high"] = [p.get("range", {}).get("high", np.nan) for p in body["predictions"]]
    out["warnings"] = [len(p["warnings"]) for p in body["predictions"]]
    return out, model


def summary(out: pd.DataFrame, training_mean: float) -> pd.DataFrame:
    rows = {}
    out = out.assign(date=pd.to_datetime(out["date"]), average_day=training_mean)
    for name, column in [("model", "predicted"), ("plain guess: the same as a week ago", "week_ago"),
                         ("plain guess: the average day of the training data", "average_day")]:
        per = []
        for _, t in out.dropna(subset=[column]).groupby("greenhouse"):
            week = t.groupby(t["date"].dt.to_period("W")).agg(y=("actual", "sum"), p=(column, "sum"), n=("actual", "size"))
            week = week[week["n"] == 7]
            spread = ((t["actual"] - t["actual"].mean()) ** 2).sum()
            per.append({"mae": (t[column] - t["actual"]).abs().mean(), "r2": 1 - ((t[column] - t["actual"]) ** 2).sum() / spread,
                        "week": (week["p"] - week["y"]).abs().mean() / week["y"].mean() * 100,
                        "season": abs(t[column].sum() / t["actual"].sum() - 1) * 100})
        p = pd.DataFrame(per)
        rows[name] = {"daily error": p["mae"].mean(), "daily R2 (median)": p["r2"].median(), "weekly error % (median)": p["week"].median(),
                      "weekly error % (worst)": p["week"].max(), "season total % (median)": p["season"].median(),
                      "season total % (worst)": p["season"].max()}
    return pd.DataFrame(rows).T


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("data/greenhouse_final_test.csv"))
    parser.add_argument("--build-only", action="store_true", help="write the table, do not score the models")
    args = parser.parse_args()
    df = build(download())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"wrote {args.out}: {len(df)} greenhouse-days, {df['greenhouse'].nunique()} greenhouses, {df['date'].min()} .. {df['date'].max()}")
    if args.build_only:
        return
    for use_case, target, guess in [("greenhouse-heating", "heat_mj_m2", "heat_week_ago_mj_m2"),
                                    ("greenhouse-electricity", "electricity_kwh_m2", "electricity_week_ago_kwh_m2")]:
        out, model = score(df, use_case, target, guess)
        training_mean = float(pd.read_csv(TRAINING_DATA)[target].mean())
        inside = ((out["actual"] >= out["low"]) & (out["actual"] <= out["high"])).mean()
        print(f"\n{use_case} (model v{model.get('model_version')}, {model.get('algorithm')}): "
              f"{inside:.0%} of days inside the 80% range; {out['warnings'].gt(0).mean():.0%} of days outside the training range of an input")
        print(summary(out, training_mean).round(2).to_string())
        out.to_csv(args.out.with_name(f"final_test_{use_case}.csv"), index=False)


if __name__ == "__main__":
    main()
