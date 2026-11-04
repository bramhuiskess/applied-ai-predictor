"""Real greenhouse data: Autonomous Greenhouse Challenge, 2nd edition (Wageningen UR, CC0).

    python scripts/make_greenhouse_challenge.py      # writes data/greenhouse_days.csv

Six real high-tech greenhouse compartments in Bleiswijk (one run by Dutch commercial growers,
five by AI teams), a cherry-tomato season from December 2019 to May 2020. The climate computer
logs every 5 minutes; heating energy and lamp electricity are reported per day.

One row = one compartment on one day: what the grower set (heating temperature, ventilation
temperature, minimum pipe temperatures, lamp hours, energy screen, CO2), the weather outside, and what it cost (heating
energy in MJ/m2, lamp electricity in kWh/m2). Unlike the nursery simulator this is real data,
so there is no answer key: the effects are checked against physics instead
(see docs/datasets.md, section 4).

This script is specific to this dataset: it aggregates the 5-minute logs to days and joins
climate, weather and resources. Generic aggregation and joining in the pipeline are separate
tickets; this script can later be replaced by them.

Source: Hemming, S. et al. (2020), Autonomous Greenhouse Challenge, Second Edition (2019),
4TU.ResearchData, https://doi.org/10.4121/uuid:88d22c60-21b3-4ea8-90db-20249a5be2a7
"""

from __future__ import annotations

import argparse
import hashlib
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

URL = "https://data.4tu.nl/file/8b35675d-7549-49cc-a8c3-dedec8d3f23e/12ed099a-41b1-4a1c-a7be-12b588f2e34a"
MD5 = "2a0c7f3332881caef54ca8f4dc60c9a3"
RAW = Path("data/raw/greenhouse_challenge")
TEAMS = ["Reference", "AICU", "Automatoes", "Digilog", "IUACAAS", "TheAutomators"]
SAMPLE_HOURS = 5 / 60  # one climate-computer sample covers 5 minutes


def download(raw: Path = RAW) -> Path:
    """Download and unpack the archive once (needs `pip install py7zr`)."""
    extracted = raw / "extracted"
    if (extracted / "Weather" / "Weather.csv").is_file():
        return extracted
    raw.mkdir(parents=True, exist_ok=True)
    archive = raw / "edition2.7z"
    if not archive.is_file():
        with urllib.request.urlopen(URL, timeout=300) as response:
            archive.write_bytes(response.read())
    if hashlib.md5(archive.read_bytes()).hexdigest() != MD5:
        raise SystemExit(f"{archive}: checksum mismatch, delete it and try again")
    try:
        import py7zr
    except ImportError as exc:
        raise SystemExit("this dataset is a .7z archive: pip install py7zr (or: pip install -e '.[datasets]')") from exc
    with py7zr.SevenZipFile(archive) as z:
        z.extractall(extracted)
    return extracted


def read_log(path: Path) -> pd.DataFrame:
    """Read a climate-computer CSV: Excel day numbers as time, 'NaN' strings, padded headers."""
    df = pd.read_csv(path, skipinitialspace=True, na_values=["NaN", "nan", ""], low_memory=False)
    df.columns = [c.strip().lstrip("%").strip().lower() for c in df.columns]
    df = df.apply(pd.to_numeric, errors="coerce")
    df["date"] = pd.to_datetime(df["time"], unit="D", origin="1899-12-30").dt.floor("D")
    return df


def daily_weather(weather: pd.DataFrame) -> pd.DataFrame:
    w = weather.groupby("date").agg(
        outside_temp_c=("tout", "mean"),
        outside_temp_min_c=("tout", "min"),
        radiation_j_cm2=("iglob", lambda s: s.fillna(0).sum() * 300 / 1e4),  # W/m2 per 5 min -> J/cm2 per day
        wind_ms=("windsp", "mean"),
        outside_rh_pct=("rhout", "mean"),
        samples=("tout", "size"),
    )
    return w[w["samples"] >= 250].drop(columns="samples")  # at least ~21 of 24 hours logged


def daily_climate(climate: pd.DataFrame) -> pd.DataFrame:
    c = climate.copy()
    c["lamps_on"] = (c["tot_par_lamps"].fillna(0) > 0).astype(float)
    d = c.groupby("date").agg(
        heat_setpoint_c=("t_heat_vip", "mean"),
        vent_setpoint_c=("t_ventlee_vip", "mean"),
        lamp_hours=("lamps_on", lambda s: s.sum() * SAMPLE_HOURS),
        energy_screen_pct=("enscr", "mean"),
        co2_setpoint_ppm=("co2_vip", "mean"),
        # Minimum pipe temperatures: professional growers use them to heat regardless of the setpoint.
        rail_min_c=("t_rail_min_vip", "mean"),
        grow_min_c=("t_grow_min_vip", "mean"),
        samples=("t_heat_vip", "size"),
    )
    d["lamp_hours"] = d["lamp_hours"].clip(upper=24)
    return d[d["samples"] >= 250].drop(columns="samples")


def daily_resources(resources: pd.DataFrame) -> pd.DataFrame:
    r = resources.groupby("date").agg(heat_mj_m2=("heat_cons", "sum"), elec_high=("elechigh", "sum"),
                                      elec_low=("eleclow", "sum"))
    r["electricity_kwh_m2"] = r["elec_high"] + r["elec_low"]
    return r


def build(extracted: Path, teams: list[str] = TEAMS) -> tuple[pd.DataFrame, dict]:
    weather = daily_weather(read_log(extracted / "Weather" / "Weather.csv"))
    frames, peak, total = [], 0.0, 0.0
    for team in teams:
        climate = daily_climate(read_log(extracted / team / "GreenhouseClimate.csv"))
        resources = daily_resources(read_log(extracted / team / "Resources.csv"))
        peak += resources["elec_high"].sum()
        total += resources["electricity_kwh_m2"].sum()
        day = climate.join(weather, how="inner").join(resources[["heat_mj_m2", "electricity_kwh_m2"]], how="inner")
        day.insert(0, "greenhouse", team)
        frames.append(day.reset_index())
    df = pd.concat(frames, ignore_index=True)
    start = df["date"].min()
    df.insert(2, "season_day", (df["date"] - start).dt.days)
    df = df.dropna(subset=["heat_setpoint_c", "outside_temp_c", "heat_mj_m2", "electricity_kwh_m2"])
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    for col in df.select_dtypes("float").columns:
        df[col] = df[col].round(3)
    # Average electricity price for the lamps, from the challenge's own tariffs and the peak share.
    info = {"peak_share": peak / total if total else np.nan,
            "eur_per_kwh": 0.08 * peak / total + 0.04 * (1 - peak / total) if total else np.nan}
    return df.sort_values(["date", "greenhouse"]).reset_index(drop=True), info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("data/greenhouse_days.csv"))
    args = parser.parse_args()
    df, info = build(download())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"wrote {args.out}: {len(df)} greenhouse-days, {df['greenhouse'].nunique()} greenhouses, "
          f"{df['date'].min()} .. {df['date'].max()}; lamp electricity {info['peak_share']:.0%} at peak "
          f"-> {info['eur_per_kwh']:.3f} EUR/kWh on average")


if __name__ == "__main__":
    main()
