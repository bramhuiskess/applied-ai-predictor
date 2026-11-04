"""Real greenhouse data: Autonomous Greenhouse Challenge, editions 1 and 2 (Wageningen UR, CC0).

    python scripts/make_greenhouse_challenge.py      # writes data/greenhouse_days.csv and
                                                     # data/greenhouse_harvests.csv

Twelve real high-tech greenhouse compartments in Bleiswijk, two seasons:

* edition 1: cucumbers, August - December 2018
* edition 2: cherry tomatoes, December 2019 - May 2020

Each season one compartment was run by Dutch commercial growers ("Reference"), five by AI
teams. The climate computer logs every 5 minutes; heating energy and lamp electricity are
booked per day.

One row = one compartment on one day: what the grower set (heating and ventilation
temperature, minimum pipe temperatures, lamp hours, energy screen, CO2), the weather outside,
and what it cost (heating energy in MJ/m2, lamp electricity in kWh/m2). Unlike the nursery
simulator this is real data, so there is no answer key: the effects are checked against
physics instead (see docs/datasets.md, section 4).

Notes on the sources (checked against the raw 5-minute data):
* heating energy is not metered: WUR computes it from the pipe temperatures,
  (rail - air) * 2.1 + (grow pipe - air) * 0.62 W/m2. Edition 1's readme says kWh/m2, but its
  numbers match that formula in MJ/m2 (ratio 1.09), so they are used as MJ/m2.
* lamps: edition 1 has 117 W/m2 (187 umol/m2/s at 1.6 umol/J), edition 2 has 81 W/m2 HPS plus
  up to 61.5 W/m2 LED.
* edition 1's team AiCU logged six day-period setpoints instead of one; their mean is used.

Second table, harvest timing (tomato, edition 2): for every harvest, the trusses' development
time from flowering to harvest (days, 10 sample stems), with the greenhouse climate in the
first HARVEST_WINDOW days after flowering. The window has a fixed length on purpose: averaging
over the whole development time would make a long development reach further back into colder
weeks, and the temperature would partly "predict" itself.

This script is specific to this dataset: it aggregates the 5-minute logs to days and joins
climate, weather and resources. Generic aggregation and joining in the pipeline are separate
tickets; this script can later be replaced by them.

Sources: Hemming, S. et al. (2019), Autonomous Greenhouse Challenge, First Edition (2018),
https://doi.org/10.4121/uuid:e4987a7b-04dd-4c89-9b18-883aad30ba9a and Hemming, S. et al.
(2020), Autonomous Greenhouse Challenge, Second Edition (2019),
https://doi.org/10.4121/uuid:88d22c60-21b3-4ea8-90db-20249a5be2a7 (4TU.ResearchData)
"""

from __future__ import annotations

import argparse
import hashlib
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path("data/raw/greenhouse_challenge")
SAMPLE_HOURS = 5 / 60  # one climate-computer sample covers 5 minutes
MIN_SAMPLES = 250      # a day needs at least ~21 of 24 hours of log

# edition 2 (2019-2020, tomato): a .7z archive
URL = "https://data.4tu.nl/file/8b35675d-7549-49cc-a8c3-dedec8d3f23e/12ed099a-41b1-4a1c-a7be-12b588f2e34a"
MD5 = "2a0c7f3332881caef54ca8f4dc60c9a3"
TEAMS = ["Reference", "AICU", "Automatoes", "Digilog", "IUACAAS", "TheAutomators"]
LAMP_W_M2 = 81 + 61.5

# edition 1 (2018, cucumber): a .zip archive
URL_2018 = "https://data.4tu.nl/file/c362564b-bc9e-40ec-856b-b616bb909183/1f955782-1d4a-4f78-a9cc-d9b17031bdde"
MD5_2018 = "243eaa9041da23d0c4bf99576715aa44"
TEAMS_2018 = {"Reference(Growers)": "Reference", "AiCU": "AiCU", "Croperators": "Croperators",
              "DeepGreens": "DeepGreens", "iGrow": "iGrow", "Sonoma": "Sonoma"}
LAMP_W_M2_2018 = 187 / 1.6


def _fetch(url: str, md5: str, archive: Path) -> None:
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.is_file():
        with urllib.request.urlopen(url, timeout=300) as response:
            archive.write_bytes(response.read())
    if hashlib.md5(archive.read_bytes()).hexdigest() != md5:
        raise SystemExit(f"{archive}: checksum mismatch, delete it and try again")


def download(raw: Path = RAW) -> Path:
    """Download and unpack edition 2 once (needs `pip install py7zr`)."""
    extracted = raw / "extracted"
    if (extracted / "Weather" / "Weather.csv").is_file():
        return extracted
    _fetch(URL, MD5, raw / "edition2.7z")
    try:
        import py7zr
    except ImportError as exc:
        raise SystemExit("this dataset is a .7z archive: pip install py7zr (or: pip install -e '.[datasets]')") from exc
    with py7zr.SevenZipFile(raw / "edition2.7z") as z:
        z.extractall(extracted)
    return extracted


def download_2018(raw: Path = RAW) -> Path:
    """Download and unpack edition 1 once."""
    extracted = raw / "extracted_2018"
    if (extracted / "meteo.csv").is_file():
        return extracted
    _fetch(URL_2018, MD5_2018, raw / "edition1.zip")
    with zipfile.ZipFile(raw / "edition1.zip") as z:
        z.extractall(extracted)
    return extracted


# --------------------------------------------------------------------------- shared

def read_log(path: Path) -> pd.DataFrame:
    """Read a climate-computer CSV: Excel day numbers as time, 'NaN' strings, padded headers."""
    df = pd.read_csv(path, skipinitialspace=True, na_values=["NaN", "nan", ""], low_memory=False)
    df.columns = [c.strip().lstrip("%").strip().lower() for c in df.columns]
    df = df.apply(pd.to_numeric, errors="coerce")
    if "time" in df.columns:
        df["date"] = excel_date(df["time"])
    return df


def excel_date(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, unit="D", origin="1899-12-30").dt.floor("D")


def daily_weather(weather: pd.DataFrame) -> pd.DataFrame:
    w = weather.groupby("date").agg(
        outside_temp_c=("tout", "mean"),
        outside_temp_min_c=("tout", "min"),
        radiation_j_cm2=("iglob", lambda s: s.fillna(0).sum() * 300 / 1e4),  # W/m2 per 5 min -> J/cm2 per day
        wind_ms=("windsp", "mean"),
        outside_rh_pct=("rhout", "mean"),
        samples=("tout", "count"),
    )
    return w[w["samples"] >= MIN_SAMPLES].drop(columns="samples")


def _daily_settings(c: pd.DataFrame) -> pd.DataFrame:
    """`c` has per 5 minutes: heat, vent, lamps_on (0/1), screen, co2, rail_min, grow_min."""
    d = c.groupby("date").agg(
        heat_setpoint_c=("heat", "mean"),
        vent_setpoint_c=("vent", "mean"),
        lamp_hours=("lamps_on", lambda s: s.sum() * SAMPLE_HOURS),
        energy_screen_pct=("screen", "mean"),
        co2_setpoint_ppm=("co2", "mean"),
        # Minimum pipe temperatures: professional growers use them to heat regardless of the setpoint.
        rail_min_c=("rail_min", "mean"),
        grow_min_c=("grow_min", "mean"),
        samples=("heat", "count"),
    )
    d["lamp_hours"] = d["lamp_hours"].clip(upper=24)
    return d[d["samples"] >= MIN_SAMPLES].drop(columns="samples")


# --------------------------------------------------------------------------- edition 2

def daily_climate(climate: pd.DataFrame) -> pd.DataFrame:
    return _daily_settings(pd.DataFrame({
        "date": climate["date"], "heat": climate["t_heat_vip"], "vent": climate["t_ventlee_vip"],
        "lamps_on": (climate["tot_par_lamps"].fillna(0) > 0).astype(float), "screen": climate["enscr"],
        "co2": climate["co2_vip"], "rail_min": climate["t_rail_min_vip"], "grow_min": climate["t_grow_min_vip"]}))


def daily_resources(resources: pd.DataFrame) -> pd.DataFrame:
    r = resources.groupby("date").agg(heat_mj_m2=("heat_cons", "sum"), elec_high=("elechigh", "sum"),
                                      elec_low=("eleclow", "sum"))
    r["electricity_kwh_m2"] = r["elec_high"] + r["elec_low"]
    return r


def build_second(extracted: Path, teams: list[str] = TEAMS) -> tuple[pd.DataFrame, dict]:
    weather = daily_weather(read_log(extracted / "Weather" / "Weather.csv"))
    frames, peak, total = [], 0.0, 0.0
    for team in teams:
        climate = daily_climate(read_log(extracted / team / "GreenhouseClimate.csv"))
        resources = daily_resources(read_log(extracted / team / "Resources.csv"))
        peak += resources["elec_high"].sum()
        total += resources["electricity_kwh_m2"].sum()
        day = climate.join(weather, how="inner").join(resources[["heat_mj_m2", "electricity_kwh_m2"]], how="inner")
        day.insert(0, "greenhouse", f"{team} 2020")
        frames.append(day.reset_index())
    df = pd.concat(frames, ignore_index=True).assign(crop="tomato", lamp_power_w_m2=LAMP_W_M2)
    # Average electricity price for the lamps, from the challenge's own tariffs and the peak share.
    info = {"peak_share": peak / total if total else np.nan,
            "eur_per_kwh": 0.08 * peak / total + 0.04 * (1 - peak / total) if total else np.nan}
    return df, info


# --------------------------------------------------------------------------- edition 1

def _setpoint(vip: pd.DataFrame, name: str) -> pd.Series:
    """One setpoint column, or the mean of the day-period columns (name_1 .. name_6) if a team used those."""
    if name in vip.columns:
        return vip[name]
    periods = [c for c in vip.columns if c.startswith(name + "_")]
    return vip[periods].mean(axis=1) if periods else pd.Series(np.nan, index=vip.index)


def build_first(extracted: Path, teams: dict[str, str] = TEAMS_2018) -> pd.DataFrame:
    # The weather file has dates without a time of day, but its rows line up with the climate logs.
    meteo = read_log(extracted / "meteo.csv").drop(columns=["time", "date"], errors="ignore")
    frames = []
    for folder, team in teams.items():
        climate = read_log(extracted / folder / "Greenhouse_climate.csv")
        vip = read_log(extracted / folder / "vip.csv")
        date = excel_date(climate["ghtime"])
        n = min(len(date), len(meteo))
        weather = daily_weather(meteo.iloc[:n].assign(date=date.to_numpy()[:n]))
        settings = _daily_settings(pd.DataFrame({
            "date": date, "heat": _setpoint(vip, "heattemp_vip"), "vent": _setpoint(vip, "ventleew_vip"),
            "lamps_on": (climate["assimlight"].fillna(0) > 50).astype(float), "screen": climate["enscr"],
            "co2": _setpoint(vip, "co2_vip"), "rail_min": _setpoint(vip, "minpipelow_vip"),
            "grow_min": _setpoint(vip, "minpipegrow_vip")}))
        res = read_log(extracted / folder / "ResourceCalculations.csv")
        res = res.groupby("date").agg(heat_mj_m2=("heating_energy", "sum"), electricity_kwh_m2=("electricity_lamp", "sum"))
        day = settings.join(weather, how="inner").join(res, how="inner")
        day.insert(0, "greenhouse", f"{team} 2018")
        frames.append(day.reset_index())
    return pd.concat(frames, ignore_index=True).assign(crop="cucumber", lamp_power_w_m2=LAMP_W_M2_2018)


# --------------------------------------------------------------------------- harvest timing (edition 2)

HARVEST_WINDOW = 39   # days after flowering; the shortest development time in the data


def harvest_timing(extracted: Path, teams: list[str] = TEAMS) -> pd.DataFrame:
    rows = []
    for team in teams:
        climate = read_log(extracted / team / "GreenhouseClimate.csv")
        day = climate.groupby("date").agg(
            air_temp_c=("tair", "mean"),                                          # 24-hour mean temperature
            light_mol_m2_day=("tot_par", lambda s: s.fillna(0).sum() * 300 / 1e6),  # sun + lamps, inside
            co2_ppm=("co2air", "mean"),
            humidity_pct=("rhair", "mean"),
            samples=("tair", "count"))
        day = day[day["samples"] >= MIN_SAMPLES]
        production = read_log(extracted / team / "Production.csv").dropna(subset=["truss development time"])
        for _, harvest in production[production["date"] >= day.index.min()].iterrows():
            days = float(harvest["truss development time"])
            flowering = harvest["date"] - pd.Timedelta(days=int(days))
            window = day.loc[flowering: flowering + pd.Timedelta(days=HARVEST_WINDOW - 1)]
            if len(window) < 0.8 * HARVEST_WINDOW:
                continue
            rows.append({"flowering_date": flowering.strftime("%Y-%m-%d"), "greenhouse": f"{team} 2020",
                         "air_temp_c": window["air_temp_c"].mean(), "light_mol_m2_day": window["light_mol_m2_day"].mean(),
                         "co2_ppm": window["co2_ppm"].mean(), "humidity_pct": window["humidity_pct"].mean(),
                         "days_to_harvest": days})
    df = pd.DataFrame(rows).round(3)
    return df.sort_values(["flowering_date", "greenhouse"]).reset_index(drop=True)


# --------------------------------------------------------------------------- both

def tidy(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["heat_setpoint_c", "outside_temp_c", "heat_mj_m2", "electricity_kwh_m2"]).copy()
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    for col in df.select_dtypes("float").columns:
        df[col] = df[col].round(3)
    first = ["date", "greenhouse", "crop"]
    return df[first + [c for c in df.columns if c not in first]].sort_values(["date", "greenhouse"]).reset_index(drop=True)


def build(extracted: Path, extracted_2018: Path | None) -> tuple[pd.DataFrame, dict]:
    second, info = build_second(extracted)
    frames = [build_first(extracted_2018), second] if extracted_2018 else [second]
    return tidy(pd.concat(frames, ignore_index=True)), info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("data/greenhouse_days.csv"))
    parser.add_argument("--harvests-out", type=Path, default=Path("data/greenhouse_harvests.csv"))
    parser.add_argument("--only-2020", action="store_true", help="skip edition 1 (2018, cucumber)")
    args = parser.parse_args()
    extracted = download()
    df, info = build(extracted, None if args.only_2020 else download_2018())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    harvests = harvest_timing(extracted)
    harvests.to_csv(args.harvests_out, index=False)
    print(f"wrote {args.harvests_out}: {len(harvests)} tomato harvests with the climate after flowering")
    print(f"wrote {args.out}: {len(df)} greenhouse-days, {df['greenhouse'].nunique()} greenhouses, "
          f"{df['date'].min()} .. {df['date'].max()}; lamp electricity {info['peak_share']:.0%} at peak "
          f"-> {info['eur_per_kwh']:.3f} EUR/kWh on average")


if __name__ == "__main__":
    main()
