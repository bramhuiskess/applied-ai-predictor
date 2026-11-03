"""Real arable dataset: CBS crop yields per province + KNMI weather per growing season.

    python scripts/make_arable.py            # writes data/arable.csv

One row = one crop in one province in one year (1999-2025). The target is the
yield in tonnes per hectare; the features describe the weather during that
crop's growing season plus the average yield of the five years before (the
"normal" level of that crop in that province, which carries the long-term
trend without leaking the current year).

Sources (both open data, CC BY 4.0):
* CBS StatLine 85636NED "Akkerbouwgewassen; productie, regio"
* KNMI daily station data ("etmaalgegevens"), one station per province
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import knmi  # noqa: E402

CBS_TABLE = "85636NED"
CBS_URL = "https://opendata.cbs.nl/ODataApi/OData/{table}/TypedDataSet?{query}"
CBS_CACHE = Path("data/raw/cbs_85636NED.json")

# CBS crop code -> (name, first month, last month of the growing season)
CROPS = {
    "A042171": ("winter_wheat", 3, 7),
    "A042356": ("ware_potatoes", 4, 9),
    "A042194": ("sugar_beet", 4, 10),
    "A042320": ("onions", 3, 8),
    "A042168": ("silage_maize", 5, 9),
}
PROVINCES = {
    "PV20": "Groningen", "PV21": "Fryslân", "PV22": "Drenthe", "PV23": "Overijssel", "PV24": "Flevoland",
    "PV25": "Gelderland", "PV26": "Utrecht", "PV27": "Noord-Holland", "PV28": "Zuid-Holland",
    "PV29": "Zeeland", "PV30": "Noord-Brabant", "PV31": "Limburg",
}
HISTORY_YEARS = 5
FIRST_YEAR, LAST_YEAR = 1999, 2025  # CBS starts in 1994; the first 5 years only feed the history


def fetch_cbs(cache: Path = CBS_CACHE) -> list[dict]:
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    rows: list[dict] = []
    for code in CROPS:
        query = urllib.parse.urlencode({
            "$filter": f"Gewassen eq '{code}' and startswith(RegioS,'PV')",
            "$select": "Gewassen,RegioS,Perioden,GeoogsteOppervlakte_2,BrutoOpbrengstPerHa_3",
        }, quote_via=urllib.parse.quote)
        with urllib.request.urlopen(CBS_URL.format(table=CBS_TABLE, query=query), timeout=120) as response:
            rows += json.load(response)["value"]
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(rows), encoding="utf-8")
    return rows


def yields_frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df = df[df["Perioden"].str.endswith("JJ00")]
    return pd.DataFrame({
        "year": df["Perioden"].str[:4].astype(int),
        "province": df["RegioS"].str.strip().map(PROVINCES),
        "crop": df["Gewassen"].str.strip().map(lambda c: CROPS[c][0]),
        "harvested_ha": pd.to_numeric(df["GeoogsteOppervlakte_2"], errors="coerce"),
        "yield_t_ha": pd.to_numeric(df["BrutoOpbrengstPerHa_3"], errors="coerce"),
    }).dropna(subset=["yield_t_ha"]).reset_index(drop=True)


def season_weather(daily: pd.DataFrame, year: int, first_month: int, last_month: int) -> dict[str, float]:
    """Agronomic weather summary of one growing season."""
    d = daily[(daily["date"].dt.year == year) & daily["date"].dt.month.between(first_month, last_month)]
    if len(d) < 25 * (last_month - first_month + 1):
        return {}
    start = d[d["date"].dt.month <= first_month + 1]  # planting / early growth
    # Precipitation deficit as KNMI uses it: cumulative (evaporation - rain), never below zero.
    deficit, peak = 0.0, 0.0
    for evap, rain in zip(d["evaporation_mm"].fillna(0), d["rain_mm"].fillna(0)):
        deficit = max(0.0, deficit + evap - rain)
        peak = max(peak, deficit)
    return {
        "season_temp_c": round(d["temp_mean_c"].mean(), 2),
        "season_rain_mm": round(d["rain_mm"].sum(), 1),
        "season_radiation_mj_m2": round(d["radiation_j_cm2"].sum() / 100, 1),
        "max_rain_deficit_mm": round(peak, 1),
        "hot_days": int((d["temp_max_c"] >= 30).sum()),
        "frost_days": int((d["temp_min_c"] < 0).sum()),
        "early_season_rain_mm": round(start["rain_mm"].sum(), 1),
        "wet_days": int((d["rain_mm"] >= 10).sum()),
    }


def build(yields: pd.DataFrame, weather: dict[str, pd.DataFrame]) -> pd.DataFrame:
    yields = yields.sort_values(["crop", "province", "year"])
    # Average of the previous five years: the local "normal" (shifted, so the current year never leaks).
    yields["yield_5y_avg"] = (
        yields.groupby(["crop", "province"])["yield_t_ha"]
        .transform(lambda s: s.shift(1).rolling(HISTORY_YEARS, min_periods=HISTORY_YEARS).mean())
        .round(2)
    )
    rows = []
    for row in yields.itertuples(index=False):
        if not FIRST_YEAR <= row.year <= LAST_YEAR or pd.isna(row.yield_5y_avg):
            continue
        _, first, last = next(v for v in CROPS.values() if v[0] == row.crop)
        summary = season_weather(weather[row.province], row.year, first, last)
        if not summary:
            continue
        rows.append({"season": f"{row.year}-01-01", "year": row.year, "province": row.province,
                     "crop": row.crop, "harvested_ha": row.harvested_ha, "yield_5y_avg": row.yield_5y_avg,
                     **summary, "yield_t_ha": row.yield_t_ha})
    return pd.DataFrame(rows).sort_values(["season", "crop", "province"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("data/arable.csv"))
    args = parser.parse_args()
    yields = yields_frame(fetch_cbs())
    weather = {province: knmi.load_daily(station, "1994-01-01", f"{LAST_YEAR}-12-31")
               for province, station in knmi.PROVINCE_STATIONS.items()}
    df = build(yields, weather)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"wrote {args.out}: {len(df)} rows, {df['year'].min()}-{df['year'].max()}, "
          f"{df['crop'].nunique()} crops x {df['province'].nunique()} provinces")


if __name__ == "__main__":
    main()
