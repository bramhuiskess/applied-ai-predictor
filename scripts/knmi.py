"""KNMI daily weather per station (open data, CC BY 4.0).

Downloads the classic "etmaalgegevens" file of a station once, caches it in
data/raw/knmi/ and returns tidy daily values in normal units.
"""

from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

URL = "https://cdn.knmi.nl/knmi/map/page/klimatologie/gegevens/daggegevens/etmgeg_{station}.zip"
CACHE = Path("data/raw/knmi")

# One representative station per province, used for the arable dataset.
PROVINCE_STATIONS = {
    "Groningen": 280,      # Eelde
    "Fryslân": 270,        # Leeuwarden
    "Drenthe": 279,        # Hoogeveen
    "Overijssel": 290,     # Twenthe
    "Flevoland": 269,      # Lelystad
    "Gelderland": 275,     # Deelen
    "Utrecht": 260,        # De Bilt
    "Noord-Holland": 235,  # De Kooy
    "Zuid-Holland": 344,   # Rotterdam
    "Zeeland": 310,        # Vlissingen
    "Noord-Brabant": 370,  # Eindhoven
    "Limburg": 380,        # Maastricht
}

# column -> (new name, scale to normal units)
COLUMNS = {
    "TG": ("temp_mean_c", 0.1),
    "TN": ("temp_min_c", 0.1),
    "TX": ("temp_max_c", 0.1),
    "RH": ("rain_mm", 0.1),
    "SQ": ("sun_hours", 0.1),
    "Q": ("radiation_j_cm2", 1.0),
    "EV24": ("evaporation_mm", 0.1),
    "UG": ("humidity_pct", 1.0),
}


def download(station: int, cache: Path = CACHE) -> Path:
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"etmgeg_{station}.txt"
    if not path.exists():
        with urllib.request.urlopen(URL.format(station=station), timeout=120) as response:
            archive = zipfile.ZipFile(io.BytesIO(response.read()))
        path.write_bytes(archive.read(f"etmgeg_{station}.txt"))
    return path


def parse(text: str) -> pd.DataFrame:
    """Parse the content of an etmgeg_<station>.txt file."""
    lines = text.splitlines()
    header = next(i for i, line in enumerate(lines) if line.startswith("# STN,YYYYMMDD"))
    names = [c.strip() for c in lines[header].lstrip("# ").split(",")]
    raw = pd.read_csv(io.StringIO("\n".join(lines[header + 1:])), names=names, skipinitialspace=True,
                      na_values=["", " "], dtype=str)
    out = pd.DataFrame({"date": pd.to_datetime(raw["YYYYMMDD"].str.strip(), format="%Y%m%d")})
    for column, (name, scale) in COLUMNS.items():
        values = pd.to_numeric(raw[column], errors="coerce")
        if column in ("RH", "SQ"):
            values = values.where(values != -1, 0)  # -1 means "< 0.05": treat as zero
        out[name] = values * scale
    return out


def load_daily(station: int, start: str | None = None, end: str | None = None, cache: Path = CACHE) -> pd.DataFrame:
    df = parse(download(station, cache).read_text(encoding="latin-1"))
    if start:
        df = df[df["date"] >= pd.Timestamp(start)]
    if end:
        df = df[df["date"] <= pd.Timestamp(end)]
    return df.reset_index(drop=True)


def synthetic_daily(start: str, end: str, seed: int = 0) -> pd.DataFrame:
    """Plausible Dutch weather without a download (for tests and offline use)."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="D")
    doy = dates.dayofyear.to_numpy()
    season = np.sin(2 * np.pi * (doy - 110) / 365)
    anomaly = np.zeros(len(dates))
    for i in range(1, len(dates)):  # weather systems last a few days
        anomaly[i] = 0.8 * anomaly[i - 1] + rng.normal(0, 1.4)
    temp = 10.5 + 7.5 * season + anomaly
    radiation = np.clip(1050 + 950 * season + rng.normal(0, 250, len(dates)), 50, None)
    rain = np.where(rng.random(len(dates)) < 0.45, rng.gamma(1.3, 3.5, len(dates)), 0.0)
    return pd.DataFrame({
        "date": dates, "temp_mean_c": temp.round(1), "temp_min_c": (temp - 4).round(1),
        "temp_max_c": (temp + 5).round(1), "rain_mm": rain.round(1),
        "sun_hours": np.clip(radiation / 160, 0, 15).round(1), "radiation_j_cm2": radiation.round(0),
        "evaporation_mm": np.clip(radiation / 420, 0.1, None).round(1), "humidity_pct": 80.0,
    })
