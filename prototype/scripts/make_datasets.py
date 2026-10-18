"""Create the demo datasets in ./data.

* greenhouse.csv            synthetic greenhouse climate, 10-minute intervals
* absence.csv               synthetic, aggregated sickness absence per department/month
* absence_with_diagnosis.csv  a deliberately non-compliant export (for the policy demo)
* ai4i2020.csv              real AI4I 2020 Predictive Maintenance data (UCI, CC BY 4.0)

The greenhouse and absence data are simulated: no real or personal data is used
for development, in line with Autoriteit Persoonsgegevens (2016).
"""

from __future__ import annotations

import argparse
import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

AI4I_URL = "https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip"


def greenhouse(days: int = 90, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    steps = days * 24 * 6
    time = pd.date_range("2025-03-01", periods=steps, freq="10min")
    hour = time.hour + time.minute / 60
    doy = time.dayofyear.to_numpy()

    clouds = np.clip(0.65 + np.cumsum(rng.normal(0, 0.03, steps)) % 0.7 - 0.2, 0.15, 1.0)
    outside = (8 + 6 * np.sin(2 * np.pi * (doy - 110) / 365) + 5 * np.sin(2 * np.pi * (hour - 9) / 24)
               + np.cumsum(rng.normal(0, 0.05, steps)) % 3 - 1.5)
    solar = np.clip(np.sin(np.pi * (hour - 6) / 14), 0, None) * (450 + 250 * np.sin(2 * np.pi * (doy - 80) / 365))
    solar = solar * clouds

    inside = np.empty(steps)
    humidity = np.empty(steps)
    co2 = np.empty(steps)
    vent = np.zeros(steps, dtype=int)
    heat = np.zeros(steps, dtype=int)
    inside[0], humidity[0], co2[0] = 18.0, 75.0, 600.0
    for t in range(steps - 1):
        day = 7 <= hour[t] < 19
        setpoint = 20.0 if day else 17.0
        heat[t] = int(inside[t] < setpoint)
        vent[t] = int(inside[t] > 24.0 or humidity[t] > 88.0)
        dT = (0.04 * (outside[t] - inside[t]) * (1 + 3 * vent[t]) + 0.0035 * solar[t] + 0.6 * heat[t])
        inside[t + 1] = inside[t] + dT + rng.normal(0, 0.05)
        transpiration = 0.004 * solar[t]
        humidity[t + 1] = np.clip(humidity[t] + transpiration - 2.5 * vent[t] + 0.2 - 0.004 * (humidity[t] - 70)
                                  + rng.normal(0, 0.3), 40, 98)
        dosing = 40 if (day and not vent[t]) else 0
        co2[t + 1] = np.clip(co2[t] + dosing - 0.002 * solar[t] * 10 - 0.08 * (co2[t] - 420) * (1 + 4 * vent[t])
                             + rng.normal(0, 5), 380, 1200)
    heat[-1], vent[-1] = heat[-2], vent[-2]

    return pd.DataFrame({
        "timestamp": time,
        "hour_of_day": time.hour,
        "outside_temp_c": outside.round(2),
        "solar_radiation_wm2": solar.round(1),
        "inside_humidity_pct": humidity.round(1),
        "co2_ppm": co2.round(0),
        "ventilation_on": vent,
        "heating_on": heat,
        "inside_temp_c": inside.round(2),
    })


def absence(departments: int = 40, months: int = 48, seed: int = 2) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    kinds = {"production": 5.5, "logistics": 5.0, "care": 6.2, "office": 3.4}
    periods = pd.period_range("2022-01", periods=months, freq="M")
    rows = []
    for d in range(departments):
        kind = list(kinds)[d % len(kinds)]
        headcount = int(rng.integers(8, 120)) if d not in (7, 23) else 4  # two tiny teams
        avg_age = float(rng.uniform(28, 52))
        previous = kinds[kind]
        for p in periods:
            flu = int(p.month in (12, 1, 2, 3))
            overtime = max(0.0, rng.normal(6 if kind != "office" else 3, 2.5))
            workload = float(np.clip(rng.normal(1.0, 0.15), 0.6, 1.5))
            pct = (kinds[kind] + 0.06 * (avg_age - 40) + 0.12 * overtime + 1.6 * flu + 2.0 * (workload - 1)
                   + 0.35 * (previous - kinds[kind]) + rng.normal(0, 0.6 + 3 / np.sqrt(headcount)))
            pct = float(np.clip(pct, 0, 30))
            rows.append({
                "period": p.to_timestamp().date().isoformat(),
                "department_id": f"dep-{d:03d}",
                "department_type": kind,
                "headcount": headcount,
                "avg_age": round(avg_age, 1),
                "overtime_hours_per_fte": round(overtime, 2),
                "workload_index": round(workload, 3),
                "flu_season": flu,
                "previous_month_absence_pct": round(previous, 2),
                "absence_pct": round(pct, 2),
            })
            previous = pct
    return pd.DataFrame(rows)


def download_ai4i(target: Path) -> None:
    with urllib.request.urlopen(AI4I_URL, timeout=60) as response:
        archive = zipfile.ZipFile(io.BytesIO(response.read()))
    target.write_bytes(archive.read("ai4i2020.csv"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("data"))
    parser.add_argument("--skip-download", action="store_true", help="do not download AI4I 2020")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    greenhouse().to_csv(args.out / "greenhouse.csv", index=False)
    data = absence()
    data.to_csv(args.out / "absence.csv", index=False)
    bad = data.head(200).copy()
    bad["diagnosis"] = np.random.default_rng(3).choice(["flu", "back pain", "burn-out"], len(bad))
    bad.to_csv(args.out / "absence_with_diagnosis.csv", index=False)
    print(f"wrote greenhouse.csv, absence.csv, absence_with_diagnosis.csv to {args.out}/")

    ai4i = args.out / "ai4i2020.csv"
    if args.skip_download or ai4i.exists():
        print("AI4I 2020: skipped" if args.skip_download else "AI4I 2020: already present")
        return
    try:
        download_ai4i(ai4i)
        print(f"downloaded AI4I 2020 to {ai4i}")
    except OSError as exc:
        print(f"could not download AI4I 2020 ({exc}); download it manually from {AI4I_URL}")


if __name__ == "__main__":
    main()
