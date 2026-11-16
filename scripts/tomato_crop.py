"""A small tomato crop model, calibrated on real greenhouses (Autonomous Greenhouse Challenge 2019-2020).

    python scripts/tomato_crop.py        # recalibrate on the raw data and test on greenhouses left out

Why a crop model: the six real tomato greenhouses ended within 5% of each other, so their data holds
almost no "what if" to learn from. A model that only learns from that data predicts yield worse than a
plain guess (12% against 4% on the season total). Crop knowledge supplies the cause and effect; the
real data only has to set three numbers.

The model, per day (a simplified version of the cohort models used in greenhouse research, e.g.
TOMSIM, Heuvelink 1996):

* **new trusses** set in proportion to the temperature above a base temperature;
* a truss **ripens** when it has collected enough warmth (a temperature sum);
* the **light** of the day becomes fruit growth, shared by the growing trusses;
* a truss cannot grow faster than its **sink strength** allows (early in the season the plant has
  more light than its few trusses can use).

So warmer means sooner ripe (and more, smaller trusses), and more light means more kilos once the
plant has enough trusses to use it.

What is measured and what is fitted:

* ripening: 705 degree-days above 7.4 C, from 138 real harvests (flowering date to harvest date);
* first flowering 15 days after planting (the first flowering dates in the data);
* fitted on the harvest curves: grams of fruit per mol of light, and the sink limit.

Tested by leaving each greenhouse out in turn (2019-2020, six compartments): season total 3.7% off,
first harvest 2.7 days off, harvest per month 12% off. There is one season of one crop behind this:
temperatures of 21-23.5 C on average, planted in December. Outside that range it is the crop
knowledge that speaks, not the data.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DEGREE_DAYS = 705.0        # warmth a truss needs from flowering to harvest
BASE_TEMP_C = 7.4          # below this a truss does not develop
FIRST_FLOWERING_DAY = 15   # days after planting
FINAL_HARVEST_FRACTION = 0.7   # the clean-up harvest on the last day also takes trusses this far along


@dataclass(frozen=True)
class CropParameters:
    fruit_g_per_mol: float = 5.629   # fresh fruit per mol of light (PAR) at the crop, when the trusses can use it
    sink_g_per_degree_day: float = 0.01338  # most a truss can grow per degree-day, per unit of truss


PARAMETERS = CropParameters()

# Light at the crop from what a grower knows (fitted on the same greenhouses, see `light_model`).
SUN_MOL_PER_J_CM2 = 0.0096      # mol/m2 of PAR inside per J/cm2 of outside radiation (cover, screens)
LAMP_MOL_PER_HOUR = 0.673       # mol/m2 per lamp hour (the challenge's HPS + LED)


def simulate(temp_c: np.ndarray, light_mol: np.ndarray, parameters: CropParameters = PARAMETERS,
             final_harvest: bool = False) -> np.ndarray:
    """Harvest in kg/m2 per day, from the daily greenhouse temperature and the light at the crop (mol/m2)."""
    n = len(temp_c)
    warmth = np.clip(np.asarray(temp_c, dtype=float) - BASE_TEMP_C, 0, None)
    light = np.asarray(light_mol, dtype=float)
    day = np.arange(n)
    size = np.where(day >= FIRST_FLOWERING_DAY, warmth, 0.0)   # trusses set that day
    age, weight = np.zeros(n), np.zeros(n)
    done = np.zeros(n, dtype=bool)
    harvest = np.zeros(n)
    for d in range(n):
        growing = (day <= d) & ~done & (size > 0)
        total = size[growing].sum()
        if total > 0:
            growth = min(parameters.fruit_g_per_mol * light[d], parameters.sink_g_per_degree_day * total * warmth[d])
            weight[growing] += growth * size[growing] / total
            age[growing] += warmth[d]
        needed = DEGREE_DAYS * (FINAL_HARVEST_FRACTION if final_harvest and d == n - 1 else 1.0)
        ripe = growing & (age >= needed)
        harvest[d] = weight[ripe].sum()
        done |= ripe
    return harvest / 1000.0


def light_at_crop(radiation_j_cm2: np.ndarray, lamp_hours: np.ndarray) -> np.ndarray:
    return SUN_MOL_PER_J_CM2 * np.asarray(radiation_j_cm2, dtype=float) + LAMP_MOL_PER_HOUR * np.asarray(lamp_hours, dtype=float)


# --------------------------------------------------------------------------- calibration on the real data

def load_real(extracted: Path) -> dict[str, pd.DataFrame]:
    """Per greenhouse and day: temperature, light at the crop, lamp hours, outside radiation, harvest (kg/m2)."""
    import make_greenhouse_challenge as agc

    weather = agc.read_log(extracted / "Weather" / "Weather.csv")
    radiation = weather.groupby("date")["iglob"].apply(lambda s: s.fillna(0).sum() * 300 / 1e4)
    real = {}
    for team in agc.TEAMS:
        climate = agc.read_log(extracted / team / "GreenhouseClimate.csv")
        days = climate.groupby("date").agg(
            temp=("tair", "mean"), light=("tot_par", lambda s: s.fillna(0).sum() * 300 / 1e6),
            lamp_hours=("tot_par_lamps", lambda s: (s.fillna(0) > 0).sum() * agc.SAMPLE_HOURS)).asfreq("D")
        days = days.interpolate().bfill().ffill()
        production = agc.read_log(extracted / team / "Production.csv")
        production = production[production["date"] >= days.index.min()]
        days["kg"] = (production["proda"] + production["prodb"]).groupby(production["date"]).sum().reindex(days.index).fillna(0)
        days["radiation"] = radiation.reindex(days.index).interpolate().bfill().ffill()
        real[team] = days
    return real


def calibrate(real: dict[str, pd.DataFrame]) -> CropParameters:
    """Fit grams per mol and the sink limit on the cumulative harvest curves."""
    from scipy.optimize import minimize

    def loss(x: np.ndarray) -> float:
        if x[0] <= 0 or x[1] <= 0:
            return 1e9
        p = CropParameters(float(x[0]), float(x[1]))
        return sum(float(np.mean((np.cumsum(simulate(g["temp"].to_numpy(), g["light"].to_numpy(), p, True))
                                  - g["kg"].cumsum().to_numpy()) ** 2)) for g in real.values())

    best = min((minimize(loss, [6.0, start], method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-7, "maxiter": 600})
                for start in (0.01, 0.03, 0.1)), key=lambda r: r.fun)
    return CropParameters(round(float(best.x[0]), 3), round(float(best.x[1]), 5))


def light_model(real: dict[str, pd.DataFrame]) -> tuple[float, float]:
    """Light at the crop = a * outside radiation + b * lamp hours (least squares through zero)."""
    table = pd.concat(real.values())
    X = table[["radiation", "lamp_hours"]].to_numpy()
    a, b = np.linalg.lstsq(X, table["light"].to_numpy(), rcond=None)[0]
    return float(a), float(b)


def validate(real: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Leave each greenhouse out: calibrate on the others, predict its whole season."""
    rows = {}
    for team, g in real.items():
        p = calibrate({k: v for k, v in real.items() if k != team})
        predicted, actual = simulate(g["temp"].to_numpy(), g["light"].to_numpy(), p, True), g["kg"].to_numpy()
        month = pd.DataFrame({"m": g.index.to_period("M"), "y": actual, "p": predicted}).groupby("m").sum()
        month = month[month["y"] > 0]
        rows[team] = {"season_total_pct": abs(predicted.sum() / actual.sum() - 1) * 100,
                      "month_pct": float(np.mean(np.abs(month["p"] - month["y"]) / month["y"]) * 100),
                      "first_harvest_days": int(np.argmax(predicted > 0.01)) - int(np.argmax(actual > 0)),
                      "fruit_g_per_mol": p.fruit_g_per_mol, "sink_g_per_degree_day": p.sink_g_per_degree_day}
    return pd.DataFrame(rows).T


def main() -> None:
    import make_greenhouse_challenge as agc

    real = load_real(agc.download())
    table = validate(real)
    print("each greenhouse left out in turn:\n" + table.round(3).to_string())
    print(f"\nmean: season total {table['season_total_pct'].mean():.1f}% off, harvest per month "
          f"{table['month_pct'].mean():.1f}% off, first harvest {table['first_harvest_days'].abs().mean():.1f} days off")
    print(f"\ncalibrated on all six: {calibrate(real)}")
    a, b = light_model(real)
    print(f"light at the crop: {a:.4f} mol per J/cm2 of outside radiation + {b:.3f} mol per lamp hour")


if __name__ == "__main__":
    main()
