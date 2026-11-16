"""Heating energy from the greenhouse temperature and the weather, fitted on the real tomato greenhouses.

    python scripts/greenhouse_heat.py        # recalibrate on the raw data and test it

A heat balance per day: what leaks out through the glass, minus what the sun and the lamps bring in,
plus a base use (minimum pipe temperatures, dehumidifying):

    heat = own_factor x max(0, loss x (inside - outside) - sun x radiation - lamp x lamp energy + base)

**Every greenhouse has its own factor.** At the same temperature the six real greenhouses used between
0.5 and 1.7 times the average, depending on how the grower heats (pipes, windows, screens). Without that
factor the heat balance is 35-47% off over a season on a greenhouse it never saw; nothing in the
settings predicts it (the same finding as for the machine-learning heating model). With the factor taken
from the greenhouse's own last four weeks, the rest of the season is 18% off on average (16% median),
against 25-84% for "it stays like the last four weeks". So a grower needs four weeks of his own heating
use, and gets a forecast that follows his plan and the weather.

The lamp share is what the data shows: 7% of the lamp energy replaced heating. Fixing it to a textbook
40-70% made the fit clearly worse (23-54% off); in these greenhouses the pipes kept running under the lamps.

Fitted on six compartments, December 2019 - May 2020 (Autonomous Greenhouse Challenge, WUR).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

KWH_PER_LAMP_HOUR = 0.1311      # measured: lamp electricity per lamp hour (HPS + LED, kWh/m2)
OWN_FACTOR_SPREAD = 0.45        # the real factors (0.50 .. 1.67) as a log-normal spread
OWN_FACTOR_DRIFT = 0.15         # how much a grower's factor moves per month (gives the 18% of the real data)


@dataclass(frozen=True)
class HeatParameters:
    loss_mj_per_degree: float = 0.1746     # per degree difference per day (U = 2.0 W/m2K with screens)
    sun_share: float = 0.0564              # MJ less per MJ/m2 of outside radiation
    lamp_share: float = 0.0706             # MJ less per MJ of lamp electricity
    base_mj: float = 0.5368                # minimum pipes, dehumidifying


PARAMETERS = HeatParameters()


def heat_balance(inside_c: np.ndarray, outside_c: np.ndarray, radiation_j_cm2: np.ndarray, lamp_kwh: np.ndarray,
                 parameters: HeatParameters = PARAMETERS) -> np.ndarray:
    """Heating energy in MJ/m2 per day for an average greenhouse (own factor 1)."""
    p = parameters
    need = (p.loss_mj_per_degree * (np.asarray(inside_c, dtype=float) - np.asarray(outside_c, dtype=float))
            - p.sun_share * np.asarray(radiation_j_cm2, dtype=float) / 100.0           # J/cm2 -> MJ/m2
            - p.lamp_share * np.asarray(lamp_kwh, dtype=float) * 3.6 + p.base_mj)
    return np.clip(need, 0.0, None)


def own_factor(heat_used: np.ndarray, balance: np.ndarray) -> float:
    """A greenhouse's own factor from a period of its real use and the heat balance for those days."""
    expected = float(np.sum(balance))
    return float(np.sum(heat_used)) / expected if expected > 1e-6 else 1.0


# --------------------------------------------------------------------------- calibration on the real data

def load_real(extracted) -> dict[str, pd.DataFrame]:
    import make_greenhouse_challenge as agc
    import tomato_crop as crop

    real = crop.load_real(extracted)
    weather = agc.read_log(extracted / "Weather" / "Weather.csv").groupby("date").agg(outside=("tout", "mean"))
    for team, days in real.items():
        resources = agc.daily_resources(agc.read_log(extracted / team / "Resources.csv"))
        days["heat"] = resources["heat_mj_m2"].reindex(days.index)
        days["lamp_kwh"] = resources["electricity_kwh_m2"].reindex(days.index)
        days["outside"] = weather["outside"].reindex(days.index)
        real[team] = days.dropna(subset=["heat", "outside", "lamp_kwh"])
    return real


def _balance(days: pd.DataFrame, p: HeatParameters) -> np.ndarray:
    return heat_balance(days["temp"].to_numpy(), days["outside"].to_numpy(), days["radiation"].to_numpy(), days["lamp_kwh"].to_numpy(), p)


def calibrate(real: dict[str, pd.DataFrame]) -> HeatParameters:
    from scipy.optimize import least_squares

    table = pd.concat(real.values())
    fit = least_squares(lambda x: _balance(table, HeatParameters(*x)) - table["heat"].to_numpy(), [0.3, 0.1, 0.3, 0.0],
                        bounds=([0, 0, 0, -5], [2, 2, 1, 5]))
    return HeatParameters(*(round(float(v), 4) for v in fit.x))


def validate(real: dict[str, pd.DataFrame], history_days: int = 28) -> pd.DataFrame:
    """Leave each greenhouse out; forecast the rest of its season from three decision days."""
    rows = []
    for team, days in real.items():
        p = calibrate({k: v for k, v in real.items() if k != team})
        balance, used = _balance(days, p), days["heat"].to_numpy()
        for start in (28, 56, 84):
            factor = own_factor(used[start - history_days:start], balance[start - history_days:start])
            actual = used[start:].sum()
            rows.append({"greenhouse": team, "decision_day": start, "own_factor": factor,
                         "with_own_history_pct": abs(factor * balance[start:].sum() / actual - 1) * 100,
                         "without_history_pct": abs(balance[start:].sum() / actual - 1) * 100,
                         "stays_like_last_weeks_pct": abs(used[start - history_days:start].mean() * len(used[start:]) / actual - 1) * 100})
    return pd.DataFrame(rows)


def main() -> None:
    import make_greenhouse_challenge as agc

    real = load_real(agc.download())
    table = validate(real)
    print("heating for the rest of the season, each greenhouse left out in turn (% off):")
    print(table.groupby("decision_day")[["with_own_history_pct", "without_history_pct", "stays_like_last_weeks_pct"]].mean().round(1).to_string())
    print(f"\nmean: {table['with_own_history_pct'].mean():.1f}% with own history, {table['without_history_pct'].mean():.1f}% without, "
          f"{table['stays_like_last_weeks_pct'].mean():.1f}% for 'it stays like the last weeks'")
    print(f"calibrated on all six: {calibrate(real)}")
    all_days = pd.concat(real.values())
    print(f"lamp electricity per lamp hour: {all_days['lamp_kwh'].sum() / all_days['lamp_hours'].sum():.4f} kWh/m2")


if __name__ == "__main__":
    main()
