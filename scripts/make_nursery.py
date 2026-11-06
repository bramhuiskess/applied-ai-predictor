"""Synthetic potted-plant nursery with a known cause-and-effect model.

    python scripts/make_nursery.py                      # real KNMI weather (Rotterdam, near Westland)
    python scripts/make_nursery.py --weather synthetic  # offline
    python scripts/make_nursery.py --faults             # also write a copy with broken sensors

One row = one crop batch ("teelt") in one greenhouse compartment. The grower
decides the climate at the start (temperature, lighting, CO2, fertiliser,
plant density); the simulator then grows the plants week by week on real
weather until they reach selling height, and books the gas and electricity.

Why simulate instead of only using real data: the simulator *is* the ground
truth. For any batch it can answer "what would have happened at 21 instead of
20 degrees?", so the scenario predictions (tickets 4 and 5) can be checked
against the real answer, not just against an opinion.

Realism on purpose:
* weather is real KNMI data (2015-2025), including cold snaps and the 2018 heat;
* growers react to the gas price (2022 crisis: colder greenhouses, less light),
  so temperature and gas use are confounded, exactly the trap a scenario model
  has to avoid;
* every batch has its own luck (pests, cutting quality), every week some noise.

The relations are simplified textbook horticulture (temperature optimum
curve, light-use saturation, CO2 enrichment, EC optimum, heat loss through
the cover). The constants are plausible, not calibrated on a real grower;
that is the first thing to adjust after talking to one.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import knmi  # noqa: E402

STATION = 344  # Rotterdam, closest KNMI station to the Westland greenhouse area
START, END = "2016-01-04", "2025-12-31"

TARGET_HEIGHT_CM, START_HEIGHT_CM = 28.0, 4.0
MAX_WEEKS = 24
MAX_GROWTH_CM_WEEK = 4.6
GAS_J_PER_M3 = 31.65e6
BOILER_EFFICIENCY = 0.92
WEATHER_WINDOW_WEEKS = 10
VENT_TEMP_C = 26.0  # weather features describe the first 10 weeks, whatever the decisions

LEVERS = ["temp_setpoint_c", "light_hours", "co2_ppm", "ec_ms_cm", "plant_density_m2"]
OUTPUTS = ["weeks_to_harvest", "plants_sold", "gas_m3", "electricity_kwh"]

# Gas price for growers (EUR/m3, rough commodity level), linear between these points.
GAS_PRICE_POINTS = [
    ("2015-01-01", 0.21), ("2017-01-01", 0.19), ("2018-10-01", 0.25), ("2020-05-01", 0.11),
    ("2021-06-01", 0.30), ("2021-12-15", 1.10), ("2022-03-07", 1.40), ("2022-06-01", 0.95),
    ("2022-08-26", 2.30), ("2022-11-01", 1.05), ("2023-03-01", 0.55), ("2023-10-01", 0.48),
    ("2024-06-01", 0.36), ("2025-02-01", 0.50), ("2025-12-31", 0.36),
]


@dataclass(frozen=True)
class Compartment:
    name: str
    area_m2: int
    u_value: float        # effective heat loss, W per m2 floor per degree
    lamp_type: str        # led | hps | none
    lamp_umol: float      # light intensity of the lamps
    habit_temp: float     # grower habits per compartment
    habit_ec: float
    habit_density: float

    @property
    def lamp_w_m2(self) -> float:
        efficacy = {"led": 2.8, "hps": 1.8}.get(self.lamp_type)  # umol per joule
        return self.lamp_umol / efficacy if efficacy else 0.0


def make_compartments(n: int, rng: np.random.Generator) -> list[Compartment]:
    lamp_types = ["led", "led", "hps", "none"]
    compartments = []
    for i in range(n):
        lamp = lamp_types[i % len(lamp_types)]
        compartments.append(Compartment(
            name=f"K{i + 1:02d}",
            area_m2=int(rng.integers(20, 61) * 100),
            u_value=round(float(rng.uniform(3.4, 6.2)), 2),
            lamp_type=lamp,
            lamp_umol=0.0 if lamp == "none" else float(rng.choice([150, 180, 210])),
            habit_temp=round(float(rng.normal(0, 0.6)), 2),
            habit_ec=round(float(rng.uniform(1.6, 2.4)), 2),
            habit_density=round(float(rng.uniform(38, 58)), 1),
        ))
    return compartments


def gas_price(dates: pd.Series | pd.DatetimeIndex) -> np.ndarray:
    xs = np.array([pd.Timestamp(d).value for d, _ in GAS_PRICE_POINTS], dtype=float)
    ys = np.array([p for _, p in GAS_PRICE_POINTS])
    # pandas 3 may store dates in microseconds: convert explicitly to nanoseconds like Timestamp.value
    return np.interp(pd.DatetimeIndex(pd.to_datetime(dates)).as_unit("ns").asi8.astype(float), xs, ys)


def weekly_weather(daily: pd.DataFrame) -> pd.DataFrame:
    """Weekly means, indexed by the Monday that starts the week."""
    weekly = daily.set_index("date")[["temp_mean_c", "temp_max_c", "radiation_j_cm2"]].resample("W-MON", label="left", closed="left").agg(
        {"temp_mean_c": "mean", "temp_max_c": "max", "radiation_j_cm2": "mean"})
    return weekly.interpolate(limit_direction="both")


# --------------------------------------------------------------------------- the plant model

def f_temperature(t: float) -> float:
    return math.exp(-((t - 22.0) / 5.0) ** 2)


def daily_light_integral(radiation_j_cm2: float, light_hours: float, lamp_umol: float) -> float:
    sun = radiation_j_cm2 * 0.0215 * 0.65          # global radiation -> PAR mol/m2/day, 65% through the glass
    lamps = light_hours * lamp_umol * 3600 / 1e6    # mol/m2/day from the lamps
    return sun + lamps


def f_light(dli: float) -> float:
    return (dli / (dli + 9.0)) / (30.0 / 39.0)


def f_co2(ppm: float) -> float:
    return 0.77 + 0.23 * (1 - math.exp(-max(ppm - 350.0, 0.0) / 250.0))


def f_ec(ec: float) -> float:
    return max(0.6, 1 - 0.12 * ((ec - 2.0) / 0.7) ** 2)


def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


class Nursery:
    """Simulates batches. The same object is the 'answer key' for scenarios."""

    def __init__(self, daily_weather: pd.DataFrame, compartments: list[Compartment]):
        self.weather = weekly_weather(daily_weather)
        self.compartments = {c.name: c for c in compartments}

    def simulate(self, batch: dict) -> dict:
        """Grow one batch. `batch` holds the start date, compartment, levers and its luck."""
        c = self.compartments[batch["compartment"]]
        week0 = pd.Timestamp(batch["start_date"])
        rng = np.random.default_rng(batch["luck_seed"])  # same luck for every scenario of this batch
        height, weeks, gas_j, light_kwh = START_HEIGHT_CM, 0.0, 0.0, 0.0
        inside_temps, max_temps, dlis = [], [], []
        while height < TARGET_HEIGHT_CM and weeks < MAX_WEEKS:
            week = week0 + pd.Timedelta(weeks=int(weeks))
            if week not in self.weather.index:
                return {}
            w = self.weather.loc[week]
            setpoint = batch["temp_setpoint_c"]
            # Passive solar heating; the vents open at 26 degrees (a separate setting from heating).
            inside = max(setpoint, min(w.temp_mean_c + 4 + w.radiation_j_cm2 / 300, VENT_TEMP_C))
            dli = daily_light_integral(w.radiation_j_cm2, batch["light_hours"], c.lamp_umol)
            growth = (MAX_GROWTH_CM_WEEK * f_temperature(inside) * f_light(dli) * f_co2(batch["co2_ppm"])
                      * f_ec(batch["ec_ms_cm"]) * (1 + 0.004 * (batch["plant_density_m2"] - 45))
                      * batch["vigour"] * rng.lognormal(0, 0.05))
            step = min(1.0, (TARGET_HEIGHT_CM - height) / growth) if growth > 0 else 1.0
            # Heat demand: loss through the cover, minus sun and lamp heat.
            lamp_w = c.lamp_w_m2 * batch["light_hours"] / 24 * 0.7
            solar_w = w.radiation_j_cm2 * 1e4 / 86400 * 0.35
            loss_w = c.u_value * max(0.0, setpoint - w.temp_mean_c)
            gas_j += max(0.0, loss_w - solar_w - lamp_w) * c.area_m2 * 604800 * step
            light_kwh += c.lamp_w_m2 * c.area_m2 * batch["light_hours"] * 7 * step / 1000
            height += growth * step
            weeks += step
            inside_temps.append(inside)
            max_temps.append(w.temp_max_c + w.radiation_j_cm2 / 300)
            dlis.append(dli)
        planted = batch["plant_density_m2"] * c.area_m2 * 0.85
        loss = (0.03
                + 0.07 * sigmoid((17.5 - float(np.mean(inside_temps))) * 1.5)   # botrytis when cold and damp
                + 0.05 * sigmoid((float(np.max(max_temps)) - 31) * 0.8)         # heat stress
                + 0.08 * sigmoid((7.0 - float(np.mean(dlis))) * 0.8)           # weak plants in the dark
                + 0.0012 * max(0.0, batch["plant_density_m2"] - 48) ** 1.6      # crowding
                + 0.30 * (1 - f_ec(batch["ec_ms_cm"]))
                + batch["bad_luck"])
        return {
            "weeks_to_harvest": round(weeks, 2),
            "plants_sold": int(round(planted * (1 - min(loss, 0.6)))),
            "gas_m3": round(gas_j / GAS_J_PER_M3 / BOILER_EFFICIENCY, 0),
            "electricity_kwh": round(light_kwh, 0),
        }

    def counterfactual(self, batch: dict, **changes: float) -> dict:
        """The true outcome of the same batch with other decisions."""
        return self.simulate({**batch, **changes})

    def exogenous_weather(self, start_date) -> dict:
        """Weather of the first weeks after the start: known as a forecast, not affected by decisions."""
        week0 = pd.Timestamp(start_date)
        window = self.weather.loc[week0: week0 + pd.Timedelta(weeks=WEATHER_WINDOW_WEEKS - 1)]
        return {"outside_temp_avg_c": round(float(window.temp_mean_c.mean()), 2),
                "radiation_avg_j_cm2": round(float(window.radiation_j_cm2.mean()), 1)}


# --------------------------------------------------------------------------- the grower

def climatology(weather: pd.DataFrame) -> pd.Series:
    """Average radiation per week of the year, what a grower expects."""
    return weather["radiation_j_cm2"].groupby(weather.index.isocalendar().week).mean()


def decide(c: Compartment, start: pd.Timestamp, price: float, expected_radiation: float,
           rng: np.random.Generator) -> dict:
    """The grower's choices: habits, season, energy price, and some trying things out."""
    expensive = float(np.clip((price - 0.25) / 0.85, 0, 1))
    dark = float(np.clip((1100 - expected_radiation) / 900, 0, 1))
    temp = 20.6 + c.habit_temp - 1.8 * expensive - 0.3 * dark + rng.normal(0, 0.8)
    light = 0.0 if c.lamp_type == "none" else 16 * dark * (1 - 0.55 * expensive) + rng.normal(0, 1.6)
    co2 = 550 + 380 * dark - 160 * expensive + rng.normal(0, 90)
    return {
        "temp_setpoint_c": round(float(np.clip(temp, 16.5, 24.5)), 1),
        "light_hours": round(float(np.clip(light, 0, 18)) * 2) / 2,
        "co2_ppm": int(np.clip(round(co2, -1), 400, 1100)),
        "ec_ms_cm": round(float(np.clip(c.habit_ec + rng.normal(0, 0.15), 1.2, 2.9)), 2),
        "plant_density_m2": round(float(np.clip(c.habit_density + rng.normal(0, 3), 32, 64)), 1),
    }


def generate(daily_weather: pd.DataFrame, compartments: int = 16, seed: int = 7,
             start: str = START, end: str = END) -> tuple[pd.DataFrame, Nursery, list[dict]]:
    """Run the nursery for all compartments. Returns (dataset, simulator, raw batches)."""
    rng = np.random.default_rng(seed)
    nursery = Nursery(daily_weather, make_compartments(compartments, rng))
    clim = climatology(nursery.weather)
    last_start = pd.Timestamp(end) - pd.Timedelta(weeks=MAX_WEEKS)
    rows, batches = [], []
    for c in nursery.compartments.values():
        week = pd.Timestamp(start) + pd.Timedelta(weeks=int(rng.integers(0, 6)))
        while week <= last_start:
            price = float(gas_price([week])[0])
            batch = {"batch_id": f"{c.name}-{week:%Y%m%d}", "compartment": c.name, "start_date": week,
                     **decide(c, week, price, float(clim.loc[week.isocalendar().week]), rng),
                     "vigour": float(rng.lognormal(0, 0.06)),
                     "bad_luck": float(rng.beta(1.2, 40)),        # mostly ~3%, sometimes a bad pest year
                     "luck_seed": int(rng.integers(0, 2**31))}
            out = nursery.simulate(batch)
            if not out:
                break
            batches.append(batch)
            rows.append({"batch_id": batch["batch_id"], "start_date": f"{week:%Y-%m-%d}",
                         "start_week": int(week.isocalendar().week), "compartment": c.name,
                         "area_m2": c.area_m2, "lamp_type": c.lamp_type, "u_value": c.u_value,
                         **{k: batch[k] for k in LEVERS},
                         "gas_price_eur_m3": round(price, 3),
                         **nursery.exogenous_weather(week), **out})
            week = week + pd.Timedelta(weeks=math.ceil(out["weeks_to_harvest"]) + 1)  # one week to clean up
    df = pd.DataFrame(rows).sort_values(["start_date", "compartment"]).reset_index(drop=True)
    return df, nursery, batches


SCENARIOS = {
    "temp_setpoint_c +1": {"temp_setpoint_c": 1.0},
    "light_hours +2": {"light_hours": 2.0},
    "co2_ppm +200": {"co2_ppm": 200.0},
    "ec_ms_cm +0.3": {"ec_ms_cm": 0.3},
    "plant_density_m2 +5": {"plant_density_m2": 5.0},
}


def true_effects(nursery: Nursery, batches: list[dict], deltas: dict[str, float]) -> dict[str, dict]:
    """Average true effect of changing levers, over all batches (the answer key)."""
    diffs = {o: [] for o in OUTPUTS}
    bases = {o: [] for o in OUTPUTS}
    for b in batches:
        if b["light_hours"] == 0 and "light_hours" in deltas and nursery.compartments[b["compartment"]].lamp_type == "none":
            continue  # no lamps: lighting is not a lever there
        base = nursery.simulate(b)
        changed = nursery.counterfactual(b, **{k: b[k] + v for k, v in deltas.items()})
        if not base or not changed:
            continue
        for o in OUTPUTS:
            diffs[o].append(changed[o] - base[o])
            bases[o].append(base[o])
    # Percentages as total change / total base: an average of per-batch percentages explodes on
    # summer batches that use almost no gas.
    return {o: {"mean_change": round(float(np.mean(diffs[o])), 2),
                "change_pct": round(float(np.sum(diffs[o]) / np.sum(bases[o]) * 100), 2) if np.sum(bases[o]) else None}
            for o in OUTPUTS}


# Columns that come from other tables once the data is split (see split_tables): the compartment's
# attributes and the weather of the first WEATHER_WINDOW_WEEKS weeks. nursery.csv keeps all of them.
JOINED_COLUMNS = ["area_m2", "lamp_type", "u_value", "outside_temp_avg_c", "radiation_avg_j_cm2"]


def split_tables(df: pd.DataFrame, daily: pd.DataFrame, nursery: Nursery) -> dict[str, pd.DataFrame]:
    """The same data as separate tables, to be connected again by the joins in the manifest.

    * base: one row per batch, without what is looked up from another table;
    * compartments: one row per compartment (key join on ``compartment``);
    * weather: daily KNMI values (time join on ``start_date``, window of WEATHER_WINDOW_WEEKS weeks).
    """
    compartments = pd.DataFrame([{"compartment": c.name, "area_m2": c.area_m2, "lamp_type": c.lamp_type,
                                  "u_value": c.u_value} for c in nursery.compartments.values()])
    weather = daily[["date", "temp_mean_c", "temp_min_c", "temp_max_c", "rain_mm", "radiation_j_cm2"]].copy()
    weather["date"] = weather["date"].dt.strftime("%Y-%m-%d")
    return {"base": df.drop(columns=JOINED_COLUMNS), "compartments": compartments, "weather": weather}


def inject_faults(df: pd.DataFrame, seed: int = 11) -> tuple[pd.DataFrame, list[dict]]:
    """A copy with realistic data problems, plus a log of exactly what was broken."""
    rng = np.random.default_rng(seed)
    bad = df.copy()
    log = []
    # 1. outside temperature sensor stuck for ~8 months in one compartment
    comp = bad["compartment"].iloc[0]
    rows = bad.index[(bad["compartment"] == comp) & bad["start_date"].between("2021-03-01", "2021-11-01")]
    bad.loc[rows, "outside_temp_avg_c"] = 12.0
    log.append({"fault": "stuck_sensor", "column": "outside_temp_avg_c", "rows": rows.tolist()})
    # 2. CO2 sensor spikes
    rows = rng.choice(bad.index, size=max(3, len(bad) // 150), replace=False)
    bad.loc[rows, "co2_ppm"] = 5000
    log.append({"fault": "spike", "column": "co2_ppm", "rows": sorted(int(r) for r in rows)})
    # 3. missing radiation values
    rows = rng.choice(bad.index, size=max(5, len(bad) // 40), replace=False)
    bad.loc[rows, "radiation_avg_j_cm2"] = np.nan
    log.append({"fault": "missing", "column": "radiation_avg_j_cm2", "rows": sorted(int(r) for r in rows)})
    # 4. one compartment books gas in kWh instead of m3 for a year (factor ~9.77)
    comp = bad["compartment"].iloc[1]
    rows = bad.index[(bad["compartment"] == comp) & bad["start_date"].str.startswith("2019")]
    bad.loc[rows, "gas_m3"] = (bad.loc[rows, "gas_m3"] * 9.77).round(0)
    log.append({"fault": "unit_error", "column": "gas_m3", "rows": rows.tolist()})
    # 5. duplicated export rows
    rows = rng.choice(bad.index, size=max(3, len(bad) // 100), replace=False)
    bad = pd.concat([bad, bad.loc[rows]]).sort_values(["start_date", "compartment"]).reset_index(drop=True)
    log.append({"fault": "duplicate_rows", "column": None, "rows": sorted(int(r) for r in rows)})
    return bad, log


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("data/nursery.csv"))
    parser.add_argument("--weather", choices=["knmi", "synthetic"], default="knmi")
    parser.add_argument("--compartments", type=int, default=16)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--faults", action="store_true", help="also write data/nursery_faulty.csv")
    args = parser.parse_args()

    if args.weather == "knmi":
        daily = knmi.load_daily(STATION, "2015-06-01", END)
    else:
        daily = knmi.synthetic_daily("2015-06-01", END)
    df, nursery, batches = generate(daily, args.compartments, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    tables = split_tables(df, daily, nursery)
    split_paths = {"base": args.out.with_name(args.out.stem + "_base.csv"),
                   "compartments": args.out.with_name("compartments.csv"),
                   "weather": args.out.with_name("weather_rotterdam.csv")}  # the name the manifests use, also for --weather synthetic
    for name, path in split_paths.items():
        tables[name].to_csv(path, index=False)

    truth = {"description": "Average true effect per scenario over all batches (answer key, same luck per batch)",
             "weather": args.weather, "batches": len(df),
             "scenarios": {name: true_effects(nursery, batches, deltas) for name, deltas in SCENARIOS.items()}}
    truth_path = args.out.with_name(args.out.stem + "_truth.json")
    truth_path.write_text(json.dumps(truth, indent=2), encoding="utf-8")
    print(f"wrote {args.out}: {len(df)} batches, {df['compartment'].nunique()} compartments, "
          f"{df['start_date'].min()} .. {df['start_date'].max()}")
    print(f"wrote {truth_path}")
    print("also wrote the same data as separate tables for the joins in the manifests: "
          + ", ".join(p.name for p in split_paths.values()))
    if args.faults:
        bad, log = inject_faults(df)
        bad_path = args.out.with_name(args.out.stem + "_faulty.csv")
        bad.to_csv(bad_path, index=False)
        bad_path.with_suffix(".faults.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
        print(f"wrote {bad_path} ({sum(len(f['rows']) for f in log)} broken rows, see {bad_path.with_suffix('.faults.json').name})")


if __name__ == "__main__":
    main()
