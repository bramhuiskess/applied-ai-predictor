"""Compare the scenario effects the trained models predict with the simulator's answer key.

    python scripts/check_scenarios.py   # after make_nursery.py and `predictor train`

For every test batch (the ones the models never saw) and every scenario, the
true effect comes from re-running the simulator with the same luck, and the
predicted effect from running the trained ONNX model on the base row and on
the changed row. This is the yardstick for the scenario tickets.

`--sweeps` also runs every lever over its whole range through the real scenario
engine (the code behind POST /models/{id}/scenario/sweep) and compares each curve
with the simulator's true curve for the same batch.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import knmi  # noqa: E402
import make_nursery as nursery_mod  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from predictor import bootstrap  # noqa: E402
from predictor.core.validation import FeatureValidator  # noqa: E402

MODELS = {"weeks_to_harvest": "nursery-harvest-weeks", "plants_sold": "nursery-plants-sold", "gas_m3": "nursery-gas",
          "electricity_kwh": "nursery-electricity"}


def predict(model, rows: list[dict]) -> np.ndarray:
    validator = FeatureValidator(model.feature_schema, reject_unknown_fields=False)
    features = [{k: r[k] for k in validator.feature_order} for r in rows]
    return np.asarray(model.runner.predict(validator.to_matrix(features)).values, dtype=float)


def compare(nursery, batches: list[dict], df: pd.DataFrame, registry, test_from: str) -> pd.DataFrame:
    by_id = {b["batch_id"]: b for b in batches}
    test = df[df["start_date"] >= test_from]
    out = []
    for scenario, deltas in nursery_mod.SCENARIOS.items():
        lever, delta = next(iter(deltas.items()))
        rows = test if lever != "light_hours" else test[test["lamp_type"] != "none"]
        changed = rows.copy()
        changed[lever] = changed[lever] + delta
        truth = {o: [] for o in MODELS}
        for _, row in rows.iterrows():
            b = by_id[row["batch_id"]]
            base, cf = nursery.simulate(b), nursery.counterfactual(b, **{lever: b[lever] + delta})
            for o in MODELS:
                truth[o].append(cf[o] - base[o])
        for output, use_case in MODELS.items():
            model = registry.resolve("nursery").only(output)  # one version, a model per output
            pred = predict(model, changed.to_dict("records")) - predict(model, rows.to_dict("records"))
            true = np.array(truth[output])
            base_total = rows[output].sum()
            out.append({
                "scenario": scenario, "output": output, "batches": len(rows),
                "true_change_pct": 100 * true.sum() / base_total,
                "predicted_change_pct": 100 * pred.sum() / base_total,
                "direction_right_pct": 100 * float(np.mean(np.sign(pred.round(6)) == np.sign(true.round(6)))),
                # Predicting "no change" is cautious; predicting the opposite direction is wrong.
                "opposite_direction_pct": 100 * float(np.mean(np.sign(pred.round(6)) * np.sign(true.round(6)) < 0)),
                "per_batch_corr": float(np.corrcoef(pred, true)[0, 1]) if pred.std() > 0 and true.std() > 0 else float("nan"),
            })
    return pd.DataFrame(out)


def compare_sweeps(nursery, batches: list[dict], df: pd.DataFrame, service, test_from: str,
                   sample: int = 60, seed: int = 0) -> pd.DataFrame:
    """Sweep every lever through the real scenario engine and compare with the true curves."""
    by_id = {b["batch_id"]: b for b in batches}
    test = df[df["start_date"] >= test_from]
    test = test.sample(min(sample, len(test)), random_state=seed)
    out = []
    for output, use_case in MODELS.items():
        info = service.describe(use_case)
        order = info["feature_schema"]["x-feature-order"]
        level = float(test[output].abs().mean())
        for lever in info["scenario"]["levers"]:
            real, invented = [], []
            for _, row in test.iterrows():
                if lever["name"] == "light_hours" and row["lamp_type"] == "none":
                    continue
                features = {k: (row[k].item() if hasattr(row[k], "item") else row[k]) for k in order}
                body = service.sweep(use_case, {"features": features}, lever["name"])
                values = [p["value"] for p in body["points"]]
                predicted = np.array([p["outputs"][output]["value"] for p in body["points"]])
                b = by_id[row["batch_id"]]
                true = np.array([nursery.counterfactual(b, **{lever["name"]: v})[output] for v in values])
                if np.ptp(true) >= 0.02 * level:  # the lever really matters for this batch
                    if np.ptp(predicted) > 0:
                        real.append(np.corrcoef(predicted, true)[0, 1])
                    else:
                        real.append(0.0)  # a flat prediction where the truth moves
                else:
                    invented.append(np.ptp(predicted) / level * 100)
            out.append({
                "output": output, "lever": lever["name"],
                "batches_with_real_effect": len(real),
                "curve_correlation": float(np.mean(real)) if real else float("nan"),
                "batches_without_effect": len(invented),
                "invented_effect_pct_of_level": float(np.mean(invented)) if invented else float("nan"),
            })
    return pd.DataFrame(out)


VERDICT_SCENARIOS = {"+1 °C": ("temp_setpoint_c", 1.0), "-1 °C": ("temp_setpoint_c", -1.0),
                     "+2 h light": ("light_hours", 2.0), "+200 ppm CO2": ("co2_ppm", 200.0),
                     "+5 plants/m2": ("plant_density_m2", 5.0)}


def compare_verdicts(nursery, batches: list[dict], df: pd.DataFrame, service, test_from: str) -> pd.DataFrame:
    """Is the margin rule of ticket 5 trustworthy? Verdicts from the real engine vs. the answer key."""
    by_id = {b["batch_id"]: b for b in batches}
    test = df[df["start_date"] >= test_from]
    rows = []
    for output, use_case in MODELS.items():
        info = service.describe(use_case)
        order = info["feature_schema"]["x-feature-order"]
        levers = {l["name"]: l for l in info["scenario"]["levers"]}
        for _, row in test.iterrows():
            features = {k: (row[k].item() if hasattr(row[k], "item") else row[k]) for k in order}
            requested = []
            for name, (lever, delta) in VERDICT_SCENARIOS.items():
                new = features[lever] + delta
                if lever == "light_hours" and row["lamp_type"] == "none":
                    continue
                if levers[lever]["min"] <= new <= levers[lever]["max"]:
                    requested.append({"name": name, "changes": {lever: {"delta": delta}}})
            body = service.run_scenarios(use_case, {"features": features}, requested)
            b = by_id[row["batch_id"]]
            base_true = nursery.simulate(b)[output]
            for s in body["scenarios"]:
                lever, delta = VERDICT_SCENARIOS[s["name"]]
                true = nursery.counterfactual(b, **{lever: b[lever] + delta})[output] - base_true
                d = s["difference"][output]
                rows.append({"output": output, "clear": d["exceeds_margin"],
                             "same_direction": np.sign(round(d["change"], 6)) == np.sign(round(true, 6)),
                             "truth_small": abs(true) < d["margin"]})
    r = pd.DataFrame(rows)
    out = []
    for (output, clear), g in r.groupby(["output", "clear"]):
        out.append({"output": output, "verdict": "clear effect" if clear else "within margin", "cases": len(g),
                    "right_direction_pct": 100 * g["same_direction"].mean(),
                    "truth_also_small_pct": 100 * g["truth_small"].mean()})
    return pd.DataFrame(out)


def compare_certainty(nursery, batches: list[dict], df: pd.DataFrame, service, test_from: str,
                      sample: int = 60, seed: int = 0) -> pd.DataFrame:
    """Is a scenario with more comparable batches really more accurate? Sweeps vs. the answer key."""
    by_id = {b["batch_id"]: b for b in batches}
    test = df[df["start_date"] >= test_from].sample(min(sample, len(df)), random_state=seed)
    rows = []
    for output, use_case in MODELS.items():
        info = service.describe(use_case)
        order = info["feature_schema"]["x-feature-order"]
        level = float(test[output].abs().mean())
        for lever in info["scenario"]["levers"]:
            for _, row in test.iterrows():
                if lever["name"] == "light_hours" and row["lamp_type"] == "none":
                    continue
                features = {k: (row[k].item() if hasattr(row[k], "item") else row[k]) for k in order}
                body = service.sweep(use_case, {"features": features}, lever["name"])
                b = by_id[row["batch_id"]]
                base_true = nursery.simulate(b)[output]
                for p in body["points"]:
                    true = nursery.counterfactual(b, **{lever["name"]: p["value"]})[output] - base_true
                    rows.append({"output": output, "level": p["certainty"]["level"],
                                 "cases": p["certainty"]["comparable_cases"],
                                 "effect_error_pct": abs(p["difference"][output]["change"] - true) / level * 100})
    r = pd.DataFrame(rows)
    order = {"low": 0, "medium": 1, "high": 2}
    out = r.groupby(["output", "level"]).effect_error_pct.agg(points="count", mean="mean", median="median").reset_index()
    return out.sort_values(["output", "level"], key=lambda s: s.map(order) if s.name == "level" else s)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weather", choices=["knmi", "synthetic"], default="knmi")
    parser.add_argument("--test-from", default=None, help="default: the test split of the trained model")
    parser.add_argument("--compartments", type=int, default=16, help="must match make_nursery.py")
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--sweeps", action="store_true", help="also compare whole lever sweeps (slower)")
    parser.add_argument("--sample", type=int, default=60, help="test batches used for --sweeps")
    parser.add_argument("--verdicts", action="store_true", help="check the margin rule (clear effect or not)")
    parser.add_argument("--certainty", action="store_true", help="check that more comparable batches = smaller error")
    args = parser.parse_args()

    daily = (knmi.load_daily(nursery_mod.STATION, "2015-06-01", nursery_mod.END) if args.weather == "knmi"
             else knmi.synthetic_daily("2015-06-01", nursery_mod.END))
    df, nursery, batches = nursery_mod.generate(daily, args.compartments)
    registry = bootstrap.build_registry()
    split = registry.resolve("nursery").descriptor["split"]
    test_from = args.test_from or sorted(df["start_date"])[split["train"] + split["validation"]]
    result = compare(nursery, batches, df, registry, test_from)
    pd.set_option("display.width", 160)
    print(f"test batches from {test_from}\n")
    print(result.round(1).to_string(index=False))
    if args.json:
        args.json.write_text(result.to_json(orient="records", indent=2), encoding="utf-8")
    if args.certainty:
        result_c = compare_certainty(nursery, batches, df, bootstrap.build_service(), test_from, args.sample)
        print("\nerror of the predicted scenario effect (% of the normal level) by certainty\n")
        print(result_c.round(1).to_string(index=False))
    if args.verdicts:
        verdicts = compare_verdicts(nursery, batches, df, bootstrap.build_service(), test_from)
        print("\nverdicts of the scenario engine vs. the answer key\n")
        print(verdicts.round(1).to_string(index=False))
    if args.sweeps:
        sweeps = compare_sweeps(nursery, batches, df, bootstrap.build_service(), test_from, args.sample)
        print("\nwhole-range sweeps through the scenario engine vs. the true curves\n")
        print(sweeps.round(2).to_string(index=False))
        if args.json:
            args.json.with_name(args.json.stem + "_sweeps.json").write_text(
                sweeps.to_json(orient="records", indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
