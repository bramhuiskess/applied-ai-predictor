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

MODELS = {"weeks_to_harvest": "nursery-harvest-weeks", "plants_sold": "nursery-plants-sold", "gas_m3": "nursery-gas"}


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
            model = registry.resolve(use_case)
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weather", choices=["knmi", "synthetic"], default="knmi")
    parser.add_argument("--test-from", default=None, help="default: the test split of the trained model")
    parser.add_argument("--compartments", type=int, default=16, help="must match make_nursery.py")
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--sweeps", action="store_true", help="also compare whole lever sweeps (slower)")
    parser.add_argument("--sample", type=int, default=60, help="test batches used for --sweeps")
    args = parser.parse_args()

    daily = (knmi.load_daily(nursery_mod.STATION, "2015-06-01", nursery_mod.END) if args.weather == "knmi"
             else knmi.synthetic_daily("2015-06-01", nursery_mod.END))
    df, nursery, batches = nursery_mod.generate(daily, args.compartments)
    registry = bootstrap.build_registry()
    split = registry.resolve("nursery-gas").descriptor["split"]
    test_from = args.test_from or sorted(df["start_date"])[split["train"] + split["validation"]]
    result = compare(nursery, batches, df, registry, test_from)
    pd.set_option("display.width", 160)
    print(f"test batches from {test_from}\n")
    print(result.round(1).to_string(index=False))
    if args.json:
        args.json.write_text(result.to_json(orient="records", indent=2), encoding="utf-8")
    if args.sweeps:
        sweeps = compare_sweeps(nursery, batches, df, bootstrap.build_service(), test_from, args.sample)
        print("\nwhole-range sweeps through the scenario engine vs. the true curves\n")
        print(sweeps.round(2).to_string(index=False))
        if args.json:
            args.json.with_name(args.json.stem + "_sweeps.json").write_text(
                sweeps.to_json(orient="records", indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
