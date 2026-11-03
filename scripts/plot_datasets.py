"""Charts for docs/datasets.md (ticket 1).

    python scripts/plot_datasets.py --scenarios-small s16.json --scenarios-large s48.json

The scenario JSON files come from `scripts/check_scenarios.py --json ...`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

OUT = Path("docs/img")
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10.5, "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.facecolor": SURFACE, "figure.facecolor": SURFACE,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.axisbelow": True, "text.color": INK,
})

KEY_ROWS = [
    ("temp_setpoint_c +1", "weeks_to_harvest", "+1 °C → weeks to harvest"),
    ("co2_ppm +200", "weeks_to_harvest", "+200 ppm CO₂ → weeks to harvest"),
    ("light_hours +2", "weeks_to_harvest", "+2 h light → weeks to harvest"),
    ("plant_density_m2 +5", "plants_sold", "+5 plants/m² → plants sold"),
    ("light_hours +2", "gas_m3", "+2 h light → gas"),
    ("co2_ppm +200", "gas_m3", "+200 ppm CO₂ → gas"),
    ("temp_setpoint_c +1", "gas_m3", "+1 °C → gas"),
]


def title(fig, text, sub):
    fig.text(0.012, 0.975, text, fontsize=14, fontweight="bold", va="top", color=INK)
    fig.text(0.012, 0.915, sub, fontsize=10.5, va="top", color=MUTED)


def scenario_chart(small: Path, large: Path) -> Path:
    panels = [(json.loads(small.read_text()), "711 batches (16 compartments)"),
              (json.loads(large.read_text()), "2,122 batches (48 compartments)")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6), sharey=True)
    y = np.arange(len(KEY_ROWS))[::-1]
    h = 0.36
    for ax, (rows, label) in zip(axes, panels):
        lookup = {(r["scenario"], r["output"]): r for r in rows}
        true = [lookup[(s, o)]["true_change_pct"] for s, o, _ in KEY_ROWS]
        pred = [lookup[(s, o)]["predicted_change_pct"] for s, o, _ in KEY_ROWS]
        ax.barh(y + h / 2, true, h - 0.04, color=BLUE, label="True effect (simulator)")
        ax.barh(y - h / 2, pred, h - 0.04, color=ORANGE, label="Model prediction")
        for yy, t, p in zip(y, true, pred):
            for value, offset in ((t, h / 2), (p, -h / 2)):
                ax.text(value + (0.6 if value >= 0 else -0.6), yy + offset, f"{value:+.1f}%", va="center",
                        ha="left" if value >= 0 else "right", fontsize=9, color=MUTED)
        ax.axvline(0, color=MUTED, linewidth=1)
        ax.set_title(label, fontsize=11, color=INK, loc="left", pad=8)
        ax.set_xlim(-33, 14)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("change vs. the same batches without the scenario (%)")
    axes[0].set_yticks(y, [name for _, _, name in KEY_ROWS])
    axes[0].legend(loc="lower left", frameon=False, fontsize=9.5)
    title(fig, "Does the model learn the true cause and effect?",
          "Test batches (Aug 2023 - 2025) the model never saw. Direct growth effects: yes. Indirect gas effects: not yet.")
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    path = OUT / "scenario_answer_key.png"
    fig.savefig(path, dpi=150)
    return path


def arable_chart(registry_dir: Path) -> Path:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from predictor.adapters.fs_registry import FilesystemRegistry
    from predictor.adapters.onnx_runner import OnnxRunner
    from predictor.core.validation import FeatureValidator

    df = pd.read_csv("data/arable.csv")
    model = FilesystemRegistry(registry_dir, {"onnx": OnnxRunner}).resolve("arable-yield")
    validator = FeatureValidator(model.feature_schema, reject_unknown_fields=False)
    test = df[df["year"] >= 2021].copy()
    rows = [{k: r[k] for k in validator.feature_order} for r in test.to_dict("records")]
    test["pred"] = model.runner.predict(validator.to_matrix(rows)).values
    test["e_model"] = (test["pred"] - test["yield_t_ha"]).abs() / test["yield_t_ha"] * 100
    test["e_guess"] = (test["yield_5y_avg"] - test["yield_t_ha"]).abs() / test["yield_t_ha"] * 100
    g = test.groupby("crop")[["e_guess", "e_model"]].mean().sort_values("e_guess")
    names = {"winter_wheat": "Winter wheat", "ware_potatoes": "Ware potatoes", "sugar_beet": "Sugar beet",
             "onions": "Onions", "silage_maize": "Silage maize"}
    fig, ax = plt.subplots(figsize=(10, 4.6))
    y = np.arange(len(g))
    h = 0.36
    ax.barh(y + h / 2, g["e_guess"], h - 0.04, color=BLUE, label='Simple guess: "same as the last 5 years"')
    ax.barh(y - h / 2, g["e_model"], h - 0.04, color=ORANGE, label="Model with growing-season weather")
    for yy, a, b in zip(y, g["e_guess"], g["e_model"]):
        ax.text(a + 0.3, yy + h / 2, f"{a:.1f}%", va="center", fontsize=9, color=MUTED)
        ax.text(b + 0.3, yy - h / 2, f"{b:.1f}%", va="center", fontsize=9, color=MUTED)
    ax.set_yticks(y, [names[c] for c in g.index])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("average error, % of the real yield (2021-2025, lower is better)")
    ax.legend(loc="lower right", frameon=False, fontsize=9.5)
    title(fig, "Real data reality check: arable yields 2021-2025",
          "Trained on 1999-2020 (CBS + KNMI). Province-level weather does not beat the 5-year average yet.")
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    path = OUT / "arable_backtest.png"
    fig.savefig(path, dpi=150)
    return path


def nursery_chart() -> Path:
    df = pd.read_csv("data/nursery.csv")
    df["year"] = df["start_date"].str[:4].astype(int)
    y = df.groupby("year").agg(price=("gas_price_eur_m3", "mean"), temp=("temp_setpoint_c", "mean"),
                               weeks=("weeks_to_harvest", "mean"))
    fig, axes = plt.subplots(3, 1, figsize=(10, 6.4), sharex=True)
    specs = [("price", "gas price (€/m³)", "{:.2f}", BLUE), ("temp", "temperature setpoint (°C)", "{:.1f}", ORANGE),
             ("weeks", "weeks to harvest", "{:.1f}", AQUA)]
    for ax, (col, label, fmt, color) in zip(axes, specs):
        ax.plot(y.index, y[col], color=color, linewidth=2, marker="o", markersize=5)
        peak = y[col].idxmax() if col != "temp" else y[col].idxmin()
        ax.annotate(fmt.format(y.loc[peak, col]), (peak, y.loc[peak, col]), textcoords="offset points",
                    xytext=(8, 0), va="center", fontsize=9, color=MUTED)
        ax.set_ylabel(label, rotation=0, ha="left", va="center", labelpad=0)
        ax.yaxis.set_label_coords(1.01, 0.5)
        ax.margins(y=0.25)
    axes[-1].set_xticks(y.index)
    title(fig, "Realistic behaviour: the 2022 gas crisis in the simulated nursery",
          "Growers turn the heating down when gas is expensive, so their batches take longer (averages per start year).")
    fig.tight_layout(rect=(0, 0, 0.84, 0.88))
    path = OUT / "nursery_gas_crisis.png"
    fig.savefig(path, dpi=150)
    return path


CONSTRAINED_ROWS = [
    ("plant_density_m2 +5", "gas_m3", "+5 plants/m² → gas"),
    ("co2_ppm +200", "weeks_to_harvest", "+200 ppm CO₂ → weeks to harvest"),
    ("light_hours +2", "weeks_to_harvest", "+2 h light → weeks to harvest"),
    ("co2_ppm +200", "gas_m3", "+200 ppm CO₂ → gas"),
    ("light_hours +2", "gas_m3", "+2 h light → gas"),
]


def constraints_chart(before: Path, after: Path) -> Path:
    b = {(r["scenario"], r["output"]): r for r in json.loads(before.read_text())}
    a = {(r["scenario"], r["output"]): r for r in json.loads(after.read_text())}
    fig, ax = plt.subplots(figsize=(11, 5.4))
    y = np.arange(len(CONSTRAINED_ROWS))[::-1] * 1.25
    h = 0.32
    series = [("True effect (simulator)", BLUE, [b[(s, o)]["true_change_pct"] for s, o, _ in CONSTRAINED_ROWS]),
              ("Model without declared effects", ORANGE, [b[(s, o)]["predicted_change_pct"] for s, o, _ in CONSTRAINED_ROWS]),
              ("Model with declared effects", AQUA, [a[(s, o)]["predicted_change_pct"] for s, o, _ in CONSTRAINED_ROWS])]
    for i, (label, color, values) in enumerate(series):
        offset = (1 - i) * h
        ax.barh(y + offset, values, h - 0.04, color=color, label=label)
        for yy, v in zip(y, values):
            ax.text(v + (0.5 if v >= 0 else -0.5), yy + offset, f"{v:+.1f}%", va="center",
                    ha="left" if v >= 0 else "right", fontsize=9, color=MUTED)
    ax.axvline(0, color=MUTED, linewidth=1)
    ax.set_yticks(y, [name for _, _, name in CONSTRAINED_ROWS])
    ax.grid(axis="y", visible=False)
    ax.set_xlim(-31, 6)
    ax.set_xlabel("change vs. the same batches without the scenario (%)")
    ax.legend(loc="upper left", frameon=False, fontsize=9.5)
    title(fig, "Declaring the direction of a lever makes scenarios more trustworthy",
          "Unseen test batches. With declared effects the model never goes the wrong way (density → gas was wrong before).")
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    path = OUT / "scenario_constraints.png"
    fig.savefig(path, dpi=150)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios-small", type=Path)
    parser.add_argument("--scenarios-large", type=Path)
    parser.add_argument("--constraints", type=Path, nargs=2, metavar=("BEFORE", "AFTER"),
                        help="check_scenarios.py JSON without and with declared effects")
    parser.add_argument("--models", type=Path, default=Path("models"))
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.constraints:
        print("wrote", constraints_chart(*args.constraints))
    if args.scenarios_small and args.scenarios_large:
        for path in (scenario_chart(args.scenarios_small, args.scenarios_large), arable_chart(args.models), nursery_chart()):
            print("wrote", path)


if __name__ == "__main__":
    main()
