"""How far can a grower trust the real-greenhouse models? Every greenhouse-day predicted out of sample.

    python scripts/check_greenhouse.py                 # heating
    python scripts/check_greenhouse.py --use-case greenhouse-electricity --chart docs/img/x.png

Two situations, next to plain guesses, per day, per week and over the season:

* **own history**: the model knows the greenhouse, the 2-week block it predicts was held out
  (every block is held out once, so every day gets an out-of-sample prediction);
* **new greenhouse**: the whole greenhouse was held out.

It uses the same inputs, declared effects and candidate models as `predictor train`, read from the
manifest, so the numbers describe the product and not a separate experiment.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone

from predictor.core.manifest import MANIFEST_FILENAME, load_manifest
from predictor.trainer import models
from predictor.trainer.profile import prepare
from predictor.trainer.run import _load_table, _scenario_constraints, category_counts

ROOT = Path(__file__).resolve().parents[1]


def out_of_sample(pipeline, X: pd.DataFrame, y: pd.Series, folds: np.ndarray) -> np.ndarray:
    """A prediction for every row from a model that did not see that row's fold."""
    pred = np.zeros(len(X))
    for fold in np.unique(folds):
        held = folds == fold
        model = clone(pipeline).fit(X[~held].to_numpy(dtype="float32"), y[~held])
        pred[held] = model.predict(X[held].to_numpy(dtype="float32"))
    return np.clip(pred, 0, None)


def scores(frame: pd.DataFrame, pred: np.ndarray) -> pd.DataFrame:
    """Per greenhouse: daily error and R2, weekly error (% of a normal week) and the season total."""
    rows = {}
    for group, t in frame.assign(pred=pred).dropna(subset=["pred"]).groupby("group"):
        week = t.groupby(t["time"].dt.to_period("W")).agg(y=("y", "sum"), pred=("pred", "sum"), n=("y", "size"))
        week = week[week["n"] == 7]
        spread = ((t["y"] - t["y"].mean()) ** 2).sum()
        rows[group] = {"mae": (t["pred"] - t["y"]).abs().mean(),
                       "r2": 1 - ((t["pred"] - t["y"]) ** 2).sum() / spread if spread else np.nan,
                       "week_pct": (week["pred"] - week["y"]).abs().mean() / week["y"].mean() * 100 if len(week) else np.nan,
                       "season_pct": abs(t["pred"].sum() / t["y"].sum() - 1) * 100}
    return pd.DataFrame(rows).T


def evaluate(use_case: str) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, str]]:
    manifest = load_manifest(ROOT / "usecases" / use_case / MANIFEST_FILENAME)
    cfg = manifest.training
    if not (cfg.test_blocks and cfg.test_blocks.group):
        raise SystemExit(f"{use_case}: needs training.test_blocks with a 'group' column")
    data = prepare(_load_table(manifest, None).df, cfg, manifest.policy)
    monotone = _scenario_constraints(manifest, data, cfg.target)
    candidates = models.candidates("regression", cfg.anomaly_share, monotone, categories=category_counts(data.features))
    frame = pd.DataFrame({"group": data.groups, "time": data.times, "y": data.y})
    blocks = ((data.times - data.times.min()).dt.days // cfg.test_blocks.days).to_numpy() % cfg.test_blocks.every

    results, chosen = {}, {}
    for situation, folds in [("own history", blocks), ("new greenhouse", data.groups.to_numpy())]:
        by_model = {name: scores(frame, out_of_sample(p, data.X, data.y, folds)) for name, p in candidates.items()}
        chosen[situation] = min(by_model, key=lambda n: by_model[n]["mae"].mean())
        results[situation] = by_model[chosen[situation]]
    # Plain guesses: what a grower can do without a model.
    week_ago = data.persistence.to_numpy(dtype=float) if data.persistence is not None else None
    if week_ago is not None:
        results[f"plain guess: {cfg.baseline.label}"] = scores(frame, week_ago)
    results["plain guess: the average day"] = scores(frame, np.full(len(frame), data.y.mean()))

    summary = pd.DataFrame({name: {"daily error": r["mae"].mean(), "daily R2 (median)": r["r2"].median(),
                                   "weekly error % (median)": r["week_pct"].median(),
                                   "weekly error % (worst)": r["week_pct"].max(),
                                   "season total % (median)": r["season_pct"].median(),
                                   "season total % (worst)": r["season_pct"].max()} for name, r in results.items()}).T
    return summary, results, chosen


def chart(results: dict[str, pd.DataFrame], use_case: str, unit: str, path: Path) -> None:
    import matplotlib.pyplot as plt

    import plot_datasets as P

    names = list(results)
    colours = [P.BLUE, P.ORANGE] + [P.MUTED] * (len(names) - 2)
    panels = [("week_pct", "Weekly total: typical error (%)"), ("season_pct", "Season total: error (%)"),
              ("mae", f"Per day: typical error ({unit})")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.9))
    for ax, (column, label) in zip(axes, panels):
        values = [results[n][column].median() if column != "mae" else results[n][column].mean() for n in names]
        bars = ax.barh(range(len(names))[::-1], values, color=colours, height=0.6)
        for bar, v in zip(bars, values):
            ax.text(bar.get_width() * 1.02, bar.get_y() + bar.get_height() / 2, f"{v:.2f}" if column == "mae" else f"{v:.0f}%",
                    va="center", fontsize=10, color=P.INK)
        ax.set_yticks(range(len(names))[::-1], names if ax is axes[0] else [""] * len(names))
        ax.set_title(label, loc="left", fontsize=11, color=P.INK)
        ax.set_xlim(0, max(values) * 1.18)
        ax.grid(axis="y", visible=False)
    P.title(fig, f"{use_case}: with the greenhouse's own history, for a new greenhouse, and plain guesses",
            "Every greenhouse-day predicted by a model that did not see it (12 real greenhouses, median over greenhouses). Lower is better.")
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--use-case", default="greenhouse-heating")
    parser.add_argument("--chart", type=Path, help="write a comparison chart to this path")
    parser.add_argument("--per-greenhouse", action="store_true", help="also print the scores of every greenhouse")
    args = parser.parse_args()
    warnings.filterwarnings("ignore")
    summary, results, chosen = evaluate(args.use_case)
    print(f"{args.use_case}: best model with own history = {chosen['own history']}, "
          f"for a new greenhouse = {chosen['new greenhouse']}\n")
    print(summary.round(2).to_string())
    if args.per_greenhouse:
        for name in ("own history", "new greenhouse"):
            print(f"\n{name}\n{results[name].round(2).to_string()}")
    if args.chart:
        manifest = load_manifest(ROOT / "usecases" / args.use_case / MANIFEST_FILENAME)
        chart(results, args.use_case, manifest.training.units.get(manifest.training.target, ""), args.chart)
        print(f"\nwrote {args.chart}")


if __name__ == "__main__":
    main()
