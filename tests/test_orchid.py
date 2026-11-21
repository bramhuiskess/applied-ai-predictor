"""The pot-orchid batch: three phases built on published research, and the batches made with it."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from predictor import bootstrap
from predictor.trainer.run import train_use_case

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import knmi  # noqa: E402
import make_orchid  # noqa: E402
import orchid_crop as crop  # noqa: E402


def grow(**changes):
    return crop.grow(**{**crop.STANDARD, **changes})


# --------------------------------------------------------------------------- what is published

def test_the_usual_schedule_gives_the_usual_batch():
    batch = grow()
    assert batch.weeks_to_sale == pytest.approx(16 + 12 + 8 + 10)      # 28 weeks warm, 8 weeks at 19 C, 10 weeks at 21 C
    assert batch.spikes_per_plant == pytest.approx(1.92, abs=0.02)      # the trial's 1.92 spikes at 4 mol of light
    assert batch.sellable_share == 1.0 and batch.early_spike_share_pct == 0


def test_more_light_in_the_warm_phases_gives_more_spikes():
    spikes = [grow(light_mol=mol).spikes_per_plant for mol in (3, 4, 5)]
    assert spikes == sorted(spikes)
    assert spikes == pytest.approx([1.88, 1.92, 2.03], abs=0.07)        # the published trial


def test_29_degrees_prevents_flowering_and_26_induces_it():
    assert grow(temp_cooling_c=29).sellable_share == 0
    assert grow(temp_cooling_c=25).sellable_share == 1
    assert grow(temp_warm_c=25).early_spike_share_pct > 0 == grow(temp_warm_c=28).early_spike_share_pct   # too cool while raising


def test_eight_weeks_of_cooling_is_best_and_19_beats_22_degrees():
    shares = [grow(weeks_cooling=w).two_spike_share for w in (4, 6, 8, 10)]
    assert shares[0] < shares[1] < shares[2] == shares[3]
    assert grow(temp_cooling_c=19).two_spike_share > grow(temp_cooling_c=22).two_spike_share > grow(temp_cooling_c=24).two_spike_share


def test_a_bigger_plant_gives_more_two_spike_plants_and_31_degrees_more_leaves():
    assert grow(weeks_phase_2=16).two_spike_share > grow(weeks_phase_2=12).two_spike_share > grow(weeks_phase_2=8).two_spike_share
    assert grow(temp_warm_c=31).leaves_at_cooling > grow(temp_warm_c=28).leaves_at_cooling


def test_cooler_cooling_takes_longer_to_sale():
    assert grow(temp_cooling_c=17).weeks_to_sale > grow().weeks_to_sale > grow(temp_cooling_c=22).weeks_to_sale


def test_energy_follows_the_season_and_the_space_per_phase():
    days = np.arange(crop_days := 400)
    winter, summer = np.full(crop_days, 3.0), np.full(crop_days, 20.0)
    dark, bright = np.full(crop_days, 200.0), np.full(crop_days, 2000.0)
    plan = dict(crop.STANDARD, weeks_finishing=10)
    cold, warm = crop.energy(winter, dark, **plan), crop.energy(summer, bright, **plan)
    assert cold["heat_mj_per_plant"] > warm["heat_mj_per_plant"] and warm["cooling_kwh_per_plant"] > cold["cooling_kwh_per_plant"] == 0
    assert cold["space_m2_weeks_per_plant"] == pytest.approx(16 / 90 + 12 / 50 + 18 / 45)
    assert len(days) == crop_days


# --------------------------------------------------------------------------- the batches

@pytest.fixture(scope="module")
def daily():
    return knmi.synthetic_daily("2015-08-01", "2025-12-31", seed=4)


@pytest.fixture(scope="module")
def table(daily):
    return make_orchid.generate(daily, batches=1600, seed=9)


def test_one_row_is_one_batch(table):
    assert table.notna().all().all() and table["batch_id"].is_unique
    assert table["two_spike_share"].between(0, 1).all() and table["sellable_share"].between(0.9, 1).all()
    assert table["weeks_to_sale"].between(30, 62).all()
    for lever in make_orchid.LEVERS:
        assert table[lever].nunique() > 3                                  # growers try things


def test_the_answer_key_is_the_same_batch_with_another_schedule(table, daily):
    clean = daily.dropna(subset=["temp_mean_c", "radiation_j_cm2"]).reset_index(drop=True)
    row = table.iloc[100]
    base = make_orchid.counterfactual(row, clean)
    assert make_orchid.counterfactual(row, clean) == base                  # no luck in the answer key
    longer = make_orchid.counterfactual(row, clean, weeks_phase_2=row["weeks_phase_2"] + 2)
    assert longer["weeks_to_sale"] == pytest.approx(base["weeks_to_sale"] + 2)
    assert longer["space_m2_weeks_per_plant"] > base["space_m2_weeks_per_plant"] and longer["two_spike_share"] >= base["two_spike_share"]


def test_the_net_result_pays_plants_by_class(table):
    row = table.iloc[3]
    result = {name: float(row[name]) for name in make_orchid.OUTPUTS}
    expected = (row["sellable_share"] * row["price_plant_eur"] + row["two_spike_share"] * row["two_spike_premium_eur"]
                - row["heat_mj_per_plant"] * row["heat_price_eur_mj"] - row["cooling_kwh_per_plant"] * row["electricity_price_eur_kwh"]
                - row["space_m2_weeks_per_plant"] * row["space_cost_eur_m2_week"])
    assert make_orchid.net_eur(row, result) == pytest.approx(expected)
    more_spikes = dict(result, spikes_per_plant=result["spikes_per_plant"] + 1)
    assert make_orchid.net_eur(row, more_spikes) == pytest.approx(expected)   # a third spike is not paid extra


def test_trained_model_and_finder_follow_the_crop_model(table, daily, tmp_path):
    """End to end: train the real manifest, compare a scenario with the answer key, and let the finder plan a batch."""
    table.to_csv(tmp_path / "orchid_batches.csv", index=False)
    manifest = json.loads((ROOT / "usecases" / "orchid-batch" / "manifest.json").read_text())
    manifest["data_source"]["path"] = str(tmp_path / "orchid_batches.csv")
    directory = tmp_path / "usecases" / "orchid-batch"
    directory.mkdir(parents=True)
    (directory / "manifest.json").write_text(json.dumps(manifest))
    train_use_case(directory, bootstrap.build_registry(tmp_path / "models"))
    client = TestClient(bootstrap.build_app(tmp_path / "usecases", tmp_path / "models"))
    order = client.get("/models/orchid-batch").json()["feature_schema"]["x-feature-order"]
    clean = daily.dropna(subset=["temp_mean_c", "radiation_j_cm2"]).reset_index(drop=True)
    rows = table[(table["start_date"] >= "2023-01-01") & table["weeks_cooling"].between(6, 9)].head(30)
    true, predicted, gains = [], [], []
    for _, row in rows.iterrows():
        features = {k: float(row[k]) for k in order}
        body = client.post("/models/orchid-batch/scenario", json={"base": {"features": features}, "scenarios": [
            {"name": "shorter cooling", "changes": {"weeks_cooling": features["weeks_cooling"] - 2}}]}).json()
        predicted.append(body["scenarios"][0]["difference"]["two_spike_share"]["change"])
        base = make_orchid.counterfactual(row, clean)
        true.append(make_orchid.counterfactual(row, clean, weeks_cooling=row["weeks_cooling"] - 2)["two_spike_share"] - base["two_spike_share"])
        best = client.post("/models/orchid-batch/scenario/best", json={"base": {"features": features}}).json()["best"]
        if best:
            gains.append(make_orchid.net_eur(row, make_orchid.counterfactual(row, clean, **best["plan"])) - make_orchid.net_eur(row, base))
    assert np.mean(predicted) < 0 and np.mean(predicted) == pytest.approx(np.mean(true), rel=0.35)
    assert len(gains) >= 20 and np.mean(gains) > 0                          # following the advice pays, on the true outcomes
