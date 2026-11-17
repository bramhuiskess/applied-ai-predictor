"""Replaying a real season: what the grower knew on the decision day, and how the forecast is scored."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import replay_real_season as replay  # noqa: E402


def real_days(n=170):
    index = pd.date_range("2019-12-16", periods=n)
    radiation = np.linspace(200, 2000, n)
    return pd.DataFrame({"temp": np.where(np.arange(n) < 56, 21.0, 23.0), "radiation": radiation, "lamp_hours": 12.0,
                         "heat": 2.0, "electricity": 1.5, "kg": 0.1}, index=index)


def test_the_model_gets_only_what_was_known_on_the_decision_day():
    days = real_days()
    known = replay.what_the_grower_knew(days, 56, {"harvest_5_months_kg_m2": 12.0}, days.index[56])
    assert known["days_since_planting"] == 56 and known["decision_week"] == days.index[56].isocalendar().week
    assert known["temp_so_far_c"] == 21.0 and known["temp_plan_c"] == 23.0     # before and after the decision
    assert known["heat_last_4_weeks_mj_m2"] == pytest.approx(28 * 2.0)
    assert known["usual_harvest_5_months_kg_m2"] == 12.0 and known["tomato_price_eur_kg"] == 3.0
    # real lamp hours become "hours on dark days": more than the average hours once the sun takes over
    assert known["lamp_hours_plan"] > 12.0


def test_before_planting_there_is_no_log_so_the_first_two_weeks_stand_in():
    known = replay.what_the_grower_knew(real_days(), 0, {}, pd.Timestamp("2019-12-16"))
    assert known["heat_last_4_weeks_mj_m2"] == pytest.approx(28 * 2.0) and known["temp_so_far_c"] == 21.0


def test_the_score_is_a_share_of_what_was_really_measured():
    r = pd.DataFrame({"output": ["harvest"] * 3, "actual": [0.0, 10.0, 10.0], "forecast": [1.0, 11.0, 9.0],
                      "plain_guess": [0.0, 15.0, 15.0], "inside_range": [False, True, True]})
    table = replay.summary(r)
    assert table.loc["harvest", "error_pct"] == pytest.approx(15.0)             # 3 off on 20, also with a month of no harvest
    assert table.loc["harvest", "bias_pct"] == pytest.approx(5.0) and table.loc["harvest", "plain_guess_error_pct"] == pytest.approx(50.0)
    assert table.loc["harvest", "inside_80_range_pct"] == pytest.approx(66.67, abs=0.01)
