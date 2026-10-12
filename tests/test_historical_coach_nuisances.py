"""Fabricated-only nuisance fitting, post-snap isolation and chronology tests."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.historical_coach_nuisances import (
    ACTION_ORDER, DEV_TRAIN_END, FINAL_FIT_END, INTERACTIONS,
    certified_prespnap_frame, development_metrics, fit_nuisances,
    predict_policy_nuisances,
)
from src.historical_fourth_down_cohort import (
    FEATURE_COLUMNS, PRESNAP_COLUMNS,
)

ROOT = Path(__file__).resolve().parents[1]


def sample(rows=80, *, phase="train"):
    records = []
    for i in range(rows):
        site = ("HOME", "AWAY", "NEUTRAL")[i%3]
        record = {
            "game_id": f"fabricated_{phase}_game_{i//2}",
            "play_id": str(i+100),
            "season": (2014+i%9 if phase == "train"
                       else 2023+i%2 if phase == "valid" else 2025),
            "week": 1+i%17,
            "quarter": 4 if i%2 else 2,
            "game_seconds_remaining": 180+i%140 if i%2 else 2100+i%140,
            "yardline_100": float(5+i%90),
            "ydstogo": float(1+i%4),
            "goal_to_go": int(i%6 == 0),
            "score_differential": float(i%17-8),
            "posteam_timeouts_remaining": i%4,
            "defteam_timeouts_remaining": (i+1)%4,
            "site_advantage": {"HOME":1, "AWAY":-1, "NEUTRAL":0}[site],
            "site": site, "feature_asof": "pre_snap",
            "observed_action": ACTION_ORDER[i%4],
            "offense_win": int(i%7 in (0, 2, 3)),
        }
        records.append(record)
    return pd.DataFrame.from_records(records)


def test_propensity_and_conditional_reward_are_separate_models():
    f = sample()
    model = fit_nuisances(f, end_year=DEV_TRAIN_END)
    pre = f[PRESNAP_COLUMNS].copy()
    observed = f["observed_action"].tolist()
    out = predict_policy_nuisances(model, pre, observed)
    assert model.training_end_year == 2022
    assert list(model.feature_names) == list(FEATURE_COLUMNS)
    assert list(model.actions) == list(ACTION_ORDER)
    assert len(out) == len(f)
    assert out["policy_action"].tolist() == observed
    for col in ("propensity_policy", "mu_policy"):
        assert out[col].between(0, 1).all()
    distribution = out[[f"coach_probability_{a}" for a in ACTION_ORDER]]
    assert np.allclose(distribution.sum(axis=1), 1., atol=1e-8)
    assert "observed_action" not in out
    assert "offense_win" not in out
    assert len(INTERACTIONS) > 0


def test_no_observed_play_or_final_score_access_during_policy_predictions():
    f = sample()
    fitted = fit_nuisances(f, end_year=DEV_TRAIN_END)
    pre = f[PRESNAP_COLUMNS].copy()
    hypothetical = ["NORMAL_GO_PASS"] * len(pre)
    baseline = predict_policy_nuisances(fitted, pre, hypothetical)
    mutated = f.copy(deep=True)
    mutated["observed_action"] = "FIELD_GOAL"
    mutated["offense_win"] = 1
    same = predict_policy_nuisances(
        fitted, mutated[PRESNAP_COLUMNS], hypothetical
    )
    pd.testing.assert_frame_equal(baseline, same)
    changed = predict_policy_nuisances(
        fitted, pre, ["PUNT"] * len(pre)
    )
    assert (changed["policy_action"] == "PUNT").all()
    assert not np.allclose(
        baseline["propensity_policy"], changed["propensity_policy"]
    )
    # No future action or post-snap label allowed in prediction schema.
    with pytest.raises(ValueError, match="EXACT feature-only schema"):
        predict_policy_nuisances(fitted, mutated, hypothetical)


def test_training_dev_period_and_final_2025_only():
    f = sample()
    f.loc[0, "season"] = 2026
    with pytest.raises(ValueError, match="2014-22 or 2014-25"):
        fit_nuisances(f, end_year=2022)
    f = sample()
    with pytest.raises(ValueError, match="2014-22 or 2014-25"):
        fit_nuisances(f, end_year=2023)
    combined = pd.concat([sample(), sample(16, phase="valid"),
                          sample(12, phase="final")], ignore_index=True)
    final = fit_nuisances(combined, end_year=FINAL_FIT_END)
    assert final.training_end_year == 2025
    assert set(final.propensity.classes_) == set(ACTION_ORDER)


def test_development_evaluation_never_scores_2025_and_has_action_breakdowns():
    fit = fit_nuisances(sample(), 2022)
    val = sample(32, phase="valid")
    report = development_metrics(fit, val)
    assert report["dev_decisions"] == 32
    assert report["dev_distinct_games"] == 16
    assert 0 <= report["factual_reward_win_brier"] <= 1
    assert 0 <= report["observed_coach_action_log_loss"]
    assert set(report["by_factual_action"]) == set(ACTION_ORDER)
    for action in ACTION_ORDER:
        assert report["by_factual_action"][action]["decisions"] == 8
    val.loc[0, "season"] = 2025
    with pytest.raises(ValueError, match="2023-24"):
        development_metrics(fit, val)


@pytest.mark.parametrize("field,value", [
    ("site", "UNKNOWN"),
    ("feature_asof", "after_play"),
    ("site_advantage", 20),
])
def test_invalid_prespnap_feature_context_rejected(field, value):
    f = sample()
    f.loc[0, field] = value
    with pytest.raises(ValueError, match="site|pre_snap|Site"):
        certified_prespnap_frame(f[PRESNAP_COLUMNS])


def test_missing_action_class_fails_closed_instead_of_fabricated_probability():
    f = sample()
    f = f.loc[f["observed_action"] != "PUNT"].copy()
    with pytest.raises(ValueError, match="all historical actions"):
        fit_nuisances(f, end_year=DEV_TRAIN_END)


def test_unobserved_win_class_fails_closed_instead_of_dummy_reward_fit():
    f = sample()
    f["offense_win"] = 1
    with pytest.raises(ValueError, match="both final win classes"):
        fit_nuisances(f, end_year=DEV_TRAIN_END)


def test_project_extractor_and_model_sources_use_only_pre_snap_whitelist():
    source = (ROOT / "scripts/train_2027_historical_nuisances.py").read_text()
    assert "data/fourth_downs_labeled.parquet" in source
    assert "nfl.load_pbp" not in source
    assert "range(2026" not in source
    assert "models/win_probability_model.joblib" not in source
    assert "joblib.dump(final.propensity" in source
    assert "joblib.dump(final.reward" in source
    assert "fourth_down_policy_study(" not in source
