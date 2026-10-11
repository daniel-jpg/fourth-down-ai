import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

source = Path(__file__).resolve().parents[1] / "src/61_audit_observed_kickoff_wp.py"
spec = importlib.util.spec_from_file_location("wp_kickoff_check", source)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


FEATURES = [
    "game_seconds_remaining", "half_seconds_remaining", "qtr", "down",
    "ydstogo", "yardline_100", "goal_to_go", "home_score_differential",
    "home_timeouts_remaining", "away_timeouts_remaining", "is_home_posteam",
    "is_kickoff", "is_overtime", "final_five_minutes",
]


def row(game, play, play_type, clock, *, receiver="SEA", own=None,
        touchback=0, desc="", result=7, receiver_score=20,
        opponent_score=17, goal_to_go=np.nan):
    return dict(
        season=2024, game_id=game, play_id=play, play_type=play_type,
        qtr=4, game_seconds_remaining=clock, half_seconds_remaining=clock,
        posteam=receiver, home_team="SEA", away_team="LAR",
        result=result, posteam_score=receiver_score,
        defteam_score=opponent_score,
        posteam_timeouts_remaining=2, defteam_timeouts_remaining=1,
        down=1 if own is not None else np.nan,
        ydstogo=10 if own is not None else np.nan,
        yardline_100=100-own if own is not None else np.nan,
        goal_to_go=goal_to_go, touchback=touchback, desc=desc,
    )


def pbp():
    return pd.DataFrame([
        row("G1", 1, "kickoff", 300, touchback=1),
        row("G1", 2, "run", 300, own=30),
        row("G1", 3, "pass", 288, own=40),
        row("G1", 4, "kickoff", 200, receiver="LAR",
            receiver_score=21, opponent_score=20),
        row("G1", 5, "run", 194, receiver="LAR", own=29,
            receiver_score=21, opponent_score=20),
        row("G2", 1, "kickoff", 100, desc="onside kick",
            receiver="SEA", touchback=0),
        row("G2", 2, "run", 95, own=50),
        row("G3", 1, "kickoff", 160, touchback=0),
        row("G3", 2, "run", 150, receiver="LAR", own=65),
    ])


class FakeModel:
    classes_ = np.array([0, 2])

    def predict_proba(self, X):
        # Test only: native missing versus engine kickoff goal_to_go=0.
        pos = X[:, FEATURES.index("home_score_differential")]
        kicked = X[:, FEATURES.index("is_kickoff")]
        goal = X[:, FEATURES.index("goal_to_go")]
        marker = np.where(np.isnan(goal), .06, 0.)
        win = np.clip(.43 + .01*pos + .04*kicked + marker, .01, .99)
        return np.column_stack((1-win, win))


def test_first_snap_pairing_excludes_onside_and_changed_receiver():
    pairs = mod.pair_2024_q4(pbp())
    assert pairs["game_id"].tolist() == ["G1", "G1"]
    assert pairs["play_id_kick"].tolist() == [1, 4]
    assert pairs["play_id_snap"].tolist() == [2, 5]


def test_score_and_timeout_axes_are_correct_for_home_and_away_receiver():
    pairs = mod.pair_2024_q4(pbp())
    k = mod.feature_frame(pairs, "kick")
    s = mod.feature_frame(pairs, "snap")
    assert k["is_home_posteam"].tolist() == [1.0, 0.0]
    assert k["home_score_differential"].tolist() == [3.0, -1.0]
    assert k["home_timeouts_remaining"].tolist() == [2.0, 1.0]
    assert k["away_timeouts_remaining"].tolist() == [1.0, 2.0]
    assert np.isnan(k["down"]).all()
    assert s["is_kickoff"].tolist() == [0.0, 0.0]


def test_native_and_synthetic_kickoffs_can_differ():
    pairs = mod.pair_2024_q4(pbp())
    frame = mod.compare_actual_pair_predictions(pairs, FakeModel(), FEATURES)
    assert len(frame) == 2
    assert frame["p_kickoff_native"].gt(frame["p_kickoff_synthetic"]).all()
    assert frame["kickoff_goal_to_go_missing"].all()
    result = mod.summary(frame)
    assert result["n"] == 2 and result["games"] == 1
    assert result["synthetic_minus_native_absmean_pp"] == pytest.approx(6.)


def test_rejects_2026_data_and_changed_score():
    with pytest.raises(ValueError, match="2024"):
        mod.pair_2024_q4(pbp().assign(season=2026))
    frame = pbp()
    frame.loc[frame["play_id"].eq(2) & frame["game_id"].eq("G1"), "posteam_score"] = 30
    result = mod.pair_2024_q4(frame)
    assert len(result) == 1


def test_model_requires_correct_features():
    pairs = mod.pair_2024_q4(pbp())
    frame = mod.feature_frame(pairs, "kick")
    with pytest.raises(ValueError, match="Missing model features"):
        mod.model_probabilities(FakeModel(), FEATURES+["missing_column"], frame)
