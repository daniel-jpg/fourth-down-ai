"""Unit tests for read-only, out-of-time 2025 kickoff encoding validation."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

file = Path(__file__).resolve().parents[1] / "src/64_validate_2025_kickoff_encoding.py"
spec = importlib.util.spec_from_file_location("observed_2025_encoding", file)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def play(game, play_id, play_type, time, *, receiver="SEA", own=None,
         kick_yard=35.0, touchback=0, desc="", score_for=20,
         score_against=17, season_type="REG", result=7):
    return {
        "season": 2025, "season_type": season_type,
        "game_id": game, "play_id": play_id, "play_type": play_type,
        "qtr": 4, "game_seconds_remaining": float(time),
        "half_seconds_remaining": float(time),
        "posteam": receiver, "home_team": "SEA", "result": result,
        "posteam_score": score_for, "defteam_score": score_against,
        "posteam_timeouts_remaining": 2,
        "defteam_timeouts_remaining": 1,
        "down": np.nan if own is None else 1.0,
        "ydstogo": 0.0 if own is None else 10.0,
        "yardline_100": float(kick_yard if own is None else 100-own),
        "goal_to_go": 0.0, "touchback": touchback, "desc": desc,
    }


def sample():
    return pd.DataFrame([
        play("g1", 1, "kickoff", 300, touchback=1),
        play("g1", 2, "run", 300, own=35),
        play("g1", 3, "pass", 285, own=42),
        play("g1", 4, "kickoff", 240, receiver="LAR",
             score_for=21, score_against=20),
        play("g1", 5, "run", 235, receiver="LAR", own=28,
             score_for=21, score_against=20),
        play("g2", 1, "kickoff", 230, desc="onside attempt"),
        play("g2", 2, "run", 225, own=44),
        play("g3", 1, "kickoff", 220, receiver="SEA"),
        play("g3", 2, "run", 215, receiver="LAR", own=30),
        play("g4", 1, "kickoff", 200, touchback=1),
        play("g4", 2, "run", 200, own=20),
        play("g5", 1, "kickoff", 180, kick_yard=40),
        play("g5", 2, "run", 174, own=29),
        play("g6", 1, "kickoff", 100, season_type="POST"),
        play("g6", 2, "run", 95, own=30, season_type="POST"),
        play("g7", 1, "kickoff", 100, desc="PENALTY on kicking team"),
        play("g7", 2, "run", 95, own=30),
    ])


def test_2025_pairs_enforce_kick_sequence_and_exclusions():
    paired = audit.ordinary_2025_q4_pairs(sample())
    assert paired["game_id"].tolist() == ["g1", "g1", "g4"]
    assert paired["play_id_kick"].tolist() == [1, 4, 1]
    assert paired["play_id_snap"].tolist() == [2, 5, 2]
    assert paired["touchback"].tolist() == [1, 0, 1]


def test_2025_requires_2025_only():
    with pytest.raises(ValueError, match="Only 2025"):
        audit.ordinary_2025_q4_pairs(sample().assign(season=2026))
    with pytest.raises(ValueError, match="Only 2025"):
        audit.ordinary_2025_q4_pairs(sample().assign(season=2024))


def test_2025_changed_score_kickoff_is_not_paired():
    pbp = sample()
    pbp.loc[pbp["game_id"].eq("g1") & pbp["play_id"].eq(2),
            "posteam_score"] = 32
    pairs = audit.ordinary_2025_q4_pairs(pbp)
    assert pairs["game_id"].tolist() == ["g1", "g4"]
    assert pairs["play_id_kick"].tolist() == [4, 1]


def test_predeclared_candidate_preserves_non_kickoff_features():
    native = pd.DataFrame({
        "down": [np.nan, np.nan],
        "ydstogo": [0., 0.],
        "yardline_100": [35., 35.],
        "goal_to_go": [0., 0.],
        "is_home_posteam": [0., 1.],
        "home_score_differential": [-3., 2.],
    })
    engine, aligned = audit.candidate_features(native)
    assert engine["ydstogo"].isna().all()
    assert engine["yardline_100"].isna().all()
    pd.testing.assert_frame_equal(aligned, native)
    with pytest.raises(ValueError, match="Missing kickoff"):
        audit.candidate_features(native.drop(columns="goal_to_go"))


def test_game_cluster_bootstrap_is_repeatable_and_validated():
    ids = np.array(["g1", "g1", "g2", "g3"])
    diffs = np.array([.01, -.02, .04, .03])
    ci1 = audit.game_cluster_ci(diffs, ids, repeats=200, seed=12)
    ci2 = audit.game_cluster_ci(diffs, ids, repeats=200, seed=12)
    assert ci1 == ci2
    assert ci1[0] <= ci1[1]
    with pytest.raises(ValueError, match="matching nonempty"):
        audit.game_cluster_ci(diffs[:-1], ids)
    with pytest.raises(ValueError, match="At least 100"):
        audit.game_cluster_ci(diffs, ids, repeats=50)
