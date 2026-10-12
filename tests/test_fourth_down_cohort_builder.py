"""Fabricated-only 2027 cohort construction: integrity, exclusions, leakage."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from src.fourth_down_cohort_builder import (
    REQUIRED_MANIFEST,
    build_synthetic_cohort,
)
from src.fourth_down_policy_study import evaluate


def fixture():
    return {
        "manifest": dict(REQUIRED_MANIFEST),
        "pre_snap": [{
            "game_id": "synthetic-game-1", "play_id": "11", "season": 2027,
            "week": 2, "quarter": 4, "down": 4,
            "offense_team": "HOME", "home_team": "HOME", "away_team": "AWAY",
            "site": "HOME", "game_seconds_remaining": 299,
            "yardline_100": 6, "ydstogo": 4, "score_differential": -4,
            "posteam_timeouts_remaining": 3, "defteam_timeouts_remaining": 3,
            "feature_asof": "pre_snap",
            "policy_commit": REQUIRED_MANIFEST["policy_commit"],
            "policy_action": "FIELD_GOAL",
            "eligible_actions": ["NORMAL_GO_PASS", "FIELD_GOAL"],
            "q_wp_by_action": {"NORMAL_GO_PASS": .34, "FIELD_GOAL": .37},
            "propensity_policy": .4, "mu_policy": .55, "close_call": True,
        }],
        "observed_plays": [{
            "game_id": "synthetic-game-1", "play_id": "11",
            "observed_action": "FIELD_GOAL", "no_play": False,
        }],
        "final_games": [{
            "game_id": "synthetic-game-1", "season": 2027, "week": 2,
            "home_team": "HOME", "away_team": "AWAY",
            "home_score_final": 28, "away_score_final": 24,
            "status": "final", "game_type": "REG",
        }],
    }


def test_complete_join_and_separate_factual_label_is_offense_perspective():
    obj = fixture()
    result = build_synthetic_cohort(obj)
    assert result["included_decisions"] == 1
    assert result["exclusion_counts"] == {}
    r = result["records"][0]
    assert r["offense_win"] == 1
    assert r["q_wp_policy"] == .37
    assert r["q_wp_observed"] == .37
    assert r["site"] == "HOME" and r["close_call"] is True
    assert r["feature_asof"] == "pre_snap"
    assert len(result["synthetic_input_sha256"]) == 64
    # Evaluator in repository accepts builder output without peeking at labels.
    stats = evaluate(result["records"], result["manifest"], bootstrap_draws=20)
    assert stats["n_decisions"] == 1
    assert stats["factual_action_calibration"]["factual_supported_decisions"] == 1


def test_away_team_final_win_and_ineligible_factual_action_preserved():
    obj = fixture()
    snap = obj["pre_snap"][0]
    snap["offense_team"] = "AWAY"
    snap["site"] = "AWAY"
    obj["observed_plays"][0]["observed_action"] = "NORMAL_GO_RUN"
    result = build_synthetic_cohort(obj)
    assert result["included_decisions"] == 1
    assert result["records"][0]["offense_win"] == 0
    assert result["records"][0]["q_wp_observed"] is None
    assert result["factual_unsupported_decisions"] == 1
    stats = evaluate(result["records"], result["manifest"], bootstrap_draws=20)
    assert stats["factual_action_calibration"]["factual_unsupported_decisions"] == 1


@pytest.mark.parametrize("field,value,reason", [
    ("quarter", 5, "overtime"),
    ("down", 3, "not_fourth_down"),
])
def test_regulation_fourth_down_only(field, value, reason):
    obj = fixture()
    obj["pre_snap"][0][field] = value
    result = build_synthetic_cohort(obj)
    assert not result["records"]
    assert result["exclusion_counts"] == {reason: 1}


@pytest.mark.parametrize("change,reason", [
    ("fake", "fake_or_unclassified_action"),
    ("no_play", "administrative_no_play"),
    ("unmatched_action", "unmatched_observed_play"),
    ("unmatched_game", "unmatched_final_game"),
    ("not_final", "game_not_final"),
    ("tied", "tied_final_game"),
])
def test_excluded_rows_receive_explicit_reason(change, reason):
    obj = fixture()
    if change == "fake":
        obj["observed_plays"][0]["observed_action"] = "FAKE_FG_PASS"
    elif change == "no_play":
        obj["observed_plays"][0]["no_play"] = True
    elif change == "unmatched_action":
        obj["observed_plays"].clear()
    elif change == "unmatched_game":
        obj["final_games"].clear()
    elif change == "not_final":
        obj["final_games"][0]["status"] = "pending"
    elif change == "tied":
        obj["final_games"][0]["away_score_final"] = 28
    result = build_synthetic_cohort(obj)
    assert result["included_decisions"] == 0
    assert result["excluded_decisions"] == [{
        "game_id": "synthetic-game-1", "play_id": "11", "reason": reason,
    }]


@pytest.mark.parametrize("part,value,description", [
    ("season", 2026, "season"),
    ("feature_asof", "post_snap", "pre-snap"),
    ("policy_commit", "HEAD", "pinned"),
    ("close_call", "yes", "close_call"),
    ("q_wp_by_action", {"FIELD_GOAL": .37}, "exactly eligible"),
    ("policy_action", "NORMAL_GO_PASS", "highest-WP"),
    ("site", "AWAY", "offense's perspective"),
    ("ydstogo", 80, "cannot cross"),
    ("posteam_timeouts_remaining", 4, "posteam_timeouts_remaining"),
    ("game_seconds_remaining", 1600, "clock"),
])
def test_invalid_presnap_models_fail_closed(part, value, description):
    obj = fixture()
    obj["pre_snap"][0][part] = value
    with pytest.raises(ValueError, match=description):
        build_synthetic_cohort(obj)


@pytest.mark.parametrize("forbidden_key", [
    "offense_win", "observed_action", "home_score_final",
    "yardline_after_play", "winner", "post_snap_epa",
])
def test_post_snap_labels_cannot_enter_presnap_features(forbidden_key):
    obj = fixture()
    obj["pre_snap"][0][forbidden_key] = 1
    with pytest.raises(ValueError, match="extra=.*pre-snap cannot contain labels"):
        build_synthetic_cohort(obj)


@pytest.mark.parametrize("field,value", [
    ("mode", "prospective"),
    ("source_type", "real_nfl_data"),
    ("policy_commit", "HEAD"),
    ("nuisance_training_end_season", 2026),
    ("holdout_season", 2026),
])
def test_prospective_and_reserved_data_remain_inaccessible(field, value):
    obj = fixture()
    obj["manifest"][field] = value
    with pytest.raises(ValueError, match="Only fabricated synthetic"):
        build_synthetic_cohort(obj)


def test_duplicate_ids_and_mismatched_game_are_fail_closed():
    obj = fixture()
    obj["pre_snap"].append(copy.deepcopy(obj["pre_snap"][0]))
    with pytest.raises(ValueError, match="Duplicate pre_snap"):
        build_synthetic_cohort(obj)
    obj = fixture()
    obj["final_games"][0]["home_team"] = "DIFFERENT"
    with pytest.raises(ValueError, match="Game team/week mismatch"):
        build_synthetic_cohort(obj)


def test_permutation_is_deterministic_and_auditable():
    obj = fixture()
    second = copy.deepcopy(obj["pre_snap"][0])
    second["play_id"] = "9"
    obj["pre_snap"].append(second)
    second_observed = copy.deepcopy(obj["observed_plays"][0])
    second_observed["play_id"] = "9"
    obj["observed_plays"].append(second_observed)
    res1 = build_synthetic_cohort(obj)
    obj["pre_snap"].reverse()
    obj["observed_plays"].reverse()
    res2 = build_synthetic_cohort(obj)
    assert res1["records"] == res2["records"]
    assert [r["play_id"] for r in res1["records"]] == ["11", "9"]
    # Fingerprint refers to the *input JSON* and hence can differ by order.
    assert res1["synthetic_input_sha256"] != res2["synthetic_input_sha256"]


def test_synthetic_cli_roundtrip(tmp_path):
    path, out = tmp_path / "fabricated.json", tmp_path / "joined.json"
    path.write_text(json.dumps(fixture()))
    subprocess.run([
        sys.executable, "-m", "src.fourth_down_cohort_builder",
        str(path), "--output", str(out),
    ], check=True, cwd=Path(__file__).resolve().parents[1])
    saved = json.loads(out.read_text())
    assert saved["included_decisions"] == 1
    assert "SYNTHETIC ONLY" in saved["warning"]


def test_builder_contains_no_external_data_loader():
    source = (Path(__file__).resolve().parents[1] / "src" /
              "fourth_down_cohort_builder.py").read_text()
    assert "import nflreadpy" not in source
    assert "requests.get" not in source
    assert "urllib.request" not in source
    assert "nflverse" not in source
