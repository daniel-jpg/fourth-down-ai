"""Every 2025 Q4 kickoff must be counted or explicitly marked unsupported."""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/73_audit_2025_kickoff_coverage.py"
spec = importlib.util.spec_from_file_location("step73_2025_kickoffs", SOURCE)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def sample():
    outcomes = [
        ("normal", "", 0, 0),
        ("onside", "An onside kickoff attempted", 0, 0),
        ("touchdown", "Kickoff returned for touchdown", 1, 0),
        ("penalty", "Kickoff return", 0, 1),
        ("unmatched", "Kickoff return but no next first snap", 0, 0),
        ("normal2", "Touchback", 0, 0),
        ("normal3", "Kickoff returned", 0, 0),
    ]
    pbp = pd.DataFrame([
        {
            "season": 2025, "season_type": "REG",
            "game_id": f"g{i}", "play_id": 100. + i,
            "play_type": "kickoff", "qtr": 4,
            "week": 1+i,
            "yardline_100": 35.,
            "game_seconds_remaining": 200.,
            "desc": description,
            "return_touchdown": td, "penalty": penalty,
            "posteam": "SEA",
            "posteam_score": 31,
            "defteam_score": 30,
        }
        for i, (name, description, td, penalty) in enumerate(outcomes)
    ])
    donors = pd.DataFrame({
        "game_id": ["g0", "g5", "g6"],
        "kick_play_id": [100., 105., 106.],
    })
    return pbp, donors


def test_every_kickoff_has_exclusive_reason_for_coverage_or_exclusion():
    pbp, donors = sample()
    categories = audit.mark_outcomes(audit.candidate_kickoffs(pbp), donors)
    assert categories["outcome_group"].to_list() == [
        "modeled_ordinary",
        "onside_attempt",
        "kickoff_return_td",
        "penalty_or_rekick",
        "other_unmatched",
        "modeled_ordinary",
        "modeled_ordinary",
    ]
    report = audit.build_report(pbp, donors)
    top = report["groups"][0]
    assert top["total_kickoffs"] == 7
    assert top["counts"]["modeled_ordinary"] == 3
    assert top["counts"]["onside_attempt"] == 1
    assert top["counts"]["kickoff_return_td"] == 1
    assert top["counts"]["penalty_or_rekick"] == 1
    assert top["counts"]["other_unmatched"] == 1
    assert top["ordinary_share"] == pytest.approx(3/7)
    assert top["ordinary_share_game_cluster_ci95"] is None
    assert top["low_support"]
    assert report["not_a_forecast"] and report["no_2026_outcomes"]


def test_onside_and_return_touchdown_priority_never_invent_regular_returns():
    pbp, donors = sample()
    pbp.loc[1, "return_touchdown"] = 1
    result = audit.mark_outcomes(audit.candidate_kickoffs(pbp), donors)
    assert result.loc[result["game_id"].eq("g1"), "outcome_group"].iloc[0] == "onside_attempt"
    stats = audit.summarize_group("all", result)
    assert stats["special_flag_overlaps"] == 1


def test_unmatched_frozen_donor_is_a_hard_failure_not_a_lower_rate():
    pbp, donors = sample()
    donors.loc[0, "game_id"] = "not-in-kickoffs"
    with pytest.raises(RuntimeError, match="donor IDs"):
        audit.build_report(pbp, donors)


def test_bad_season_duplicate_kickoff_and_missing_schema_fail_closed():
    pbp, _ = sample()
    with pytest.raises(ValueError, match="Only 2025"):
        audit.candidate_kickoffs(pbp.assign(season=2026))
    with pytest.raises(ValueError, match="Duplicate"):
        audit.candidate_kickoffs(pd.concat([pbp, pbp.iloc[[0]]]))
    with pytest.raises(ValueError, match="fields missing"):
        audit.candidate_kickoffs(pbp.drop(columns=["return_touchdown"]))


def test_observed_scope_excludes_late_clock_and_different_kicking_spot():
    pbp, donors = sample()
    irrelevant = pbp.iloc[[0]].copy()
    irrelevant.loc[:, "game_id"] = "extra"
    irrelevant.loc[:, "game_seconds_remaining"] = 40.
    notstandard = pbp.iloc[[0]].copy()
    notstandard.loc[:, "game_id"] = "different-spot"
    notstandard.loc[:, "yardline_100"] = 50.
    full = pd.concat([pbp, irrelevant, notstandard], ignore_index=True)
    assert len(audit.candidate_kickoffs(full)) == 7
    assert audit.build_report(full, donors)["groups"][0]["total_kickoffs"] == 7


def test_game_cluster_fraction_is_reproducible():
    pbp, donors = sample()
    base = audit.mark_outcomes(audit.candidate_kickoffs(pbp), donors)
    # 30 genuinely distinct game IDs with unequal numbers of kickoff events.
    frames = [
        base.assign(game_id=f"full-{i}")
        for i in range(30)
    ]
    many = pd.concat(frames, ignore_index=True)
    first = audit.cluster_coverage_interval(many)
    assert first == audit.cluster_coverage_interval(many)
    assert first is not None and len(first) == 2
    assert first[0] <= many["modeled_ordinary"].mean() <= first[1]


def test_no_future_2026_download_or_production_file_mutation():
    source = SOURCE.read_text()
    assert "nfl.load_pbp([REPORT_YEAR])" in source
    assert "REPORT_YEAR = 2025" in source
    assert "nfl.load_pbp([2026])" not in source
    assert "joblib.dump(" not in source
    assert "models/win_probability_model.joblib" not in source
