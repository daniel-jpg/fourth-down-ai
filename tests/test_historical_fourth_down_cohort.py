"""Synthetic-only historical cohort contract and leak-proof feature boundary."""
from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from src.historical_fourth_down_cohort import (
    ACTION_COLUMNS, FEATURE_COLUMNS, OUTCOME_COLUMNS, PRESNAP_COLUMNS,
    extract_historical_cohort, fingerprint_frame, join_training_labels,
)


def fixture():
    base = {
        "season": 2020, "season_type": "REG", "week": 8,
        "game_id": "2020_08_AAA_BBB", "play_id": 100.,
        "down": 4., "qtr": 4., "game_seconds_remaining": 299.,
        "yardline_100": 95., "ydstogo": 3., "goal_to_go": 0.,
        "score_differential": 4., "posteam_timeouts_remaining": 3.,
        "defteam_timeouts_remaining": 2., "posteam": "BBB",
        "home_team": "BBB", "away_team": "AAA", "location": "Home",
        "action": "NORMAL_GO_PASS", "result": 4.,
        "converted": 1, "yards_gained": 4., "epa": 1.1, "desc": "completion",
    }
    rows = [base]
    second = deepcopy(base)
    second.update(play_id=200., posteam="AAA", score_differential=-4.,
                  action="PUNT", result=4.)
    rows.append(second)
    neutral = deepcopy(base)
    neutral.update(
        game_id="2020_08_CCC_DDD", play_id=101., home_team="CCC",
        away_team="DDD", posteam="DDD", location="Neutral",
        action="FIELD_GOAL", score_differential=-3., result=-7.,
    )
    rows.append(neutral)
    return pd.DataFrame(rows)


def test_separate_features_actions_and_offense_win_labels():
    result = extract_historical_cohort(fixture())
    p, a, y = (result[x] for x in ("pre_snap", "observed_actions", "final_labels"))
    assert list(p) == PRESNAP_COLUMNS
    assert list(a) == ACTION_COLUMNS
    assert list(y) == OUTCOME_COLUMNS
    assert set(FEATURE_COLUMNS).isdisjoint(
        {"action", "result", "first_down", "yards_gained", "converted", "epa"}
    )
    assert len(p) == len(a) == len(y) == 3
    assert set(p["feature_asof"]) == {"pre_snap"}
    grouped = join_training_labels(result)
    assert grouped.loc[grouped["posteam_timeouts_remaining"] == 3].shape[0] == 3
    assert grouped["observed_action"].tolist() == [
        "NORMAL_GO_PASS", "PUNT", "FIELD_GOAL"
    ]
    assert grouped["offense_win"].tolist() == [1, 0, 1]
    assert grouped.loc[grouped["site"] == "NEUTRAL", "site_advantage"].iloc[0] == 0
    assert result["audit"]["included_decisions"] == 3
    assert result["audit"]["external_final_score_verification"] == "NOT PERFORMED"


def test_altering_final_labels_and_post_play_fields_cannot_change_features():
    frame = fixture()
    original = extract_historical_cohort(frame)
    changed = frame.copy(deep=True)
    changed["result"] *= -1
    changed["converted"] = 0
    changed["yards_gained"] = 200.
    changed["epa"] = -999.
    changed["desc"] = "fake result with touchdown"
    changed.loc[0, "action"] = "NORMAL_GO_RUN"
    rerun = extract_historical_cohort(changed)
    pd.testing.assert_frame_equal(original["pre_snap"], rerun["pre_snap"])
    assert fingerprint_frame(original["pre_snap"]) == fingerprint_frame(rerun["pre_snap"])
    assert original["final_labels"]["offense_win"].tolist() != rerun["final_labels"]["offense_win"].tolist()
    assert original["observed_actions"]["observed_action"].tolist() != rerun["observed_actions"]["observed_action"].tolist()


def test_determinism_is_independent_of_source_row_order():
    obj = fixture()
    before = extract_historical_cohort(obj)
    after = extract_historical_cohort(obj.iloc[::-1].reset_index(drop=True))
    for key in ("pre_snap", "observed_actions", "final_labels", "excluded"):
        pd.testing.assert_frame_equal(before[key], after[key])
    assert before["audit"] == after["audit"]


@pytest.mark.parametrize("year", [2013, 2026, 2027])
def test_fails_closed_on_unapproved_years(year):
    obj = fixture()
    obj.loc[0, "season"] = year
    with pytest.raises(ValueError, match="2014-2025"):
        extract_historical_cohort(obj)


def test_missing_result_tie_fake_ot_and_no_play_are_counted_as_exclusions():
    obj = fixture()
    rows = []
    configs = [
        {"game_id": "2020_09_FAKE_A", "action": "FAKE_PUNT_RUN"},
        {"game_id": "2020_09_NOP_A", "action": "NO_PLAY"},
        {"game_id": "2020_09_TIE_A", "result": 0.},
        {"game_id": "2020_09_MISSING_A", "result": np.nan},
        {"game_id": "2020_09_OT_A", "qtr": 5., "game_seconds_remaining": 50.},
        {"game_id": "2020_09_TOS_A", "posteam_timeouts_remaining": np.nan},
        {"game_id": "2020_09_VENUE_A", "location": None},
    ]
    for i, settings in enumerate(configs):
        r = obj.iloc[0].to_dict()
        r.update(settings, play_id=float(10+i))
        rows.append(r)
    combined = pd.concat([obj, pd.DataFrame(rows)], ignore_index=True)
    report = extract_historical_cohort(combined)
    assert report["audit"]["included_decisions"] == 3
    assert report["audit"]["source_candidates"] == 10
    assert report["audit"]["exclusion_counts"] == {
        "administrative_no_play": 1,
        "fake_or_unclassified_action": 1,
        "missing_final_result": 1,
        "missing_or_invalid_presnap_feature": 1,
        "missing_venue": 1,
        "overtime": 1,
        "tied_final_game": 1,
    }
    assert len(report["excluded"]) == 7


def test_duplicate_ids_fatal_and_conflicting_final_game_margin_fatal():
    obj = fixture()
    bad = pd.concat([obj, obj.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate game/play"):
        extract_historical_cohort(bad)
    obj.loc[1, "result"] = -3.
    with pytest.raises(ValueError, match="Conflicting final home margins"):
        extract_historical_cohort(obj)


def test_unclassified_actions_do_not_become_normal_go():
    obj = fixture()
    obj.loc[0, "action"] = "UNKNOWN"
    obj.loc[1, "action"] = "FAKE_FG_PASS"
    out = extract_historical_cohort(obj)
    assert out["audit"]["included_decisions"] == 1
    assert out["audit"]["exclusion_counts"]["fake_or_unclassified_action"] == 2


def test_impossible_distance_and_wrong_clock_excluded():
    obj = fixture()
    obj.loc[0, "yardline_100"] = 98.
    obj.loc[0, "ydstogo"] = 99.
    obj.loc[1, "qtr"] = 1.
    obj.loc[1, "game_seconds_remaining"] = 299.
    out = extract_historical_cohort(obj)
    assert out["audit"]["included_decisions"] == 1
    assert out["audit"]["exclusion_counts"]["impossible_yards_to_go"] == 1
    assert out["audit"]["exclusion_counts"]["missing_or_invalid_presnap_feature"] == 1


def test_missing_game_or_play_identifier_fatal():
    obj = fixture()
    obj.loc[0, "play_id"] = float("nan")
    with pytest.raises(ValueError, match="Missing historical game or play ID"):
        extract_historical_cohort(obj)
