from pathlib import Path
import os
import runpy

import pytest


ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)

ENGINE = runpy.run_path(
    str(ROOT / "src" / "42_build_decision_engine.py")
)

recommend = ENGINE["recommend"]

NORMAL_ACTIONS = {
    "NORMAL_GO_PASS",
    "NORMAL_GO_RUN",
    "FIELD_GOAL",
    "PUNT",
}


def run_policy(state):
    return recommend(
        state,
        n_simulations=1200,
        seed=43,
    )


def normal_rows(result):
    rows = result["results"]

    return rows[
        rows["action"].isin(NORMAL_ACTIONS)
    ].copy()


def eligible_normal_rows(result):
    rows = normal_rows(result)

    return rows[
        rows["eligible"]
    ].copy()


def best_normal_action(result):
    rows = eligible_normal_rows(result)

    return (
        rows
        .sort_values(
            "win_probability_pct",
            ascending=False,
        )
        .iloc[0]["action"]
    )


def action_value(result, action):
    row = result["results"][
        result["results"]["action"] == action
    ].iloc[0]

    assert bool(row["eligible"])

    return float(
        row["win_probability_pct"]
    )


def action_is_eligible(result, action):
    row = result["results"][
        result["results"]["action"] == action
    ].iloc[0]

    return bool(row["eligible"])


# ---------------------------------------------------------------------------
# Release policy smoke tests
# ---------------------------------------------------------------------------

def test_late_two_point_deficit_prefers_field_goal():
    result = run_policy(
        {
            "qtr": 4,
            "game_seconds_remaining": 10.0,
            "yardline_100": 25.0,
            "ydstogo": 10.0,
            "score_differential": -2.0,
            "posteam_timeouts_remaining": 1.0,
            "defteam_timeouts_remaining": 1.0,
            "site": "HOME",
            "roof": "outdoors",
        }
    )

    assert best_normal_action(result) == "FIELD_GOAL"


def test_known_fourth_and_three_prefers_go_strategy():
    result = run_policy(
        {
            "qtr": 4,
            "game_seconds_remaining": 458.0,
            "yardline_100": 30.0,
            "ydstogo": 3.0,
            "score_differential": -3.0,
            "posteam_timeouts_remaining": 3.0,
            "defteam_timeouts_remaining": 2.0,
            "site": "AWAY",
            "roof": "outdoors",
        }
    )

    pass_wp = action_value(
        result,
        "NORMAL_GO_PASS",
    )
    run_wp = action_value(
        result,
        "NORMAL_GO_RUN",
    )
    fg_wp = action_value(
        result,
        "FIELD_GOAL",
    )

    go_wp = max(
        pass_wp,
        run_wp,
    )

    assert go_wp > fg_wp
    assert (
        best_normal_action(result)
        in {
            "NORMAL_GO_PASS",
            "NORMAL_GO_RUN",
        }
    )


def test_late_lead_from_own_35_prefers_punt():
    result = run_policy(
        {
            "qtr": 4,
            "game_seconds_remaining": 90.0,
            "yardline_100": 65.0,
            "ydstogo": 10.0,
            "score_differential": 7.0,
            "posteam_timeouts_remaining": 2.0,
            "defteam_timeouts_remaining": 0.0,
            "site": "HOME",
            "roof": "outdoors",
        }
    )

    assert best_normal_action(result) == "PUNT"


def test_goal_line_trailing_by_four_prefers_go():
    result = run_policy(
        {
            "qtr": 4,
            "game_seconds_remaining": 45.0,
            "yardline_100": 1.0,
            "ydstogo": 1.0,
            "score_differential": -4.0,
            "posteam_timeouts_remaining": 2.0,
            "defteam_timeouts_remaining": 2.0,
            "site": "AWAY",
            "roof": "outdoors",
        }
    )

    assert (
        best_normal_action(result)
        in {
            "NORMAL_GO_PASS",
            "NORMAL_GO_RUN",
        }
    )

    assert not action_is_eligible(
        result,
        "PUNT",
    )


def test_regular_season_ot_response_prefers_field_goal():
    result = run_policy(
        {
            "qtr": 5,
            "game_seconds_remaining": 300.0,
            "yardline_100": 35.0,
            "ydstogo": 5.0,
            "score_differential": 0.0,
            "posteam_timeouts_remaining": 2.0,
            "defteam_timeouts_remaining": 2.0,
            "site": "HOME",
            "roof": "outdoors",
            "ot_format": "REGULAR_SEASON",
            "ot_phase": "RESPONSE",
            "ot_period": 1,
        }
    )

    assert best_normal_action(result) == "FIELD_GOAL"
