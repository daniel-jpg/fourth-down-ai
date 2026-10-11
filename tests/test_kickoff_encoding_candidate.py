"""Regression tests for an opt-in (never production) kickoff metadata encoding."""

import numpy as np
import pandas as pd
import pytest

from src.kickoff_encoding_candidate import (
    native_like_q4_scoring_kickoffs, use_native_like_q4_kickoff_wp,
)
from src.kickoff_scenario import use_q4_kickoff_scenario


def kickoff(**kwargs):
    s = {
        "qtr": 4.0,
        "_score_change_original": 3.0,
        "is_kickoff": 1.0,
        "yardline_100": np.nan,
        "ydstogo": np.nan,
        "down": np.nan,
        "goal_to_go": 0.0,
        "game_seconds_remaining": 290.0,
        "half_seconds_remaining": 290.0,
        "home_score_differential": 1.0,
        "is_home_posteam": 0.0,
        "final_five_minutes": 1.0,
    }
    return dict(s, **kwargs)


def fake_engine():
    ns = {}
    exec(
        "def predict_original_team_wp(states, original_on_home_axis, neutral_site=False):\n"
        "    return [(s['yardline_100'], s['ydstogo'], s['is_kickoff']) for s in states]\n"
        "def dispatch(states):\n"
        "    return predict_original_team_wp(states, True)\n"
        "def evaluate_post_play_states(base, states):\n"
        "    return predict_original_team_wp(states, True)\n",
        ns,
    )
    return {
        "predict_original_team_wp": ns["predict_original_team_wp"],
        "dispatch": ns["dispatch"],
        "evaluate_post_play_states": ns["evaluate_post_play_states"],
    }


def test_only_unresolved_q4_scoring_kickoffs_are_changed():
    s = kickoff()
    not_scored = kickoff(_score_change_original=0)
    earlier = kickoff(qtr=3)
    snapped = kickoff(is_kickoff=0, yardline_100=65, down=1, ydstogo=10)
    special_spot = kickoff(yardline_100=40)
    changed = native_like_q4_scoring_kickoffs(
        [s, not_scored, earlier, snapped, special_spot]
    )
    assert changed[0]["yardline_100"] == 35.
    assert changed[0]["ydstogo"] == 0.
    assert np.isnan(changed[0]["down"])
    assert changed[0]["goal_to_go"] == 0.
    assert changed[1] is not_scored
    assert changed[2] is earlier
    assert changed[3] is snapped
    assert changed[4] is special_spot
    assert np.isnan(s["yardline_100"])  # no mutation of original state


def test_patch_only_changes_wp_inputs_and_restores():
    engine = fake_engine()
    original = engine["predict_original_team_wp"]
    s = kickoff()
    assert np.isnan(engine["dispatch"]([s])[0][0])
    with use_native_like_q4_kickoff_wp(engine):
        observation = engine["dispatch"]([s])
        assert observation == [(35., 0., 1.)]
        assert np.isnan(s["yardline_100"])
        assert engine["predict_original_team_wp"] is original
    assert np.isnan(engine["dispatch"]([s])[0][0])
    assert original.__globals__["predict_original_team_wp"] is original


def test_patch_restored_after_exception_and_cannot_nest():
    engine = fake_engine()
    with pytest.raises(RuntimeError, match="oops"):
        with use_native_like_q4_kickoff_wp(engine):
            raise RuntimeError("oops")
    assert engine["predict_original_team_wp"].__globals__["predict_original_team_wp"] is engine["predict_original_team_wp"]
    with use_native_like_q4_kickoff_wp(engine):
        with pytest.raises(RuntimeError, match="already overridden"):
            with use_native_like_q4_kickoff_wp(engine):
                pass


def test_invalid_standard_kickoff_locations_rejected():
    with pytest.raises(ValueError, match="Kickoff yardline"):
        native_like_q4_scoring_kickoffs([kickoff()], kickoff_yardline_100=np.nan)
    with pytest.raises(ValueError, match="Kickoff yardline"):
        with use_native_like_q4_kickoff_wp(fake_engine(), kickoff_yardline_100=101):
            pass


def test_combined_kickoff_resolution_does_not_reencode_first_down():
    engine = fake_engine()
    pool = pd.DataFrame([
        {"season": 2024, "game_id": "g1", "quarter": 4, "transition_type": "touchback",
         "own_yardline": 30., "elapsed_seconds": 0.},
        {"season": 2024, "game_id": "g1", "quarter": 4, "transition_type": "return",
         "own_yardline": 28., "elapsed_seconds": 6.},
    ])
    s = kickoff()
    evaluate = engine["evaluate_post_play_states"].__globals__
    with use_native_like_q4_kickoff_wp(engine):
        assert evaluate["evaluate_post_play_states"]({"qtr":4}, [s])[0] == (35., 0., 1.)
        with use_q4_kickoff_scenario(engine, pool, touchback_rate=1., seed=9):
            resolved = evaluate["evaluate_post_play_states"]({"qtr":4}, [s])
            assert resolved == [(65., 10., 0.)]
        assert evaluate["evaluate_post_play_states"]({"qtr":4}, [s])[0] == (35., 0., 1.)
    assert np.isnan(s["yardline_100"])
