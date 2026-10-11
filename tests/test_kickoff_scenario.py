"""Unit coverage for the isolated opt-in engine integration hook."""

import numpy as np
import pandas as pd
import pytest

from src.kickoff_scenario import use_q4_kickoff_scenario


def fake_pool():
    rows = []
    for i in range(80):
        rows += [
            {"season": 2024, "game_id": f"g{i}", "quarter": 4,
             "transition_type": "touchback", "own_yardline": 30.0,
             "elapsed_seconds": 0.0},
            {"season": 2024, "game_id": f"g{i}", "quarter": 4,
             "transition_type": "return", "own_yardline": 28.0,
             "elapsed_seconds": 6.0},
        ]
    return pd.DataFrame(rows)


def fake_engine():
    # Functions need a dedicated, mutable globals dictionary to approximate
    # engine=runpy.run_path(..), whose functions share their own module globals.
    namespace = {}
    exec(
        "def evaluate_post_play_states(base, states):\n"
        "    return [(s['is_kickoff'], s['yardline_100'], "
        "s['game_seconds_remaining']) for s in states]\n",
        namespace,
    )
    return {"evaluate_post_play_states": namespace["evaluate_post_play_states"]}


def state(*, qtr=4, score=3.0, seconds=290.0):
    return {
        "qtr": qtr, "_score_change_original": score,
        "is_kickoff": 1.0, "yardline_100": np.nan, "down": np.nan,
        "ydstogo": np.nan, "goal_to_go": 0.0,
        "game_seconds_remaining": seconds,
        "half_seconds_remaining": seconds,
        "final_five_minutes": 1.0,
    }


def call_engine(engine, base, states):
    fn = engine["evaluate_post_play_states"]
    return fn.__globals__["evaluate_post_play_states"](base, states)


def test_opt_in_only_and_restores_original_on_exit():
    engine = fake_engine()
    original = engine["evaluate_post_play_states"]
    s = state()
    assert call_engine(engine, {"qtr": 4}, [s])[0][0] == 1.0
    with use_q4_kickoff_scenario(engine, fake_pool(), touchback_rate=1.0, seed=10):
        actual = call_engine(engine, {"qtr": 4}, [s])[0]
        assert actual == (0.0, 65.0, 290.0)
        assert s["is_kickoff"] == 1.0  # original input preserved
        assert engine["evaluate_post_play_states"] is original
    assert call_engine(engine, {"qtr": 4}, [s])[0][0] == 1.0


def test_recursion_children_inherit_parent_kickoff_state():
    engine = fake_engine()
    with use_q4_kickoff_scenario(engine, fake_pool(), touchback_rate=1.0):
        parent = state()
        # Model: evaluator receives resolved parent before it derives Try children.
        observed = []
        resolved = engine["evaluate_post_play_states"].__globals__["evaluate_post_play_states"]

        def simulate_try(base, children):
            observed.extend(children)
            return resolved(base, children)

        # Each child inherits parent kickoff outcome, so neither should draw
        # a new return from the pool or alter field position.
        _, yardline, _ = resolved({"qtr": 4}, [parent])[0]
        child = dict(parent)
        child["is_kickoff"] = 0.0
        child["yardline_100"] = yardline
        one = simulate_try({"qtr": 4}, [child])[0]
        two = simulate_try({"qtr": 4}, [child])[0]
        assert one[1] == two[1] == 65.0
        assert observed[0]["is_kickoff"] == 0.0


def test_engine_restored_on_exception():
    engine = fake_engine()
    original = engine["evaluate_post_play_states"]
    with pytest.raises(RuntimeError, match="unexpected"):
        with use_q4_kickoff_scenario(engine, fake_pool(), touchback_rate=1.0):
            raise RuntimeError("unexpected")
    assert call_engine(engine, {"qtr": 4}, [state()])[0][0] == 1.0
    assert original.__globals__["evaluate_post_play_states"] is original


def test_no_changes_outside_q4():
    engine = fake_engine()
    s = state(qtr=3)
    with use_q4_kickoff_scenario(engine, fake_pool(), touchback_rate=1.0):
        assert call_engine(engine, {"qtr": 3}, [s])[0][0] == 1.0


def test_nested_contexts_and_invalid_rate_rejected():
    engine = fake_engine()
    with pytest.raises(ValueError):
        with use_q4_kickoff_scenario(engine, fake_pool(), touchback_rate=1.5):
            pass
    with use_q4_kickoff_scenario(engine, fake_pool(), touchback_rate=1.0):
        with pytest.raises(RuntimeError, match="already overridden"):
            with use_q4_kickoff_scenario(engine, fake_pool(), touchback_rate=1.0):
                pass


def test_repeatable_kickoff_rng_across_contexts():
    engine = fake_engine()
    a = [state() for _ in range(120)]
    with use_q4_kickoff_scenario(engine, fake_pool(), touchback_rate=0.3, seed=42):
        values1 = call_engine(engine, {"qtr": 4}, a)
    with use_q4_kickoff_scenario(engine, fake_pool(), touchback_rate=0.3, seed=42):
        values2 = call_engine(engine, {"qtr": 4}, a)
    assert values1 == values2
