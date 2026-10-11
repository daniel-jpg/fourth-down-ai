"""Frozen 2025 kickoff donor extraction, sampling and production isolation."""

import numpy as np
import pandas as pd
import pytest

from src.kickoff_2025_empirical import (
    pool_from_matched_2025, validate_frozen_pool,
    resolve_post_score_q4_2025, use_frozen_2025_kickoffs,
)


def pool():
    rows = [
        ("g1", 1, "touchback", 35, 0),
        ("g2", 2, "return", 29, 6),
        ("g3", 3, "touchback", 20, 0),
        ("g4", 4, "return", 31, 5),
    ]
    return pd.DataFrame([
        {
            "season": 2025,
            "game_id": game,
            "week": week,
            "kick_play_id": float(100 + week),
            "quarter": 4,
            "transition_type": typ,
            "own_yardline": float(own),
            "elapsed_seconds": float(elapsed),
        }
        for game, week, typ, own, elapsed in rows
    ])


def state(*, quarter=4, score=3., seconds=299.,
          kickoff=1., midfield=False):
    return {
        "qtr": quarter,
        "_score_change_original": score,
        "is_kickoff": kickoff,
        "yardline_100": np.nan,
        "ydstogo": np.nan,
        "down": np.nan,
        "goal_to_go": 0.,
        "half_seconds_remaining": seconds,
        "game_seconds_remaining": seconds,
        "final_five_minutes": 1.,
        "home_score_differential": -1.,
        "is_home_posteam": 1.,
        "_kickoff_from_50": midfield,
    }


def fake_engine():
    namespace = {}
    exec(
        "def evaluate_post_play_states(base, states):\n"
        "    return [(s['is_kickoff'], s['yardline_100'], "
        "s['game_seconds_remaining']) for s in states]\n",
        namespace,
    )
    return {"evaluate_post_play_states": namespace["evaluate_post_play_states"]}


def test_builds_exact_observed_2025_transition_pool():
    pair = pd.DataFrame([
        {
            "game_id": "G2", "week": 14, "play_id_kick": 104.,
            "qtr_kick": 4, "touchback": 0,
            "yardline_100_kick": 35., "yardline_100_snap": 72.,
            "game_seconds_remaining_kick": 290.,
            "game_seconds_remaining_snap": 284.,
        },
        {
            "game_id": "G1", "week": 2, "play_id_kick": 102.,
            "qtr_kick": 4, "touchback": 1,
            "yardline_100_kick": 35., "yardline_100_snap": 65.,
            "game_seconds_remaining_kick": 120.,
            "game_seconds_remaining_snap": 120.,
        },
    ])
    got = pool_from_matched_2025(pair)
    assert got["game_id"].tolist() == ["G1", "G2"]
    assert got["transition_type"].tolist() == ["touchback", "return"]
    assert got["own_yardline"].tolist() == [35., 28.]
    assert got["elapsed_seconds"].tolist() == [0., 6.]
    assert len(validate_frozen_pool(got, min_count=2, min_games=2)) == 2
    with pytest.raises(ValueError, match="terminal-clock"):
        pool_from_matched_2025(
            pair.assign(game_seconds_remaining_kick=40.)
        )


def test_pool_rejects_other_seasons_bad_touchback_and_duplicates():
    validate_frozen_pool(pool(), min_count=4, min_games=4)
    with pytest.raises(ValueError, match="2025 REG"):
        validate_frozen_pool(
            pool().assign(season=2026), min_count=4, min_games=4,
        )
    wrong = pool()
    wrong.loc[0, "own_yardline"] = 30.
    with pytest.raises(ValueError, match="own 20/35"):
        validate_frozen_pool(wrong, min_count=4, min_games=4)
    duplicate = pd.concat([pool(), pool().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate"):
        validate_frozen_pool(duplicate, min_count=2, min_games=1)


def test_samples_joint_touchback_return_spot_and_clock():
    original = state()
    hundred = [original for _ in range(400)]
    got = resolve_post_score_q4_2025(
        hundred, pool(), rng=np.random.default_rng(9),
    )
    observed = {(round(x["yardline_100"]), x["game_seconds_remaining"])
                for x in got}
    assert observed.issubset({
        (65, 299.), (80, 299.), (71, 293.), (69, 294.),
    })
    assert len(observed) == 4
    assert original["is_kickoff"] == 1
    assert np.isnan(original["yardline_100"])
    assert all(x["down"] == 1 and x["is_kickoff"] == 0 for x in got)


def test_2026_midfield_rule_and_noneligible_states():
    donor_pool = pool()
    other_states = [
        state(quarter=3), state(score=0), state(kickoff=0),
    ]
    untouched = resolve_post_score_q4_2025(
        other_states, donor_pool, rng=np.random.default_rng(0),
    )
    assert all(x["is_kickoff"] == orig["is_kickoff"]
               for x,orig in zip(untouched, other_states))
    special = [state(midfield=True) for _ in range(500)]
    got = resolve_post_score_q4_2025(
        special, donor_pool, rng=np.random.default_rng(44),
    )
    assert any(x["_kickoff_transition_type"] == "touchback" for x in got)
    assert all(x["yardline_100"] == 80.
               for x in got if x["_kickoff_transition_type"] == "touchback")


def test_isolated_engine_globals_restored_and_no_double_draw():
    engine = fake_engine()
    orig = engine["evaluate_post_play_states"]
    parent = state()
    with use_frozen_2025_kickoffs(engine, pool(), seed=45):
        active = orig.__globals__["evaluate_post_play_states"]
        assert active is not orig
        a = active({"qtr": 4}, [parent])[0]
        # A touchdown child carries its resolved state with is_kickoff=0.
        child = state(kickoff=0)
        child["yardline_100"] = a[1]
        b = active({"qtr": 4}, [child])[0]
        assert a[1] == b[1]
        with pytest.raises(RuntimeError, match="already overridden"):
            with use_frozen_2025_kickoffs(engine, pool()):
                pass
    assert orig.__globals__["evaluate_post_play_states"] is orig
    assert parent["is_kickoff"] == 1


def test_isolation_restored_on_exception():
    engine = fake_engine()
    f = engine["evaluate_post_play_states"]
    with pytest.raises(RuntimeError, match="planned"):
        with use_frozen_2025_kickoffs(engine, pool()):
            raise RuntimeError("planned")
    assert f.__globals__["evaluate_post_play_states"] is f


def test_static_loader_rejects_missing_csv(tmp_path):
    from src.kickoff_2025_empirical import load_frozen_pool
    with pytest.raises(FileNotFoundError, match="Missing frozen"):
        load_frozen_pool(tmp_path / "does-not-exist.csv")
