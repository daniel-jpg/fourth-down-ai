import numpy as np
import pandas as pd
import pytest

from src.kickoff_transition import build_2024_pool, resolve_q4_kickoffs


def fake_pool(tb_own=30.0, return_own=28.0, return_clock=6.0):
    rows = []
    for j in range(100):
        rows.append(dict(season=2024, game_id=f"g{j}", quarter=4,
                         transition_type="touchback", own_yardline=tb_own,
                         elapsed_seconds=0.0))
        rows.append(dict(season=2024, game_id=f"g{j}", quarter=4,
                         transition_type="return", own_yardline=return_own,
                         elapsed_seconds=return_clock))
    return pd.DataFrame(rows)


def state(*, quarter=4, scoring=3.0, kickoff=1.0, seconds=289.0, midfield=False):
    return {
        "qtr": quarter, "_score_change_original": scoring,
        "_kickoff_from_50": midfield, "is_kickoff": kickoff,
        "game_seconds_remaining": seconds,
        "half_seconds_remaining": seconds,
        "yardline_100": np.nan, "down": np.nan,
        "ydstogo": np.nan, "goal_to_go": 0,
        "final_five_minutes": float(quarter == 4 and seconds <= 300),
        "home_score_differential": -1.0,
        "is_home_posteam": 1.0,
        "_touchdown_original": 0,
    }


def test_direct_touchback_uses_35_after_2024():
    base = state()
    got = resolve_q4_kickoffs([base], fake_pool(), touchback_rate=1,
                              rng=np.random.default_rng(1), rules_year=2026)[0]
    assert got["yardline_100"] == 65.0
    assert got["game_seconds_remaining"] == 289.0
    assert got["down"] == 1
    assert got["is_kickoff"] == 0
    assert base["is_kickoff"] == 1  # no in-place mutations


def test_landing_zone_touchback_at_20():
    got = resolve_q4_kickoffs([state()], fake_pool(tb_own=20),
                              touchback_rate=1, rng=np.random.default_rng(1))[0]
    assert got["yardline_100"] == 80


def test_2026_midfield_touchback_at_20():
    got = resolve_q4_kickoffs([state(midfield=True)], fake_pool(),
                              touchback_rate=1, rng=np.random.default_rng(1))[0]
    assert got["yardline_100"] == 80


def test_return_consumes_real_clock_and_preserves_score():
    orig = state()
    got = resolve_q4_kickoffs([orig], fake_pool(return_own=31, return_clock=8),
                              touchback_rate=0, rng=np.random.default_rng(2))[0]
    assert got["yardline_100"] == 69
    assert got["game_seconds_remaining"] == 281
    assert got["half_seconds_remaining"] == 281
    assert got["home_score_differential"] == orig["home_score_differential"]
    assert got["_kickoff_transition_type"] == "return"


def test_ordinary_non_scoring_and_non_q4_states_unchanged():
    originals = [state(scoring=0), state(quarter=3), state(kickoff=0)]
    results = resolve_q4_kickoffs(originals, fake_pool(), touchback_rate=1,
                                  rng=np.random.default_rng(0))
    assert all(result["is_kickoff"] == orig["is_kickoff"]
               for result, orig in zip(results, originals))


def test_invalid_rates_and_heldout_pool_are_rejected():
    with pytest.raises(ValueError):
        resolve_q4_kickoffs([state()], fake_pool(), touchback_rate=1.1,
                            rng=np.random.default_rng(0))
    pool = fake_pool()
    pool["season"] = 2025
    with pytest.raises(ValueError):
        resolve_q4_kickoffs([state()], pool, touchback_rate=0.5,
                            rng=np.random.default_rng(0))


def test_reproducible_seed():
    states = [state() for _ in range(80)]
    x = resolve_q4_kickoffs(states, fake_pool(), touchback_rate=0.35,
                            rng=np.random.default_rng(41))
    y = resolve_q4_kickoffs(states, fake_pool(), touchback_rate=0.35,
                            rng=np.random.default_rng(41))
    assert [(s["yardline_100"], s["game_seconds_remaining"]) for s in x] == [
        (s["yardline_100"], s["game_seconds_remaining"]) for s in y
    ]


def test_builder_uses_receiver_and_pairs_next_snap():
    # One touchback and one genuine return in the same game.
    pbp = pd.DataFrame([
        dict(game_id="g1", play_id=10, season=2024, play_type="kickoff",
             qtr=4, game_seconds_remaining=300, posteam="SEA",
             yardline_100=np.nan, down=np.nan, touchback=1, desc="kickoff"),
        dict(game_id="g1", play_id=11, season=2024, play_type="run",
             qtr=4, game_seconds_remaining=300, posteam="SEA",
             yardline_100=70, down=1, touchback=np.nan, desc="run"),
        dict(game_id="g1", play_id=20, season=2024, play_type="kickoff",
             qtr=4, game_seconds_remaining=250, posteam="LAR",
             yardline_100=np.nan, down=np.nan, touchback=0, desc="kickoff"),
        dict(game_id="g1", play_id=21, season=2024, play_type="pass",
             qtr=4, game_seconds_remaining=244, posteam="LAR",
             yardline_100=72, down=1, touchback=np.nan, desc="pass"),
    ])
    pool = build_2024_pool(pbp)
    assert len(pool) == 2
    assert list(pool["transition_type"]) == ["touchback", "return"]
    assert list(pool["own_yardline"]) == [30, 28]
    assert list(pool["elapsed_seconds"]) == [0, 6]


def test_builder_rejects_2025_data():
    df = pd.DataFrame({
        "game_id": ["g"], "play_id": [1], "season": [2025],
        "play_type": ["kickoff"], "qtr": [4],
        "game_seconds_remaining": [300], "posteam": ["A"],
        "yardline_100": [np.nan], "down": [np.nan],
        "touchback": [1], "desc": ["kickoff"],
    })
    with pytest.raises(ValueError):
        build_2024_pool(df)
