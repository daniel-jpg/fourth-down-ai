"""Tests for pre-kickoff chronological continuation, not production integration."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

path = Path(__file__).resolve().parents[1] / "src/65_crossfit_2025_continuation_wp.py"
spec = importlib.util.spec_from_file_location("crossfit_wp", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

FEATURES = [
    "game_seconds_remaining", "half_seconds_remaining", "qtr", "down",
    "ydstogo", "yardline_100", "goal_to_go", "home_score_differential",
    "home_timeouts_remaining", "away_timeouts_remaining", "is_home_posteam",
    "is_kickoff", "is_overtime", "final_five_minutes",
]


class FakeWP:
    classes_ = np.array([0, 2])

    def predict_proba(self, values):
        # Does not require the actual kickoff return or final game outcome.
        own = 100 - values[:, FEATURES.index("yardline_100")]
        clock = values[:, FEATURES.index("game_seconds_remaining")]
        p = np.clip(.20 + .002*own + .0001*(900-clock), .01, .99)
        return np.column_stack([1-p, p])


def kick_frame():
    return pd.DataFrame([{
        "game_seconds_remaining": 299., "half_seconds_remaining": 299.,
        "qtr": 4., "down": np.nan, "ydstogo": 0., "yardline_100": 35.,
        "goal_to_go": 0., "home_score_differential": -1.,
        "home_timeouts_remaining": 3., "away_timeouts_remaining": 3.,
        "is_home_posteam": 1., "is_kickoff": 1., "is_overtime": 0.,
        "final_five_minutes": 1.,
    }])


def make_pairs():
    rows = []
    for week in range(1, 19):
        for j in range(8):
            tb = (j % 2 == 0)
            own = 35 if tb else 29
            elapsed = 0 if tb else 6
            rows.append({
                "game_id": f"G{week}-{j//2}",
                "week": week,
                "touchback": int(tb),
                "yardline_100_snap": 100-own,
                "game_seconds_remaining_kick": 300.,
                "game_seconds_remaining_snap": 300.-elapsed,
            })
    return pd.DataFrame(rows)


def test_pre_kickoff_expectation_is_exact_donor_mean():
    donors = pd.DataFrame({
        "own": [20., 35., 25.],
        "elapsed": [0., 0., 6.],
        "touchback": [True, True, False],
    })
    p = module.expected_first_snap_home_wp(
        FakeWP(), FEATURES, kick_frame(), donors,
    )
    expected = np.mean([
        .20 + .002*20 + .0001*(900-299),
        .20 + .002*35 + .0001*(900-299),
        .20 + .002*25 + .0001*(900-293),
    ])
    assert p.shape == (1,)
    assert p[0] == pytest.approx(expected)


def test_batches_and_per_kick_score_remain_separate():
    donors = pd.DataFrame({
        "own": [35., 29.], "elapsed": [0., 6.],
        "touchback": [True, False],
    })
    kicks = pd.concat([kick_frame() for _ in range(4)], ignore_index=True)
    kicks.loc[1, "game_seconds_remaining"] = 260.
    kicks.loc[1, "half_seconds_remaining"] = 260.
    a = module.expected_first_snap_home_wp(
        FakeWP(), FEATURES, kicks, donors, max_kicks_per_batch=1,
    )
    b = module.expected_first_snap_home_wp(
        FakeWP(), FEATURES, kicks, donors, max_kicks_per_batch=4,
    )
    np.testing.assert_allclose(a, b)
    assert a[0] != a[1]


def test_chronological_leakage_and_frozen_2025_donors():
    all_pairs = make_pairs()
    prior = all_pairs.loc[all_pairs["week"] <= 8]
    valid = all_pairs.loc[all_pairs["week"].between(9, 12)]
    donors = module.earlier_donors(prior, valid)
    assert len(donors) == len(prior)
    assert 0 < donors["touchback"].mean() < 1
    with pytest.raises(ValueError, match="Chronological leakage"):
        module.earlier_donors(
            all_pairs.loc[all_pairs["week"] <= 9], valid
        )
    with pytest.raises(ValueError, match="Game leakage"):
        module.earlier_donors(
            prior, valid.assign(game_id=prior["game_id"].iloc[0])
        )


def test_attachment_of_kickoff_weeks_rejects_non_2025_and_late_weeks():
    pairs = pd.DataFrame([
        {"game_id": "A", "play_id_kick": 1, "game_seconds_remaining_kick": 299.},
        {"game_id": "B", "play_id_kick": 2, "game_seconds_remaining_kick": 20.},
    ])
    pbp = pd.DataFrame([
        {"game_id": "A", "play_id": 1, "week": 9, "season": 2025,
         "season_type": "REG", "play_type": "kickoff"},
        {"game_id": "B", "play_id": 2, "week": 10, "season": 2025,
         "season_type": "REG", "play_type": "kickoff"},
    ])
    out = module.with_kickoff_weeks(pairs, pbp)
    assert len(out) == 1 and out.iloc[0]["week"] == 9
    with pytest.raises(ValueError, match="2025"):
        module.with_kickoff_weeks(pairs, pbp.assign(season=2026))


def test_cluster_bootstrap_is_reproducible_and_guarded():
    values = np.array([0.1, -0.1, 0.2, 0.1])
    ids = np.array(["A", "A", "B", "C"])
    a = module.game_cluster_ci(values, ids, repeats=150, seed=23)
    b = module.game_cluster_ci(values, ids, repeats=150, seed=23)
    assert a == b
    assert a[0] <= a[1]
    with pytest.raises(ValueError, match="must align"):
        module.game_cluster_ci(values[:-1], ids)


def test_invalid_donors_or_features_rejected():
    donors = pd.DataFrame({"own": [35.], "elapsed": [0.], "touchback": [True]})
    with pytest.raises(ValueError, match="Missing required WP"):
        module.expected_first_snap_home_wp(
            FakeWP(), FEATURES[:-1], kick_frame(), donors,
        )
    with pytest.raises(ValueError, match="max_kicks_per_batch"):
        module.expected_first_snap_home_wp(
            FakeWP(), FEATURES, kick_frame(), donors, max_kicks_per_batch=0,
        )
