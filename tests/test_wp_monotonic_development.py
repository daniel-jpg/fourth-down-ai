"""Guard fixed-spec constrained WP candidate and chronological report setup."""

from pathlib import Path
import importlib.util

import numpy as np
import pandas as pd
import pytest

from src.wp_monotonic_candidate import (
    ConstrainedRegulationWP,
    home_advantage_matrix,
    monotonic_constraints,
    new_binary_model,
    fit_binary_wp,
)

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / "src/67_train_modern_wp_candidate.py"
spec = importlib.util.spec_from_file_location("modern_wp_training", path)
train = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train)

FEATURES = [
    "game_seconds_remaining", "half_seconds_remaining", "qtr", "down",
    "ydstogo", "yardline_100", "goal_to_go", "home_score_differential",
    "home_timeouts_remaining", "away_timeouts_remaining", "is_home_posteam",
    "is_kickoff", "is_overtime", "final_five_minutes",
]


def simple_state(*, site=1., qtr=4., is_kickoff=0., yard=65.,
                 margin=-3.) -> np.ndarray:
    return np.array([
        100., 100., qtr, 1., 10., yard, 0., margin, 2., 2.,
        site, is_kickoff, float(qtr >= 5), 1.,
    ])


class FakeBinary:
    classes_ = np.array([0, 1])

    def predict_proba(self, X):
        # strictly increasing with transformed field position
        a = np.clip(.25+.004*X[:, FEATURES.index("yardline_100")], .001, .999)
        return np.column_stack([1-a, a])


class FakeFrozen:
    classes_ = np.array([0, 1, 2])

    def predict_proba(self, X):
        n = len(X)
        return np.column_stack([
            np.full(n, .55), np.full(n, .02), np.full(n, .43),
        ])


def test_home_field_position_direction_changes_with_possession():
    home = np.stack([
        simple_state(site=1., yard=80.),
        simple_state(site=1., yard=20.),
    ])
    away = np.stack([
        simple_state(site=0., yard=20.),
        simple_state(site=0., yard=80.),
    ])
    transformed_home = home_advantage_matrix(home, FEATURES)
    transformed_away = home_advantage_matrix(away, FEATURES)
    i = FEATURES.index("yardline_100")
    assert transformed_home[:, i].tolist() == [20., 80.]
    assert transformed_away[:, i].tolist() == [20., 80.]
    assert home[:, i].tolist() == [80., 20.]  # not mutated


def test_monotonic_constraints_match_football_invariants():
    constraints = monotonic_constraints(FEATURES)
    assert len(constraints) == len(FEATURES)
    assert constraints[FEATURES.index("yardline_100")] == 1
    assert constraints[FEATURES.index("home_score_differential")] == 1
    assert constraints[FEATURES.index("home_timeouts_remaining")] == 1
    assert constraints[FEATURES.index("away_timeouts_remaining")] == -1
    assert constraints[FEATURES.index("is_home_posteam")] == 0
    with pytest.raises(ValueError, match="Duplicate"):
        monotonic_constraints(FEATURES + ["qtr"])
    with pytest.raises(ValueError, match="Missing required"):
        monotonic_constraints(FEATURES[:-6])
    with pytest.raises(ValueError, match="ordered feature"):
        home_advantage_matrix(np.zeros((3, 4)), FEATURES)


def test_three_class_candidate_preserves_frozen_ties_and_unmodified_kickoffs():
    candidate = ConstrainedRegulationWP(FakeBinary(), FakeFrozen(), FEATURES)
    reg_home = simple_state()
    kickoff = simple_state(is_kickoff=1)
    overtime = simple_state(qtr=5)
    reg_away = simple_state(site=0)
    matrix = np.vstack([reg_home, kickoff, overtime, reg_away])
    probs = candidate.predict_proba(matrix)
    np.testing.assert_allclose(probs.sum(axis=1), 1)
    np.testing.assert_allclose(probs[1], [.55, .02, .43])
    np.testing.assert_allclose(probs[2], [.55, .02, .43])
    np.testing.assert_allclose(probs[[0, 3], 1], [.02, .02])
    assert probs[0, 2] != pytest.approx(.43)


def test_candidate_home_wp_moves_with_offensive_progress_for_both_teams():
    candidate = ConstrainedRegulationWP(FakeBinary(), FakeFrozen(), FEATURES)
    home = np.vstack([
        simple_state(site=1, yard=80),
        simple_state(site=1, yard=20),
    ])
    away = np.vstack([
        simple_state(site=0, yard=20),
        simple_state(site=0, yard=80),
    ])
    assert candidate.predict_proba(home)[0, 2] < candidate.predict_proba(home)[1, 2]
    assert candidate.predict_proba(away)[0, 2] < candidate.predict_proba(away)[1, 2]


def test_trainer_emits_expected_score_and_timeouts_without_labels():
    raw = pd.DataFrame([
        {
            "season": 2024, "season_type": "REG", "game_id": "test",
            "play_type": "run", "qtr": 4, "game_seconds_remaining": 100,
            "half_seconds_remaining": 100, "down": 1, "ydstogo": 10,
            "yardline_100": 70, "goal_to_go": 0,
            "posteam_score": 21, "defteam_score": 24,
            "posteam_timeouts_remaining": 2,
            "defteam_timeouts_remaining": 1,
            "posteam": "AWY", "home_team": "HOM", "result": -3,
        },
        {
            "season": 2024, "season_type": "REG", "game_id": "test",
            "play_type": "kickoff", "qtr": 4, "game_seconds_remaining": 90,
            "half_seconds_remaining": 90, "down": np.nan, "ydstogo": 0,
            "yardline_100": 35, "goal_to_go": 0,
            "posteam_score": 21, "defteam_score": 24,
            "posteam_timeouts_remaining": 2,
            "defteam_timeouts_remaining": 1,
            "posteam": "AWY", "home_team": "HOM", "result": -3,
        },
    ])
    row = train.usable_regulation_states(raw, FEATURES)
    assert len(row) == 1
    assert row["home_score_differential"].iloc[0] == pytest.approx(3)
    assert row["home_timeouts_remaining"].iloc[0] == 1
    assert row["away_timeouts_remaining"].iloc[0] == 2
    assert row["is_home_posteam"].iloc[0] == 0
    assert row["final_five_minutes"].iloc[0] == 1
    assert row["final_home_margin"].iloc[0] == -3


def test_game_level_confidence_intervals_are_repeatable():
    data = np.array([-.1, .2, .1, -.05, .15])
    games = np.array(["a", "b", "a", "c", "a"])
    one = train.cluster_brier_ci(data, games, n_boot=200)
    two = train.cluster_brier_ci(data, games, n_boot=200)
    assert one == two
    assert one[0] <= one[1]
    with pytest.raises(ValueError, match="align"):
        train.cluster_brier_ci(data[:-1], games)


def test_classifier_model_spec_supports_constraints():
    model = new_binary_model(FEATURES)
    assert list(model.monotonic_cst) == monotonic_constraints(FEATURES)
    n = 240
    samples = np.vstack([
        simple_state(site=(i % 2), yard=10.+(i % 90),
                     margin=float((i % 19)-9))
        for i in range(n)
    ])
    labels = np.where(np.arange(n) % 2 == 0, 7., -3.)
    games = np.array([f"G{i//8}" for i in range(n)])
    fitted = fit_binary_wp(samples, labels, FEATURES, games)
    p = fitted.predict_proba(home_advantage_matrix(samples, FEATURES))
    assert p.shape == (n, 2)
    assert np.isfinite(p).all()


def test_source_does_not_read_2026_or_overwrite_production_artifacts():
    source = path.read_text()
    assert "nfl.load_pbp([year])" in source
    assert "range(TRAIN_START, TEST_YEAR + 1)" in source
    assert "joblib.dump(candidate, ARTIFACT_DIR /" in source
    assert "joblib.dump(candidate, BASE_MODEL)" not in source
