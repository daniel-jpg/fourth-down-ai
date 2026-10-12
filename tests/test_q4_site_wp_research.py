"""Pure synthetic checks for development-only Q4 site / WP audit."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.q4_site_wp_research import (
    GROUPS, WP_FEATURES, check_data, classifier_probabilities,
    fixed_situation_masks, gap_game_cluster_ci, mirror_offense_sites,
    summarize,
)

ROOT = Path(__file__).resolve().parents[1]


class FakeWinModel:
    classes_ = np.array([0, 1, 2])

    def predict_proba(self, X):
        f = {name: i for i, name in enumerate(WP_FEATURES)}
        margin = X[:, f["home_score_differential"]]
        home_possession = X[:, f["is_home_posteam"]]
        home_to = X[:, f["home_timeouts_remaining"]]
        away_to = X[:, f["away_timeouts_remaining"]]
        home_prob = np.clip(
            .50 + .035*margin + .02*home_possession
            + .01*(home_to-away_to), .04, .94,
        )
        tie_prob = np.full(len(X), .02)
        home_prob = np.minimum(home_prob, .98-tie_prob)
        away_prob = 1-home_prob-tie_prob
        return np.column_stack((away_prob, tie_prob, home_prob))


def fixture():
    base = {
        "game_seconds_remaining": 200., "half_seconds_remaining": 200.,
        "qtr": 4., "down": 1., "ydstogo": 10.,
        "yardline_100": 95., "goal_to_go": 0.,
        "home_score_differential": 4.,
        "home_timeouts_remaining": 2., "away_timeouts_remaining": 3.,
        "is_home_posteam": 1., "is_kickoff": 0., "is_overtime": 0.,
        "final_five_minutes": 1.,
    }
    records = []
    for idx, (season, is_home, margin, yard) in enumerate([
        (2023, 1, 4, 95),
        (2023, 0, -4, 95),
        (2024, 1, -3, 85),
        (2024, 0, 3, 75),
        (2024, 1, 4, 95),
    ]):
        rec = dict(base)
        rec.update(season=season, game_id=f"g{idx}",
                   is_home_posteam=float(is_home),
                   home_score_differential=float(margin),
                   yardline_100=float(yard),
                   final_home_margin=float(7 if idx % 2 == 0 else -7))
        records.append(rec)
    return pd.DataFrame(records)


def test_site_mirror_preserves_offense_score_timeouts_field_and_clock():
    frame = fixture()
    X = frame[list(WP_FEATURES)].to_numpy(float)
    home, away = mirror_offense_sites(X)
    idx = {x: i for i, x in enumerate(WP_FEATURES)}
    assert np.all(home[:, idx["is_home_posteam"]] == 1)
    assert np.all(away[:, idx["is_home_posteam"]] == 0)
    assert np.all(home[:, idx["yardline_100"]] == X[:, idx["yardline_100"]])
    assert np.all(away[:, idx["game_seconds_remaining"]] == X[:, idx["game_seconds_remaining"]])
    assert np.allclose(
        home[:, idx["home_score_differential"]],
        -away[:, idx["home_score_differential"]],
    )
    assert np.allclose(
        home[:, idx["home_timeouts_remaining"]],
        away[:, idx["away_timeouts_remaining"]],
    )
    # Starting from the mirrored AWAY coordinate returns the same site
    # cases; flipping only is_home_posteam without score/TO axes cannot.
    next_home, next_away = mirror_offense_sites(away)
    assert np.allclose(next_home, home)
    assert np.allclose(next_away, away)


def test_fixed_q4_deep_and_leading_masks_are_consistent():
    f = fixture()
    groups = fixed_situation_masks(f)
    assert set(groups) == set(GROUPS)
    assert int(groups["Q4_FINAL_5"].sum()) == len(f)
    assert int(groups["Q4_FINAL_5_OWN_10"].sum()) == 3
    assert int(groups["Q4_FINAL_5_OWN_10_LEADING_4"].sum()) == 3


def test_factual_calibration_is_separated_from_site_prediction_sensitivity():
    result = summarize(fixture(), FakeWinModel(), draws=40)
    first = result["summary_by_fixed_situation_and_offense_site"]["Q4_FINAL_5_OWN_10"]
    assert first["all"]["states"] == 3
    assert first["all"]["games"] == 3
    assert first["HOME"]["states"] == 2
    assert first["AWAY"]["states"] == 1
    assert first["all"]["status"] == "SPARSE_DESCRIPTIVE_ONLY"
    assert "brier" in first["all"]
    assert "observed_offense_win_rate" in first["all"]
    mirror = result["site_mirror_prediction_sensitivity_not_causal"]
    assert mirror["Q4_FINAL_5_OWN_10"]["states"] == 3
    assert mirror["Q4_FINAL_5_OWN_10"]["home_minus_away_mirrored_pp_p10_p50_p90"]
    assert "2025_outcomes" in result
    assert result["2025_outcomes"] == "NOT READ"


def test_tied_finals_excluded_from_binary_calibration_not_silently_wins():
    f = fixture()
    f.loc[0, "final_home_margin"] = 0
    r = summarize(f, FakeWinModel(), draws=20)
    assert r["final_tied_states_excluded_from_binary_scores"] == 1
    assert r["summary_by_fixed_situation_and_offense_site"]["Q4_ALL"]["all"]["states"] == 4


def test_cluster_ci_reproducible_and_one_game_unavailable():
    values = np.array([-.1, -.1, .05, .2, .2])
    games = np.array(["a", "a", "b", "c", "c"])
    ci = gap_game_cluster_ci(values, games, draws=90, seed=17)
    assert ci == gap_game_cluster_ci(values, games, draws=90, seed=17)
    assert ci[0] <= ci[1]
    assert gap_game_cluster_ci(values[:2], games[:2], draws=25) is None


@pytest.mark.parametrize("key,value,error", [
    ("season", 2025, "2023-2024"),
    ("qtr", 5, "regulation"),
    ("is_home_posteam", float("nan"), "known"),
    ("is_kickoff", 1, "non-kickoff"),
    ("home_score_differential", float("nan"), "nonfinite"),
    ("final_home_margin", float("nan"), "nonfinite"),
    ("yardline_100", 100, "Invalid"),
])
def test_illegal_sources_and_outcome_features_rejected(key, value, error):
    frame = fixture()
    frame.loc[0, key] = value
    with pytest.raises(ValueError, match=error):
        check_data(frame)


def test_invalid_probability_simplex_rejected():
    class Invalid:
        classes_ = [0, 2]
        def predict_proba(self, X):
            return np.ones((len(X), 2))
    X = fixture()[list(WP_FEATURES)].to_numpy(float)
    with pytest.raises(ValueError, match="valid class probabilities"):
        classifier_probabilities(Invalid(), X)


def test_site_historical_driver_fetches_only_development_seasons():
    s = (ROOT / "scripts" / "audit_q4_site_wp_2023_2024.py").read_text()
    assert "nfl.load_pbp([2023, 2024])" in s
    assert "nfl.load_pbp([2025" not in s
    assert "nfl.load_pbp([2026" not in s
    assert "joblib.dump" not in s
