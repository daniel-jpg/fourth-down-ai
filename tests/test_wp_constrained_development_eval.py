"""Synthetic-only QA for chronological constrained-WP evaluation.

No NFL outcome sources, future data, network, retraining or deployed engine.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.wp_constrained_development_eval import (
    DRAWS, GROUPS, MIN_STABLE_GAMES, WP_FEATURES,
    compare, fixed_groups, group_report, paired_game_cluster_ci,
    site_mirror_summary, validate_states,
)
from src.wp_monotonic_candidate import home_advantage_matrix, monotonic_constraints

ROOT = Path(__file__).resolve().parents[1]


class ToyWP:
    """Valid three-class probabilities from pre-snap coordinates only."""
    classes_ = np.array([0, 1, 2])

    def __init__(self, *, home_bias: float):
        self.home_bias = home_bias

    def predict_proba(self, X):
        col = {n: i for i, n in enumerate(WP_FEATURES)}
        margin = X[:, col["home_score_differential"]]
        is_home = X[:, col["is_home_posteam"]]
        to = X[:, col["home_timeouts_remaining"]] - X[:, col["away_timeouts_remaining"]]
        home_p = np.clip(
            .50 + .015*margin + self.home_bias*is_home + .005*to,
            .02, .94,
        )
        tie = np.full(len(X), .02)
        away = 1-home_p-tie
        return np.column_stack([away, tie, home_p])


def fixture():
    template = {
        "game_seconds_remaining": 200., "half_seconds_remaining": 200.,
        "qtr": 4., "down": 1., "ydstogo": 10.,
        "yardline_100": 95., "goal_to_go": 0.,
        "home_score_differential": 4.,
        "home_timeouts_remaining": 3., "away_timeouts_remaining": 3.,
        "is_home_posteam": 1., "is_kickoff": 0., "is_overtime": 0.,
        "final_five_minutes": 1.,
    }
    combos = [
        (2023, 4, 200, 95, 1, 4, 7),
        (2023, 4, 200, 95, 0, -4, -7),
        (2024, 4, 300, 80, 1, 4, 4),
        (2024, 4, 640, 55, 0, 3, -7),
        (2024, 1, 3290, 30, 1, -7, -8),
        (2024, 4, 110, 95, 1, 4, 4),
        (2024, 4, 290, 95, 0, -4, -3),
    ]
    result = []
    for idx, (year, qtr, seconds, own, pos, home_margin, final) in enumerate(combos):
        row = dict(template)
        row.update({
            "game_seconds_remaining": float(seconds),
            "half_seconds_remaining": float(seconds if qtr >= 3 else seconds - 1800),
            "qtr": float(qtr),
            "yardline_100": float(own),
            "is_home_posteam": float(pos),
            "home_score_differential": float(home_margin),
            "final_five_minutes": float(qtr == 4 and seconds <= 300),
            "game_id": f"fabricated-game-{idx//2}",
            "season": year,
            "final_home_margin": float(final),
        })
        result.append(row)
    return pd.DataFrame(result)


def test_prespecified_groups_are_complete_and_territory_uses_offense_axis():
    f = validate_states(fixture())
    m = fixed_groups(f)
    assert set(m) == set(GROUPS)
    assert m["REGULATION_ALL"].sum() == len(f)
    assert m["Q1_TO_Q3"].sum() == 1
    assert m["Q4_FINAL_5_OWN_10_LEADING_4"].sum() == 4
    assert m["Q4_FINAL_5_OWN_10_LEADING_1_TO_8"].sum() == 4
    assert m["REGULATION_2023"].sum() == 2
    assert m["REGULATION_2024"].sum() == 5


def test_comparison_is_a_descriptive_nonpromotable_development_audit():
    frozen, alternative = ToyWP(home_bias=.06), ToyWP(home_bias=.015)
    out = compare(fixture(), frozen, alternative, draws=40)
    assert out["training"] == "2014-22 only"
    assert out["n_states"] == 7
    assert out["n_distinct_games"] == 4
    all_states = out["group_reports"]["REGULATION_ALL"]
    assert all_states["states"] == 7
    assert all_states["status"] == "SPARSE_DESCRIPTIVE_ONLY"
    assert len(all_states["brier_gain_95pct_game_cluster_ci"]) == 2
    assert np.isfinite(all_states["brier_gain_frozen_minus_challenger"])
    mirror = out["q4_site_mirror_sensitivity_not_causal"]["Q4_FINAL_5_OWN_10"]
    assert mirror["states"] == 4
    assert mirror["candidate_home_minus_away_pp_mean"] is not None
    assert out["promotion_gate"]["prospectively_validated"] is False
    assert out["promotion_gate"]["may_replace_frozen_model"] is False
    assert "unchosen GO/FG" in str(out["promotion_gate"])


def test_paired_game_cluster_resampling_is_reproducible_with_repeated_plays():
    gains = [.1, .2, -.1, -.1, .3, .3]
    game_ids = ["g1", "g1", "g2", "g2", "g3", "g3"]
    first = paired_game_cluster_ci(gains, game_ids, draws=80, seed=9)
    assert first == paired_game_cluster_ci(gains, game_ids, draws=80, seed=9)
    assert first[0] <= first[1]
    assert paired_game_cluster_ci([.1], ["g1"], draws=80) is None
    with pytest.raises(ValueError, match="align"):
        paired_game_cluster_ci([.2], ["g1", "g2"])


def test_site_mirror_prediction_diagnostic_is_label_blind():
    f = fixture()
    a = site_mirror_summary(f, ToyWP(home_bias=.08), ToyWP(home_bias=.01))
    f["final_home_margin"] *= -1.
    b = site_mirror_summary(f, ToyWP(home_bias=.08), ToyWP(home_bias=.01))
    assert a == b
    assert set(a) == set(GROUPS)


@pytest.mark.parametrize("field,value,expected", [
    ("season", 2025, "2023-24"),
    ("season", 2026, "2023-24"),
    ("qtr", 5, "Regulation"),
    ("is_kickoff", 1, "Regulation"),
    ("is_home_posteam", float("nan"), "HOME/AWAY"),
    ("home_score_differential", float("nan"), "pre-snap"),
    ("final_home_margin", float("inf"), "home win labels"),
    ("yardline_100", 100., "field position"),
])
def test_invalid_holdout_or_missing_presnap_fields_rejected(field, value, expected):
    f = fixture()
    f.loc[0, field] = value
    with pytest.raises(ValueError, match=expected):
        validate_states(f)


def test_invalid_model_probabilities_cannot_produce_fake_brier_results():
    class Invalid(ToyWP):
        def predict_proba(self, X):
            return np.tile([.6, .3, .5], (len(X), 1))
    with pytest.raises(ValueError, match="valid multiclass"):
        compare(fixture(), Invalid(home_bias=0), ToyWP(home_bias=.01), draws=20)


def test_empty_group_is_explicitly_unsupported():
    r = group_report(np.array([]), np.array([]), np.array([]), np.array([]), draws=20)
    assert r["states"] == 0
    assert r["status"] == "NO_SUPPORT"


def test_existing_monotonic_feature_contract_is_not_changed():
    constraints = monotonic_constraints(list(WP_FEATURES))
    assert len(constraints) == len(WP_FEATURES)
    assert constraints[WP_FEATURES.index("home_score_differential")] == 1
    assert constraints[WP_FEATURES.index("away_timeouts_remaining")] == -1
    f = fixture()
    mat = f[list(WP_FEATURES)].to_numpy(float)
    transformed = home_advantage_matrix(mat, list(WP_FEATURES))
    idx = WP_FEATURES.index("yardline_100")
    pos = WP_FEATURES.index("is_home_posteam")
    assert np.allclose(
        transformed[:, idx], np.where(mat[:, pos] == 1, 100-mat[:, idx], mat[:, idx])
    )


def test_script_uses_only_2014_2024_sources_not_2025_or_future_gates():
    source = (ROOT / "scripts/evaluate_constrained_wp_2014_2024.py").read_text()
    assert 'range(2014, 2025)' in source
    assert "nfl.load_pbp(years)" in source
    assert "nfl.load_pbp([2025" not in source
    assert "nfl.load_pbp([2026" not in source
    assert "models/win_probability_model.joblib" in source
    assert "joblib.dump(candidate, OUTDIR" in source
