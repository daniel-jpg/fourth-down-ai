"""Synthetic-only proof of the GO challenger scope, data fences and metrics."""
from __future__ import annotations

import ast
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from src.field_position_go_research import (
    CHALLENGER_FEATURES, FROZEN_FEATURES,
    adjusted_run_probability, benchmark, extra_go_features,
    paired_game_cluster_ci, prepared_data, split_dev, support_counts,
    train_candidate,
)

ROOT = Path(__file__).resolve().parents[1]


def rows():
    out = []
    for i, (year, action, outcome, yardline) in enumerate([
        (2019, "NORMAL_GO_PASS", 1, 60),
        (2019, "NORMAL_GO_RUN", 0, 90),
        (2021, "NORMAL_GO_PASS", 0, 95),
        (2022, "FAKE_PUNT_RUN", 1, 42),
        (2023, "NORMAL_GO_PASS", 1, 85),
        (2023, "NORMAL_GO_RUN", 0, 40),
        (2024, "NORMAL_GO_RUN", 1, 70),
        (2024, "NORMAL_GO_PASS", 0, 60),
    ]):
        item = extra_go_features(
            {"yardline_100": float(yardline)},
            is_pass=action.endswith("_PASS"),
        )
        item.update({
            "season": year,
            "split": "train" if year <= 2022 else "validation",
            "game_id": f"synthetic_{i}",
            "action": action,
            "converted": outcome,
        })
        out.append(item)
    return pd.DataFrame(out)


def test_frozen_feature_vector_matches_the_real_engine_without_import_side_effects():
    source = (ROOT / "src" / "42_build_decision_engine.py").read_text()
    tree = ast.parse(source)
    assignments = [
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "GO_FEATURES"
                for t in node.targets)
    ]
    assert len(assignments) == 1
    assert tuple(assignments[0]) == FROZEN_FEATURES
    assert CHALLENGER_FEATURES == FROZEN_FEATURES + ("yardline_100",)

    frozen = joblib.load(ROOT / "models" / "fake_conversion_model.joblib")
    assert frozen.n_features_in_ == len(FROZEN_FEATURES)


def test_train_only_2014_22_score_only_2023_24():
    train, val = split_dev(rows())
    assert set(train["season"]) == {2019, 2021, 2022}
    assert set(val["season"]) == {2023, 2024}
    model = train_candidate(train)
    out = model.predict_proba(val[list(CHALLENGER_FEATURES)].to_numpy(float))
    assert out.shape == (4, 2)
    assert np.allclose(out.sum(axis=1), 1)
    assert np.isfinite(out).all()


@pytest.mark.parametrize("year", [2013, 2025, 2026, 2027])
def test_no_future_or_pre2014_rows(year):
    df = rows()
    df.loc[0, "season"] = year
    with pytest.raises(ValueError, match="2014-22 train"):
        prepared_data(df)


def test_wrong_chronological_split_and_game_leakage_fail_closed():
    df = rows()
    df.loc[0, "split"] = "validation"
    with pytest.raises(ValueError, match="2014-22 train"):
        split_dev(df)
    df = rows()
    df.loc[4, "game_id"] = "synthetic_0"
    with pytest.raises(ValueError, match="Game leakage"):
        split_dev(df)


@pytest.mark.parametrize("feature", ["yardline_100", "ydstogo", "goal_to_go"])
def test_missing_or_nonfinite_pre_snap_feature_rejected(feature):
    df = rows()
    df.loc[1, feature] = float("nan")
    with pytest.raises(ValueError, match="Pre-snap feature"):
        prepared_data(df)


def test_support_counts_and_zero_own5_is_not_a_verified_policy_effect():
    train, val = split_dev(rows())
    out = support_counts(train, val)
    small = [x for x in out
             if x["territory"] == "OWN_1_TO_5" and x["period"] == "train_2014_2022"]
    assert len(small) == 1
    assert small[0]["fourth_and_2_or_3_count"] == 1
    # Synthetic input deliberately has an own-5 pass; actual development
    # data are independently checked by the research audit.
    assert small[0]["pass_count"] == 1
    assert sum(x["fourth_and_2_or_3_count"] >= 0 for x in out) == len(out)


def test_clustered_brier_is_deterministic_resampling_games_not_plays():
    b = np.array([.25, .25, .1, .4, .4, .4])
    c = np.array([.2, .2, .12, .3, .3, .3])
    games = ["a", "a", "b", "c", "c", "c"]
    lo1 = paired_game_cluster_ci(b, c, games, draws=200, seed=19)
    lo2 = paired_game_cluster_ci(b, c, games, draws=200, seed=19)
    assert lo1 == lo2
    assert lo1[0] <= lo1[1]
    assert paired_game_cluster_ci(b[:2], c[:2], ["a", "a"], draws=50) is None
    with pytest.raises(ValueError, match="align"):
        paired_game_cluster_ci(b, c[:-1], games)


def test_benchmark_produces_paired_metrics_on_factual_observed_labels():
    _, val = split_dev(rows())
    old = np.array([.3, .6, .4, .7])
    new = np.array([.6, .4, .5, .4])
    result = benchmark(val, old, new, seed=7, draws=40)
    assert result["decisions"] == 4
    assert result["distinct_games"] == 4
    assert result["challenger_brier"] < result["frozen_brier"]
    assert result["brier_gain_frozen_minus_challenger"] > 0
    assert result["brier_gain_95pct_game_cluster_ci"] is not None
    assert np.isfinite([result["frozen_log_loss"],
                        result["challenger_log_loss"]]).all()


def test_existing_run_adjustment_is_applied_equally_not_to_passes():
    p = .54
    assert adjusted_run_probability(p, "NORMAL_GO_RUN", 3) > p
    assert adjusted_run_probability(p, "NORMAL_GO_RUN", 4) == p
    assert adjusted_run_probability(p, "NORMAL_GO_PASS", 3) == p


def test_model_research_source_excludes_protected_outcomes():
    source = (ROOT / "scripts" / "audit_go_field_position_challenger.py").read_text()
    assert '("season", "<=", 2024)' in source
    assert "nfl.load_pbp" not in source
    assert "requests.get" not in source
    assert "models/win_probability_model.joblib" not in source
