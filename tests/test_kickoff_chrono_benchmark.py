import numpy as np
import pandas as pd
import pytest

from src.kickoff_chrono_benchmark import (
    FOLDS, benchmark, candidate_predictions,
    smooth_features, summarize_predictions, regular_season_kickoffs,
)


def synthetic():
    rows = []
    for week in range(1, 19):
        for j in range(15):
            margin = [-10, -4, -3, 0, 1, 3, 9][j % 7]
            time = 40 + (week * 23 + j * 44) % 840
            tb = int((week * 13 + j * 17) % 100 < 22 + int(margin < -3) * 24)
            rows.append(dict(season=2025, week=week, game_id=f"G{week}-{j//3}",
                             kick_seconds=time, receiver_margin=margin,
                             status="touchback" if tb else "return"))
    return pd.DataFrame(rows)


def test_chronology_and_no_validation_reuse():
    result = benchmark(synthetic())
    assert result["week"].gt(result["train_end"]).all()
    assert result["week"].between(9, 18).all()
    assert len(result) == 15 * 10
    assert result.groupby("week").size().eq(15).all()
    assert result["val_block"].nunique() == len(FOLDS)


def test_fold_constant_uses_only_prior_data():
    sample = synthetic()
    result = benchmark(sample)
    for train_end, start, end in FOLDS:
        expected = sample.loc[sample["week"].le(train_end), "status"].eq("touchback").mean()
        block = result.loc[result["week"].between(start, end)]
        assert np.allclose(block["constant"], expected)


def test_continuous_score_features_have_no_category_cliff():
    data = pd.DataFrame({"receiver_margin": [-4, -3], "kick_seconds": [299, 299]})
    x = smooth_features(data, "score_clock")
    assert x.shape == (2, 3)
    assert np.isclose(abs(x[0, 0] - x[1, 0]), 1/7)
    assert np.isclose(abs(x[0, 1] - x[1, 1]), 1/7)
    assert x[0, 2] == x[1, 2]


def test_2026_and_lookahead_are_rejected():
    sample = synthetic()
    with pytest.raises(ValueError, match="2025"):
        benchmark(sample.assign(season=2026))
    with pytest.raises(ValueError, match="Chronological leakage"):
        candidate_predictions(sample.loc[sample["week"] <= 10],
                              sample.loc[sample["week"] == 10], "score")
    with pytest.raises(ValueError, match="weeks 1-18"):
        benchmark(sample.assign(week=19))


def test_scoring_and_bootstrap_are_reproducible():
    result = benchmark(synthetic())
    a = summarize_predictions(result, bootstrap_repeats=100, seed=9)
    b = summarize_predictions(result, bootstrap_repeats=100, seed=9)
    pd.testing.assert_frame_equal(a, b)
    assert set(a["candidate"]) == {"constant", "score", "clock", "score_clock"}
    assert a.loc[a["candidate"] == "constant", "improvement_vs_constant"].item() == 0
    assert a["brier"].between(0, 1).all()


def test_missing_or_mixed_seasons_in_pbp_rejected():
    pairs = pd.DataFrame({"season": [2025], "game_id": ["G"], "kick_play_id": [1],
                          "kick_qtr": [4], "status": ["touchback"], "kick_seconds": [100],
                          "receiver_margin": [1]})
    pbp = pd.DataFrame({"season": [2026], "game_id": ["G"], "play_id": [1],
                        "play_type": ["kickoff"], "season_type": ["REG"], "week": [1]})
    with pytest.raises(ValueError, match="2026 is excluded"):
        regular_season_kickoffs(pairs, pbp)


def test_postseason_excluded_at_metadata_join():
    pairs = pd.DataFrame({
        "season": [2025, 2025], "game_id": ["G1", "G2"], "kick_play_id": [1, 1],
        "kick_qtr": [4, 4], "status": ["touchback", "return"],
        "kick_seconds": [100, 100], "receiver_margin": [1, -2],
    })
    pbp = pd.DataFrame({
        "season": [2025, 2025], "game_id": ["G1", "G2"], "play_id": [1, 1],
        "play_type": ["kickoff", "kickoff"], "season_type": ["REG", "POST"],
        "week": [12, 19],
    })
    out = regular_season_kickoffs(pairs, pbp)
    assert len(out) == 1 and out.iloc[0]["week"] == 12
