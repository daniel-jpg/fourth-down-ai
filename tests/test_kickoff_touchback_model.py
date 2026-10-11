import runpy

import numpy as np
import pandas as pd
import pytest

from src.kickoff_touchback_model import (
    ContextTouchbackModel, feature_matrix, ordinary_q4,
    fit_2025_model, summarize_holdout,
)


def synthetic(season=2025):
    rows = []
    groups = [(1, 0.12), (-3, 0.21), (-7, 0.54), (0, 0.22), (7, 0.23)]
    for group_index, (margin, rate) in enumerate(groups):
        for k in range(100):
            time = 180 if k % 2 else 610
            threshold = int(100 * (rate + (0.03 if time <= 300 else 0)))
            rows.append({
                "season": season, "game_id": f"g{k//2}",
                "kick_qtr": 4, "kick_seconds": time,
                "receiver_margin": margin,
                "status": "touchback" if (k % 100) < threshold else "return",
            })
    return pd.DataFrame(rows)


def test_fit_is_context_sensitive_and_deterministic():
    data = synthetic()
    a = fit_2025_model(data)
    b = fit_2025_model(data)
    assert isinstance(a, ContextTouchbackModel)
    assert a.training_count == 500
    assert 0 < a.training_touchback_rate < 1
    assert a.probability(-7, 200) > a.probability(1, 200)
    assert 0 < a.probability(-3, 299) < 1
    assert a.probability(-7, 200) == b.probability(-7, 200)


def test_training_2026_is_rejected():
    with pytest.raises(ValueError, match="2026 is holdout"):
        fit_2025_model(synthetic(2026))
    mixed = pd.concat([synthetic(), synthetic(2026)], ignore_index=True)
    with pytest.raises(ValueError, match="2026 is holdout"):
        fit_2025_model(mixed)


def test_q4_excludes_onside_and_unmatched_and_missing_margins():
    data = synthetic()
    data.loc[0, "status"] = "onside"
    data.loc[1, "status"] = "other/unmatched"
    data.loc[2, "receiver_margin"] = np.nan
    data.loc[3, "kick_qtr"] = 3
    clean = ordinary_q4(data)
    assert len(clean) == len(data) - 4
    assert set(clean["status"].unique()) <= {"touchback", "return"}


def test_features_and_incorrect_clock_are_rejected():
    one = pd.DataFrame({"receiver_margin": [-4, 1], "kick_seconds": [299, 800]})
    X = feature_matrix(one)
    assert X.loc[0, "trailing_4plus_in_last_five"] == 1
    assert X.loc[1, "trailing_4plus_in_last_five"] == 0
    assert X.shape == (2, 8)
    with pytest.raises(ValueError, match="Q4 clock"):
        feature_matrix(pd.DataFrame({"receiver_margin": [1], "kick_seconds": [999]}))


def test_evaluation_only_2026_and_game_cluster_ci():
    model = fit_2025_model(synthetic())
    result = summarize_holdout(model, synthetic(2026), bootstrap_repeats=100, seed=7)
    assert result["n"] == 500
    assert result["games"] == 50
    assert np.isfinite(result["constant_brier"])
    assert np.isfinite(result["contextual_brier"])
    assert result["improvement_ci"][0] <= result["improvement_ci"][1]
    with pytest.raises(ValueError, match="only 2026"):
        summarize_holdout(model, synthetic(), bootstrap_repeats=0)


def test_2026_kickoff_audit_accepts_35_and_20_touchbacks():
    pair = runpy.run_path("src/55_audit_2025_kickoffs.py")["audit_kickoff_pairs"]
    pbp = pd.DataFrame([
        dict(season=2026, game_id="X", play_id=1, play_type="kickoff", qtr=4,
             game_seconds_remaining=300, posteam="A", yardline_100=np.nan,
             down=np.nan, touchback=1, desc="end zone touchback"),
        dict(season=2026, game_id="X", play_id=2, play_type="run", qtr=4,
             game_seconds_remaining=300, posteam="A", yardline_100=65,
             down=1, touchback=np.nan, desc="first play"),
        dict(season=2026, game_id="X", play_id=3, play_type="kickoff", qtr=4,
             game_seconds_remaining=250, posteam="B", yardline_100=np.nan,
             down=np.nan, touchback=1, desc="landing zone touchback"),
        dict(season=2026, game_id="X", play_id=4, play_type="pass", qtr=4,
             game_seconds_remaining=250, posteam="B", yardline_100=80,
             down=1, touchback=np.nan, desc="first play"),
    ])
    out = pair(pbp, 2026)
    assert out["status"].tolist() == ["touchback", "touchback"]
    assert out["own"].tolist() == [35, 20]
