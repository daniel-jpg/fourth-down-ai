import numpy as np
import pandas as pd
import pytest

from src.kickoff_chrono_benchmark import smooth_features
from src.kickoff_touchback_model import ContextTouchbackModel
from src import kickoff_chrono_benchmark as chrono

import runpy
module = runpy.run_path("src/59_evaluate_2026_smooth_kickoff.py")
fit = module["fit_2025_regular_score_clock"]
check = module["early_2026_regular_kickoffs"]
compare = module["compare_2026"]


def example_train():
    rows = []
    for week in range(1, 19):
        for j in range(16):
            margin = (-7, -4, -3, 0, 1, 3, 7)[j % 7]
            clock = (week*49 + j*51) % 900
            outcome = "touchback" if (j*23+week*11)%100 < 32 else "return"
            rows.append({"season": 2025, "week": week, "game_id": f"25-{week}-{j//2}",
                         "status": outcome, "receiver_margin": margin,
                         "kick_seconds": clock})
    return pd.DataFrame(rows)


def example_test():
    rows = []
    for week in range(1, 6):
        for j in range(10):
            margin = (-8, -3, 0, 1, 5)[j % 5]
            rows.append({"season": 2026, "week": week, "game_id": f"26-{week}-{j//2}",
                         "status": "touchback" if (week+j)%4 == 0 else "return",
                         "receiver_margin": margin, "kick_seconds": (j*78)%900})
    return pd.DataFrame(rows)


def test_fit_is_2025_only_and_predictions_are_finite():
    model, rate = fit(example_train())
    assert 0 < rate < 1
    p = model.predict_proba(smooth_features(example_test(), "score_clock"))[:, 1]
    assert np.all((p > 0) & (p < 1))
    with pytest.raises(ValueError, match="2025 REG"):
        fit(example_train().assign(season=2026))


def test_2026_comparison_is_deterministic():
    model, rate = fit(example_train())
    one = compare(model, rate, example_test(), bootstrap_repeats=100, seed=6)
    two = compare(model, rate, example_test(), bootstrap_repeats=100, seed=6)
    assert one == two
    assert one["n"] == 50
    assert one["games"] == 25
    assert one["ci_low"] <= one["ci_high"]


def test_later_2026_weeks_are_excluded():
    model, rate = fit(example_train())
    with pytest.raises(ValueError, match="weeks 1-5"):
        compare(model, rate, example_test().assign(week=6))


def test_2025_evaluation_season_is_rejected():
    model, rate = fit(example_train())
    with pytest.raises(ValueError, match="2026"):
        compare(model, rate, example_train())


def test_2026_week_metadata_and_postseason_filtered():
    pbp = pd.DataFrame([
        {"season":2026,"game_id":"g1","play_id":1,"play_type":"kickoff","season_type":"REG","week":4},
        {"season":2026,"game_id":"g2","play_id":2,"play_type":"kickoff","season_type":"REG","week":6},
        {"season":2026,"game_id":"g3","play_id":3,"play_type":"kickoff","season_type":"POST","week":1},
    ])
    pairs = pd.DataFrame([
        {"season":2026,"game_id":"g1","kick_play_id":1,"kick_qtr":4,"status":"return","receiver_margin":-3,"kick_seconds":250},
        {"season":2026,"game_id":"g2","kick_play_id":2,"kick_qtr":4,"status":"touchback","receiver_margin":1,"kick_seconds":250},
        {"season":2026,"game_id":"g3","kick_play_id":3,"kick_qtr":4,"status":"return","receiver_margin":1,"kick_seconds":250},
    ])
    result = check(pairs, pbp)
    assert result["game_id"].tolist() == ["g1"]
