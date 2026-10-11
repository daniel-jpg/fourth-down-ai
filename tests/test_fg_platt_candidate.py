"""Fixed FG probability recalibration remains chronological and monotone."""

from pathlib import Path
import importlib.util

import numpy as np
import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / (
    "src/70_evaluate_fg_platt_candidate.py"
)
spec = importlib.util.spec_from_file_location("fg_platt_fixed", SCRIPT)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def mock_attempts(year=2023, n=800):
    rng = np.random.default_rng(5 + year)
    p = np.linspace(.25, .98, n)
    actual = (rng.random(n) < p).astype(int)
    return pd.DataFrame({
        "season": year,
        "game_id": [f"{year}-g{i//4}" for i in range(n)],
        "action": "FIELD_GOAL",
        "predicted": p,
        "actual": actual,
        "fg_distance": np.linspace(20, 60, n),
        "qtr": 4,
        "seconds": np.linspace(0, 600, n),
    })


def test_fixed_calibrator_never_trains_on_examined_2025():
    model = m.fit_fixed_calibrator(mock_attempts(2023))
    assert model["fit_years"] == [2023]
    assert model["slope"] > 0
    assert model["fixed_C"] == 10.0
    with pytest.raises(ValueError, match="development"):
        m.fit_fixed_calibrator(mock_attempts(2025))
    with pytest.raises(ValueError, match="only"):
        m.fit_fixed_calibrator(
            mock_attempts(2023).assign(action="NORMAL_GO_PASS")
        )


def test_monotonic_transform_preserves_original_fg_make_ranking():
    params = m.fit_fixed_calibrator(mock_attempts(2023))
    p = np.linspace(.001, .999, 500)
    candidate = m.apply_fixed_calibrator(p, params)
    assert np.isfinite(candidate).all()
    assert candidate.min() > 0
    assert candidate.max() < 1
    assert np.diff(candidate).min() >= 0
    with pytest.raises(ValueError, match="one-dimensional"):
        m.logit_scores(np.ones((2, 2)))


def test_paired_game_bootstrap_is_reproducible():
    orig = np.array([.20, .12, .08, .05])
    newer = np.array([.18, .09, .11, .02])
    ids = np.array(["a", "a", "b", "c"])
    first = m.paired_game_bootstrap(orig, newer, ids, repetitions=300)
    second = m.paired_game_bootstrap(orig, newer, ids, repetitions=300)
    assert first == second
    assert first[0] <= first[1]
    with pytest.raises(ValueError, match="aligned"):
        m.paired_game_bootstrap(orig, newer, ids[:-1], repetitions=300)


def test_group_report_uses_actual_attempts_and_displays_short_fg():
    p = mock_attempts(2024)
    params = m.fit_fixed_calibrator(mock_attempts(2023))
    groups = {row["name"]: row for row in m.prespecified_groups(p, params)}
    assert groups["ALL"]["n"] == len(p)
    assert groups["18-29 YARDS"]["n"] > 0
    assert groups["50+ YARDS"]["n"] > 0
    assert groups["Q4 LAST FIVE 18-29 YARDS"]["n"] > 0
    assert np.isfinite(groups["ALL"]["brier_improvement"])


def test_development_only_no_training_artifact_or_post_2025_download():
    source = SCRIPT.read_text()
    assert "((2023,), 2024)" in source
    assert "((2023, 2024), 2025)" in source
    assert "fit_fixed_calibrator(fit)" in source
    assert "load_pbp([2026])" not in source
    assert "field_goal_outcome_model.joblib" in source
    assert "joblib.dump(" not in source
