"""Smoke tests for the read-only Step 63 evaluation driver."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


path = Path(__file__).resolve().parents[1] / "src/63_compare_kickoff_encoding.py"
spec = importlib.util.spec_from_file_location("read_only_kickoff_compare", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_step63_import_and_game_cluster_bootstrap_repeatable():
    improvement = np.array([0.1, -0.05, 0.2, 0.4, -0.1])
    games = np.array(["g1", "g1", "g2", "g3", "g3"])
    a = module.paired_game_bootstrap(improvement, games, repeats=50, seed=15)
    b = module.paired_game_bootstrap(improvement, games, repeats=50, seed=15)
    assert a == b
    assert np.isfinite(a).all()
    assert a[0] <= a[1]


def test_step63_action_values_preserve_ineligible_run():
    data = pd.DataFrame([
        {"action": "FIELD_GOAL", "eligible": True, "win_probability_pct": 30.4},
        {"action": "NORMAL_GO_PASS", "eligible": True, "win_probability_pct": 31.2},
        {"action": "NORMAL_GO_RUN", "eligible": False, "win_probability_pct": np.nan},
    ])
    got = module.values({"results": data})
    assert got["FIELD_GOAL"] == pytest.approx(30.4)
    assert got["NORMAL_GO_PASS"] == pytest.approx(31.2)
    assert got["NORMAL_GO_RUN"] is None
