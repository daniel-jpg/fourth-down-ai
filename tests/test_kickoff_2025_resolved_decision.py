"""Bounds for rare kickoff scoring events are not policy probabilities."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT=Path(__file__).resolve().parents[1]
SCRIPT=ROOT/"src/77_compare_penalty_resolved_wp.py"
spec=importlib.util.spec_from_file_location("step77_kickoff",SCRIPT)
audit=importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


@pytest.mark.parametrize("assumed,expected",[
    (0.,0.),(.001,.001),(.0025,.0025),(.005,.005),(1.,1.),
])
def test_single_action_missing_return_td_effect_is_bounded(
    assumed,expected,
):
    assert audit.max_single_action_win_probability_shift(
        assumed
    )==expected


@pytest.mark.parametrize("bad",[-.1,1.1,np.nan,np.inf,-np.inf])
def test_td_stress_rejects_invalid_probabilities(bad):
    with pytest.raises(ValueError,match="in \\[0,1\\]"):
        audit.max_single_action_win_probability_shift(bad)


def test_one_action_probability_bound_is_a_valid_worst_case():
    baseline=.3411
    p=.005
    for outcome in [0.,.2,.5,.9,1.]:
        blended=(1-p)*baseline+p*outcome
        assert abs(blended-baseline)<=audit.max_single_action_win_probability_shift(p)+1e-12


def test_comparison_keeps_monte_carlo_near_ties_visible():
    rows=pd.DataFrame([
        {
            "action":"FIELD_GOAL","eligible":True,
            "expected_win_probability":.3411,"mc_se":.004,
        },
        {
            "action":"NORMAL_GO_PASS","eligible":True,
            "expected_win_probability":.3449,"mc_se":.004,
        },
    ])
    item=audit.paired_action_snapshot("test",{"results":rows})
    assert item["scenario"]=="test"
    assert item["near_tie_mc_only"]
    assert item["go_minus_fg_pp"]==pytest.approx(.38)


def test_research_uses_previously_examined_2025_only():
    s=SCRIPT.read_text()
    assert "models_frozen" in s
    assert "2026_outcomes_used" in s
    assert "preview_observed_2025_resolved_recommendation(" in s
    assert "include_verified_penalties=True" in s
    assert "nfl.load_pbp([2026])" not in s
    assert "joblib.dump(" not in s
