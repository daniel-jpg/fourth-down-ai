"""Opt-in native kickoff-input comparison cannot change official model state."""

import ast
from pathlib import Path

import pandas as pd
import pytest

from src.streamlit_kickoff_preview import (
    isolated_engine_for_preview,
    preview_native_kickoff_feature_candidate,
    preview_observed_2025_resolved_recommendation,
    preview_strategy_rank_with_uncertainty,
    recommendation_sensitivity,
)

ROOT=Path(__file__).resolve().parents[1]
HOME={
    "qtr":4, "game_seconds_remaining":458.,
    "yardline_100":30., "ydstogo":3.,
    "score_differential":-3.,
    "posteam_timeouts_remaining":3.,
    "defteam_timeouts_remaining":3.,
    "site":"HOME","roof":"outdoors",
}


def normal(result):
    return (
        result["results"].loc[
            result["results"]["eligible"],
            ["action","expected_win_probability","mc_se"],
        ]
        .sort_values("action")
        .reset_index(drop=True)
    )


def test_native_preview_ranks_home_site_independently_from_original():
    official=isolated_engine_for_preview()["recommend"](
        HOME,n_simulations=240,seed=42
    )
    candidate=preview_native_kickoff_feature_candidate(
        HOME,simulations=240,action_seed=42
    )
    original_leader=preview_strategy_rank_with_uncertainty(official)
    candidate_leader=preview_strategy_rank_with_uncertainty(candidate)
    assert original_leader["leader"]=="FIELD GOAL"
    assert candidate_leader["leader"]=="GO FOR IT"
    sens=recommendation_sensitivity(official,candidate)
    assert sens["model_sensitive"] and sens["reversed"]


def test_native_preview_never_changes_538_donor_legacy_comparison():
    before=preview_observed_2025_resolved_recommendation(
        HOME,simulations=220,action_seed=42
    )
    candidate=preview_native_kickoff_feature_candidate(
        HOME,simulations=220,action_seed=42
    )
    after=preview_observed_2025_resolved_recommendation(
        HOME,simulations=220,action_seed=42
    )
    pd.testing.assert_frame_equal(normal(before),normal(after),check_exact=True)
    assert not normal(candidate).equals(normal(before))


def test_research_candidate_requires_q4_and_real_simulation_count():
    with pytest.raises(ValueError,match="Q4 regulation"):
        preview_native_kickoff_feature_candidate({**HOME,"qtr":3})
    with pytest.raises(ValueError,match="200"):
        preview_native_kickoff_feature_candidate(HOME,simulations=199)


def test_deployed_streamlit_wires_opt_in_only_and_no_official_reassignment():
    app=(ROOT/"app.py").read_text()
    ast.parse(app)
    assert "experimental_native_candidate_enabled = False" in app
    assert "preview_native_kickoff_feature_candidate(" in app
    assert "Additional model-input comparison (research-only)" in app
    assert "The official selection above" in app
    assert "away kicking teams" in app
    assert "candidate_result = preview_native_kickoff_feature_candidate(" in app
