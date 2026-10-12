"""Opt-in native kickoff-input comparison cannot change official model state."""

import ast
from pathlib import Path

import pandas as pd
import pytest

from src.streamlit_kickoff_preview import (
    isolated_engine_for_preview,
    isolated_engine_from_loaded_engine,
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


def test_lightweight_engine_namespace_reuses_heavy_artifacts_without_mutating_live():
    original = isolated_engine_for_preview()
    original_eval = original["evaluate_post_play_states"]
    original_recommend = original["recommend"]
    private = isolated_engine_from_loaded_engine(original)
    assert private is not original
    assert private["recommend"] is not original_recommend
    assert private["recommend"].__globals__ is private
    assert private["evaluate_post_play_states"].__globals__ is private
    assert private["wp_model"] is original["wp_model"]
    assert private["go_model"] is original["go_model"]
    assert private["go_pool"] is original["go_pool"]
    result = preview_native_kickoff_feature_candidate(
        HOME, simulations=240, action_seed=42, isolated_engine=private,
    )
    assert len(result["results"]) > 0
    assert private["evaluate_post_play_states"].__globals__ is private
    assert original["evaluate_post_play_states"] is original_eval
    assert original["recommend"] is original_recommend
    assert private["evaluate_post_play_states"] is not original_eval


def test_deployed_streamlit_uses_native_q4_inputs_without_extra_public_controls():
    app=(ROOT/"app.py").read_text()
    ast.parse(app)
    # Q4 research-demo estimates now use the candidate in a private model,
    # never as an opt-in second panel or a frozen-stable model refit.
    assert "preview_native_kickoff_feature_candidate(" in app
    assert "isolated_engine_from_loaded_engine(engine)" in app
    assert "isolated_engine=q4_private_engine" in app
    assert "simulations=ADAPTIVE_HIGH_SIMULATIONS" in app
    assert "experimental_native_candidate_enabled" not in app
    assert "Additional model-input comparison (research-only)" not in app
    assert "2025 kickoff comparison" not in app
    assert "INCONCLUSIVE" not in app
    assert "Close call" in app
    # Automatic post-FG sensitivity diagnostic is still preserved.
    assert "inspect_made_fg_kickoff_encoding(" in app
    assert "fg_input_sensitive" in app
    assert "st.warning(fg_encoding_warning)" not in app
