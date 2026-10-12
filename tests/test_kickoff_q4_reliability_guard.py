"""Q4 kickoff-input warning is deterministic, selective and non-mutating."""

import ast
from pathlib import Path

import numpy as np
import pytest

from src.kickoff_q4_reliability_guard import (
    DISPLAY_ABS_DIFFERENCE_PP,
    inspect_made_fg_kickoff_encoding,
    reliability_warning_markdown,
)
from src.streamlit_kickoff_preview import isolated_engine_for_preview

ROOT = Path(__file__).resolve().parents[1]

BASE = {
    "qtr": 4,
    "game_seconds_remaining": 458.0,
    "yardline_100": 30.0,
    "ydstogo": 3.0,
    "score_differential": -3.0,
    "posteam_timeouts_remaining": 3.0,
    "defteam_timeouts_remaining": 3.0,
    "site": "AWAY",
    "roof": "outdoors",
}


def test_known_home_site_anomaly_triggers_automatic_conditional_warning():
    engine = isolated_engine_for_preview()
    orig = engine["evaluate_post_play_states"]
    home = inspect_made_fg_kickoff_encoding(
        {**BASE,"site":"HOME"}, isolated_engine=engine
    )
    assert home["applicable"] and home["material"]
    assert home["absolute_difference_pp"] > 20
    assert home["legacy_conditional_made_fg_wp"] == pytest.approx(.7598,abs=.001)
    assert home["native_conditional_made_fg_wp"] == pytest.approx(.5034,abs=.001)
    assert home["post_fg_seconds_remaining"] == 453.0
    assert home["assumed_fg_execution_seconds"] == 5.0
    assert home["original_offense_site"] == "HOME"
    assert home["models_equal_weights"]
    assert home["is_action_level_go_vs_fg_comparison"] is False
    assert home["2026_data_read"] is False
    assert engine["evaluate_post_play_states"] is orig
    message = reliability_warning_markdown(home)
    assert "conditional post-FG" in message
    assert "NOT the" in message
    assert "frozen-model weights are unchanged" in message
    assert "separate experimental panel" not in message


def test_small_away_site_delta_does_not_spam_user():
    away = inspect_made_fg_kickoff_encoding(BASE)
    assert away["applicable"]
    assert not away["material"]
    assert away["absolute_difference_pp"] < DISPLAY_ABS_DIFFERENCE_PP
    assert reliability_warning_markdown(away) is None


def test_rams_style_q4_four_minutes_fifty_nine_seconds_detects_issue():
    rams = {
        **BASE,
        "game_seconds_remaining": 299.,
        "yardline_100": 6.,
        "ydstogo": 4.,
        "score_differential": -4.,
        "site": "AWAY",
    }
    probe = inspect_made_fg_kickoff_encoding(rams)
    assert probe["applicable"] and probe["material"]
    assert probe["absolute_difference_pp"] > 4.9
    assert probe["legacy_conditional_made_fg_wp"] > (
        probe["native_conditional_made_fg_wp"]
    )


@pytest.mark.parametrize("overrides",[
    {"qtr":3},
    {"qtr":5},
    {"game_seconds_remaining": 40.},
    {"game_seconds_remaining": 45.},
    {"game_seconds_remaining": 0.},
    {"game_seconds_remaining": 901.},
])
def test_research_warning_skips_unsupported_periods_and_clock(overrides):
    # An empty engine must never be touched for excluded situations.
    diag=inspect_made_fg_kickoff_encoding(
        {**BASE,**overrides},isolated_engine={}
    )
    assert not diag["applicable"]
    assert reliability_warning_markdown(diag) is None


@pytest.mark.parametrize("threshold",[-2.,0.,np.nan,np.inf,100.01])
def test_materiality_input_must_be_finite_and_positive(threshold):
    with pytest.raises(ValueError,match="threshold"):
        inspect_made_fg_kickoff_encoding(
            BASE,material_threshold_pp=threshold,
        )


def test_threshold_is_only_a_display_heuristic_not_a_confidence_bound():
    base=inspect_made_fg_kickoff_encoding(BASE)
    lower=inspect_made_fg_kickoff_encoding(
        BASE,material_threshold_pp=.5,
    )
    assert base["absolute_difference_pp"] == pytest.approx(
        lower["absolute_difference_pp"],abs=1e-12
    )
    assert lower["material"]
    assert base["material"] is False
    assert lower["is_action_level_go_vs_fg_comparison"] is False


def test_production_recommendation_before_and_after_reliability_probe_identical():
    private = isolated_engine_for_preview()
    original = private["evaluate_post_play_states"]
    prior=private["recommend"](BASE,n_simulations=220,seed=42)
    probe=inspect_made_fg_kickoff_encoding(BASE,isolated_engine=private)
    subsequent=private["recommend"](BASE,n_simulations=220,seed=42)
    assert probe["applicable"]
    assert original.__globals__["evaluate_post_play_states"] is original
    assert prior["pre_decision_wp"] == subsequent["pre_decision_wp"]
    import pandas as pd
    pd.testing.assert_frame_equal(
        prior["results"],subsequent["results"],check_exact=True
    )


def test_exact_q4_459_native_candidate_recomputes_real_action_probabilities():
    # Same budget and random seed isolate representation changes from
    # Monte Carlo sampling noise. No model refit or future-season data.
    from src.streamlit_kickoff_preview import (
        preview_native_kickoff_feature_candidate,
    )
    state = {
        **BASE, "game_seconds_remaining": 299.,
        "yardline_100": 6., "ydstogo": 4.,
        "score_differential": -4., "site": "AWAY",
    }
    private = isolated_engine_for_preview()
    baseline = private["recommend"](state, n_simulations=1200, seed=43)
    corrected = preview_native_kickoff_feature_candidate(
        state, simulations=1200, action_seed=43,
    )
    old = baseline["results"].set_index("action")
    new = corrected["results"].set_index("action")
    assert old.index.tolist() == new.index.tolist()
    assert bool(old.loc["FIELD_GOAL", "eligible"])
    assert bool(new.loc["FIELD_GOAL", "eligible"])
    old_fg = float(old.loc["FIELD_GOAL", "expected_win_probability"])
    corrected_fg = float(new.loc["FIELD_GOAL", "expected_win_probability"])
    assert 0 <= corrected_fg <= 1
    assert abs(old_fg - corrected_fg) > .001, (
        "The corrected Q4 model must produce a different action-level "
        "field-goal win probability, not a cosmetic change."
    )
    assert baseline["pre_decision_wp"] == corrected["pre_decision_wp"]


def test_streamlit_guard_is_automatic_but_requires_eligible_fg_and_q4():
    app=(ROOT/"app.py").read_text()
    ast.parse(app)
    assert "fg_eligible_rows = result[\"results\"].loc[" in app
    assert 'result["results"]["action"].eq("FIELD_GOAL")' in app
    assert 'result["results"]["eligible"].eq(True)' in app
    assert "inspect_made_fg_kickoff_encoding(" in app
    assert "fg_input_sensitive = bool(" in app
    assert "Close call" in app
    assert "preview_native_kickoff_feature_candidate(" in app
    assert "if len(fg_eligible_rows) == 1:" in app
    before=app.index("fg_encoding_probe = inspect_made_fg_kickoff_encoding(")
    # The automatic guard remains part of the official decision path,
    # preceding ordinary play-call advice; no optional kickoff UI remains.
    assert before < app.index("    # GO play call")
    assert "experimental_kickoff_enabled" not in app
    src=(ROOT/"src/kickoff_q4_reliability_guard.py").read_text()
    assert "2026" in src
    assert "load_pbp([2026])" not in src
    assert "joblib.dump(" not in src
