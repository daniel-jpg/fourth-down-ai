"""Rule-only Q4 terminal-value safeguard; prospective frozen source unchanged."""
import numpy as np
import pytest

from src.q4_demo_terminal_guard import use_q4_demo_terminal_guard
from src.kickoff_q4_native_features_candidate import (
    native_q4_normal_kickoff,
    use_q4_native_kickoff_candidate,
)
from src.streamlit_kickoff_preview import (
    isolated_engine_for_preview,
    isolated_engine_from_loaded_engine,
)


@pytest.fixture(scope="module")
def shared_engine():
    return isolated_engine_for_preview()


def state(qtr=4, clock=45., margin=-4., site="AWAY"):
    return {
        "qtr":qtr, "game_seconds_remaining":float(clock),
        "yardline_100":1., "ydstogo":1.,
        "score_differential":float(margin),
        "posteam_timeouts_remaining":2.,
        "defteam_timeouts_remaining":2.,
        "site":site,"roof":"outdoors",
    }


@pytest.mark.parametrize("margin,site,expected", [
    (-4., "AWAY", 0.), (-4., "HOME", 0.),
    (-2., "AWAY", 1.), (-2., "HOME", 1.),
    (-1., "NEUTRAL", 1.),
])
def test_unambiguous_zero_clock_is_exact_after_made_fg(shared_engine, margin, site, expected):
    private = isolated_engine_from_loaded_engine(shared_engine)
    base = private["normalize_state"](state(margin=margin, site=site))
    terminal = private["make_wp_state"](
        base, possession_original=False,
        yardline_100=np.nan, down=np.nan, ydstogo=np.nan,
        elapsed_seconds=60., score_change_original=3., is_kickoff=True,
    )
    assert terminal["game_seconds_remaining"] == 0.
    original = private["evaluate_post_play_states"]
    with use_q4_demo_terminal_guard(private):
        with use_q4_native_kickoff_candidate(private):
            values = private["evaluate_post_play_states"](base, [terminal])
            assert float(values[0]) == expected
    assert private["evaluate_post_play_states"] is original
    assert "_q4_demo_unchanged_evaluator" not in private
    assert shared_engine["evaluate_post_play_states"] is not private["evaluate_post_play_states"]


def test_nonterminal_wp_is_identical_to_native_candidate(shared_engine):
    private = isolated_engine_from_loaded_engine(shared_engine)
    base = private["normalize_state"](state())
    pending = private["make_wp_state"](
        base, possession_original=False,
        yardline_100=np.nan, down=np.nan, ydstogo=np.nan,
        elapsed_seconds=5., score_change_original=3., is_kickoff=True,
    )
    normal = native_q4_normal_kickoff(pending)
    unchanged = float(private["evaluate_post_play_states"](base, [normal])[0])
    with use_q4_demo_terminal_guard(private):
        with use_q4_native_kickoff_candidate(private):
            updated = float(private["evaluate_post_play_states"](base, [pending])[0])
    assert updated == pytest.approx(unchanged, abs=1e-12)


def test_q3_is_unaffected_even_under_guard(shared_engine):
    private = isolated_engine_from_loaded_engine(shared_engine)
    base = private["normalize_state"](state(qtr=3, clock=1100.))
    ended = private["make_wp_state"](
        base, possession_original=False,
        yardline_100=np.nan, down=np.nan, ydstogo=np.nan,
        elapsed_seconds=0., score_change_original=3., is_kickoff=True,
    )
    original = float(private["evaluate_post_play_states"](base, [ended])[0])
    with use_q4_demo_terminal_guard(private):
        test_value = float(private["evaluate_post_play_states"](base, [ended])[0])
    assert test_value == original


def test_guard_rejects_nesting_and_restores_on_failure(shared_engine):
    private = isolated_engine_from_loaded_engine(shared_engine)
    previous = private["evaluate_post_play_states"]
    with pytest.raises(RuntimeError, match="deliberate"):
        with use_q4_demo_terminal_guard(private):
            with pytest.raises(RuntimeError, match="Nested"):
                with use_q4_demo_terminal_guard(private):
                    pass
            raise RuntimeError("deliberate")
    assert private["evaluate_post_play_states"] is previous
    assert "_q4_demo_unchanged_evaluator" not in private


def test_public_app_uses_guard_not_registered_prospective_module():
    from pathlib import Path
    app = Path("app.py").read_text()
    assert "with use_q4_demo_terminal_guard(q4_private_engine):" in app
    assert "preview_native_kickoff_feature_candidate(" in app
    native = Path("src/kickoff_q4_native_features_candidate.py").read_text()
    assert "use_q4_demo_terminal_guard" not in native
    frozen = Path("src/83_forward_2026_native_challenger_once.py").read_text()
    assert "use_q4_demo_terminal_guard" not in frozen



def test_real_q4_demo_known_decision_anchor_and_late_case(shared_engine):
    """Re-run two real 1,200-draw Q4 results through the exact demo wrapper."""
    from src.streamlit_kickoff_preview import preview_native_kickoff_feature_candidate

    situations = [
        # The original 4:59 Q4 demonstration should not drift.
        ({"qtr":4, "game_seconds_remaining":299.,
          "yardline_100":6., "ydstogo":4.,
          "score_differential":-4.,
          "posteam_timeouts_remaining":3.,
          "defteam_timeouts_remaining":3.,
          "site":"AWAY", "roof":"outdoors"}, "FIELD GOAL"),
        # The exact Issue 12 scenario, with 0:45 and down four.
        (state(clock=45., margin=-4., site="AWAY"), "GO FOR IT"),
    ]
    for current, expected in situations:
        private = isolated_engine_from_loaded_engine(shared_engine)
        with use_q4_demo_terminal_guard(private):
            result = preview_native_kickoff_feature_candidate(
                current, simulations=1200, action_seed=43,
                isolated_engine=private,
            )
        rows = result["results"]
        assert rows.loc[rows["eligible"], "expected_win_probability"].between(0, 1).all()
        assert result["recommendation"] in (
            "FIELD GOAL", "GO — RUN", "GO — PASS/DROPBACK",
        )
        best_name = ("GO FOR IT" if result["recommendation"].startswith("GO") else result["recommendation"])
        assert best_name == expected
        if current["game_seconds_remaining"] == 299.:
            fg = rows.loc[rows["action"].eq("FIELD_GOAL"), "expected_win_probability"].iloc[0]
            assert round(100 * float(fg), 2) == 37.00
        else:
            fg = rows.loc[rows["action"].eq("FIELD_GOAL"), "expected_win_probability"].iloc[0]
            assert round(100 * float(fg), 2) == 23.72
