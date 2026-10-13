"""Issue #12: field-goal historical support is never mislabeled as proven."""
import pytest
from src.situational_support import (
    late_q4_trailing_after_fg_notice,
    supported_recommendation_display,
)


def state(*, quarter=4, seconds=45, margin=-4):
    return {
        "qtr": quarter, "game_seconds_remaining": float(seconds),
        "score_differential": float(margin),
        "yardline_100": 1., "ydstogo": 1.,
    }


@pytest.mark.parametrize("seconds", [1, 20, 40, 45, 60, 120])
def test_late_q4_fg_alternative_displays_evidence_warning(seconds):
    message = late_q4_trailing_after_fg_notice(
        state(seconds=seconds), {"GO FOR IT", "FIELD GOAL"},
    )
    assert message is not None
    assert "zero" in message.lower()
    assert "unvalidated model extrapolation" in message.lower()
    assert "clock has expired" in message
    # Preserve GO selection and all numeric probabilities; flag FG only.
    assert supported_recommendation_display(
        state(seconds=seconds), "GO FOR IT",
    ) == "GO FOR IT"
    assert supported_recommendation_display(
        state(seconds=seconds), "FIELD GOAL",
    ) == "INSUFFICIENT DATA"


@pytest.mark.parametrize("overrides", [
    {"quarter": 3}, {"quarter": 5}, {"seconds": 0},
    {"seconds": 121}, {"margin": -2}, {"margin": -1}, {"margin": -5},
])
def test_no_warning_outside_counted_region(overrides):
    case = state(**overrides)
    assert late_q4_trailing_after_fg_notice(case, {"FIELD GOAL"}) is None
    assert supported_recommendation_display(case, "FIELD GOAL") == "FIELD GOAL"


def test_no_warning_if_field_goal_is_ineligible():
    assert late_q4_trailing_after_fg_notice(
        state(), {"GO FOR IT", "PUNT"},
    ) is None


@pytest.mark.parametrize("bad", [
    {}, {"qtr": 4, "game_seconds_remaining": "bad",
          "score_differential": -4},
    {"qtr": 4, "game_seconds_remaining": float("nan"),
          "score_differential": -4},
])
def test_invalid_state_never_triggers_support_claim(bad):
    assert late_q4_trailing_after_fg_notice(
        bad, {"FIELD GOAL"},
    ) is None


def test_public_app_includes_warning_without_changing_model():
    from pathlib import Path
    app = Path("app.py").read_text()
    assert "fg_support_notice = late_q4_trailing_after_fg_notice(" in app
    assert "st.warning(fg_support_notice)" in app
    assert "preview_native_kickoff_feature_candidate(" in app
    assert "2026_forward_kickoff_wp_final" not in app
