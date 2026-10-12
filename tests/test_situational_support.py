"""Issue 9: warn about sparse GO support, without modifying eligibility."""
import pytest

from src.situational_support import (
    deep_own_short_go_notice, supported_recommendation_display,
    late_q4_goal_line_site_notice,
)


@pytest.mark.parametrize("quarter,yardline,togo", [
    (1, 95, 3), (2, 90, 2), (3, 99, 3),
    (4, 97, 2.5),
])
def test_deep_own_short_go_warning_is_narrow_and_truthful(
    quarter, yardline, togo,
):
    state = {"qtr": quarter, "yardline_100": yardline, "ydstogo": togo}
    text = deep_own_short_go_notice(state, "GO FOR IT")
    assert "Limited historical GO support" in text
    assert "only one normal GO attempt" in text
    assert "none inside the own 5" in text
    assert "not a dependable coaching" in text


@pytest.mark.parametrize("quarter,yardline,togo,strategy", [
    (4, 6, 4, "GO FOR IT"),
    (4, 30, 3, "GO FOR IT"),
    (1, 80, 3, "GO FOR IT"),
    (1, 95, 1, "GO FOR IT"),
    (1, 95, 4, "GO FOR IT"),
    (5, 95, 3, "GO FOR IT"),
    (1, 95, 3, "PUNT"),
    (1, 95, 3, "FIELD GOAL"),
])
def test_support_notice_does_not_leak_to_supported_or_other_actions(
    quarter, yardline, togo, strategy,
):
    state = {"qtr": quarter, "yardline_100": yardline, "ydstogo": togo}
    assert deep_own_short_go_notice(state, strategy) is None


def test_nonfinite_fields_skip_notice():
    assert deep_own_short_go_notice({
        "qtr": 1, "yardline_100": float("nan"), "ydstogo": 3,
    }, "GO FOR IT") is None
    assert deep_own_short_go_notice({
        "qtr": 1, "yardline_100": 95, "ydstogo": float("inf"),
    }, "GO FOR IT") is None


@pytest.mark.parametrize("quarter,field,togo,selected,display", [
    (1, 95, 3, "GO FOR IT", "INSUFFICIENT DATA"),
    (2, 90, 2, "GO FOR IT", "INSUFFICIENT DATA"),
    (4, 6, 4, "GO FOR IT", "GO FOR IT"),
    (1, 95, 3, "PUNT", "PUNT"),
    (5, 95, 3, "GO FOR IT", "GO FOR IT"),
])
def test_supported_recommendation_only_abstains_in_extreme_sparse_go_region(
    quarter, field, togo, selected, display,
):
    state = {"qtr": quarter, "yardline_100": field, "ydstogo": togo}
    assert supported_recommendation_display(state, selected) == display


@pytest.mark.parametrize("clock,opp_yard,offense_margin,strategies,expected", [
    (299, 6, -4, {"GO FOR IT", "FIELD GOAL"}, True),
    (60, 10, -8, {"GO FOR IT", "FIELD GOAL"}, True),
    (301, 6, -4, {"GO FOR IT", "FIELD GOAL"}, False),
    (299, 11, -4, {"GO FOR IT", "FIELD GOAL"}, False),
    (299, 6, 4, {"GO FOR IT", "FIELD GOAL"}, False),
    (299, 6, -9, {"GO FOR IT", "FIELD GOAL"}, False),
    (299, 6, -4, {"GO FOR IT", "PUNT"}, False),
])
def test_late_q4_goal_line_site_notice_is_strictly_scoped(
    clock, opp_yard, offense_margin, strategies, expected,
):
    state = {"qtr": 4, "game_seconds_remaining": clock,
             "yardline_100": opp_yard, "score_differential": offense_margin}
    actual = late_q4_goal_line_site_notice(state, strategies)
    assert (actual is not None) == expected
    if expected:
        assert "13 games" in actual
        assert "not all following GO failures" in actual


def test_site_notice_never_triggers_for_overtime_or_invalid_states():
    state = {"qtr": 5, "game_seconds_remaining": 90,
             "yardline_100": 6, "score_differential": -4}
    assert late_q4_goal_line_site_notice(
        state, {"GO FOR IT", "FIELD GOAL"}
    ) is None
    state["qtr"] = 4
    state["score_differential"] = float("nan")
    assert late_q4_goal_line_site_notice(
        state, {"GO FOR IT", "FIELD GOAL"}
    ) is None
