"""Issue 9: warn about sparse GO support, without modifying eligibility."""
import pytest

from src.situational_support import deep_own_short_go_notice


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
