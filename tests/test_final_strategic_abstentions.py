"""Keep unsupported coaching choices transparent, never substitute action odds."""
import pytest

from src.situational_support import (
    deep_own_long_go_notice, late_q4_trailing_punt_notice,
    supported_recommendation_display,
)


def state(*, qtr=4, secs=18, margin=-4, yardline=60, togo=10, timeouts=0):
    return dict(
        qtr=qtr, game_seconds_remaining=float(secs),
        score_differential=float(margin),
        yardline_100=float(yardline), ydstogo=float(togo),
        posteam_timeouts_remaining=float(timeouts),
    )


def test_final_18s_down_four_own_40_punt_abstains():
    st = state()
    txt = late_q4_trailing_punt_notice(st, "PUNT")
    assert txt and "no offensive timeouts" in txt
    assert "not a supported PUNT recommendation" in txt
    assert supported_recommendation_display(st, "PUNT") == "INSUFFICIENT DATA"
    # The guard must not select GO automatically instead of PUNT.
    assert supported_recommendation_display(st, "GO FOR IT") == "GO FOR IT"


@pytest.mark.parametrize("edit", [
    {"qtr":3}, {"secs":61}, {"secs":0}, {"margin":0},
    {"yardline":49}, {"timeouts":1},
])
def test_last_minute_punt_notice_narrowly_scoped(edit):
    st = state(**edit)
    assert late_q4_trailing_punt_notice(st, "PUNT") is None
    assert supported_recommendation_display(st, "PUNT") == "PUNT"


@pytest.mark.parametrize("qtr", [1, 2, 3, 4])
def test_fourth_and_eleven_at_own_one_go_abstains(qtr):
    st = state(qtr=qtr, yardline=99, togo=11)
    txt = deep_own_long_go_notice(st, "GO FOR IT")
    assert txt and "0.72" in txt and "unusually" in txt
    assert supported_recommendation_display(st, "GO FOR IT") == "INSUFFICIENT DATA"


@pytest.mark.parametrize("edit", [
    {"yardline":94}, {"togo":6}, {"qtr":5},
])
def test_long_goal_line_qualifier_not_generalized(edit):
    st = state(**edit)
    assert deep_own_long_go_notice(st, "GO FOR IT") is None
    assert supported_recommendation_display(st, "GO FOR IT") == "GO FOR IT"


def test_other_strategies_not_requalified():
    st = state(yardline=99, togo=11)
    assert deep_own_long_go_notice(st, "PUNT") is None
    assert late_q4_trailing_punt_notice(st, "GO FOR IT") is None


@pytest.mark.parametrize("invalid", [
    {}, {"qtr":4, "game_seconds_remaining":float("nan")},
    {"qtr":"wrong"}, {"qtr":4, "game_seconds_remaining":"NaN"},
])
def test_bad_context_fails_closed(invalid):
    assert deep_own_long_go_notice(invalid, "GO FOR IT") is None
    assert late_q4_trailing_punt_notice(invalid, "PUNT") is None
