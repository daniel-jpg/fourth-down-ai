"""Deterministic accounting and pure-rule checks for issue #12 research audit."""
import pytest

from scripts.audit_issue12_q4_late_fg import (
    CLOCKS, MARGINS, SITES, SEEDS,
    group_fg_outcome, make_state, terminal_win_if_decided,
)


@pytest.mark.parametrize("clock", CLOCKS)
@pytest.mark.parametrize("margin", MARGINS)
@pytest.mark.parametrize("site", SITES)
def test_supported_synthetic_grid(clock, margin, site):
    state = make_state(clock, margin, site)
    assert state["qtr"] == 4
    assert state["ydstogo"] == state["yardline_100"] == 1.
    assert state["game_seconds_remaining"] == float(clock)
    assert state["score_differential"] == float(margin)
    assert len(SEEDS) == 2


@pytest.mark.parametrize("clock,margin,site", [
    (0, -4, "AWAY"), (901, -4, "HOME"), (45, -3, "AWAY"),
    (45, -4, "UNKNOWN"),
])
def test_non_scoped_grid_rejected(clock, margin, site):
    with pytest.raises(ValueError):
        make_state(clock, margin, site)


@pytest.mark.parametrize("margin,expected", [
    (-4., 0.), (-2., 1.), (-1., 1.), (-3., None),
])
def test_regulation_zero_clock_final_score_is_rule_only(margin, expected):
    assert terminal_win_if_decided(margin) == expected


@pytest.mark.parametrize("data,expected", [
    ({"_score_change_original":3, "is_kickoff":1,
      "_possession_original":False}, "made"),
    ({"_score_change_original":-6, "is_kickoff":1},
      "opponent_scored_on_kick"),
    ({"_score_change_original":0, "_possession_original":True},
      "kicking_team_recovered"),
    ({"_score_change_original":0, "_possession_original":False},
      "missed_or_other_failure"),
])
def test_branch_labels_are_exclusive(data, expected):
    assert group_fg_outcome(data) == expected
