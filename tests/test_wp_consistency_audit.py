import numpy as np
import pandas as pd
import pytest

from src.wp_consistency_audit import (
    validate_2024_q4_donors, weighted_continuation,
    field_position_reversals, evaluate_made_fg_case,
)


def pool():
    return pd.DataFrame({
        "season": [2024] * 4,
        "quarter": [4] * 4,
        "transition_type": ["touchback", "touchback", "return", "return"],
        "own_yardline": [20, 30, 25, 30],
        "elapsed_seconds": [0, 0, 5, 7],
    })


def test_only_2024_q4_donors_allowed():
    d = validate_2024_q4_donors(pool())
    assert d.touchback_count == 2
    assert d.return_count == 2
    assert d.touchback_own20_probability == pytest.approx(0.5)
    with pytest.raises(ValueError, match="2024"):
        validate_2024_q4_donors(pool().assign(season=2025))


def test_exact_mixture_weighting():
    # 50% touches: (own20 WP .2 + own35 WP .4)/2 = .3
    # 50% returns: mean(.6,.8) = .7 -> final .5 exactly.
    x = weighted_continuation(.5, .2, .4, [.6, .8],
                              touchback_own20_probability=.5)
    assert x == pytest.approx(.5)
    with pytest.raises(ValueError, match="Touchback rate"):
        weighted_continuation(1.5, .2, .4, [.6, .8],
                              touchback_own20_probability=.5)


def test_field_position_reversal_detection():
    bad = field_position_reversals([2, 4, 6, 10], [.1, .11, .1, .11], minimum_pp=.5)
    assert len(bad) == 2
    assert bad[0]["from_own"] == 2
    assert bad[0]["kicker_wp_increase_pp"] == pytest.approx(1)
    assert field_position_reversals([2, 4, 6], [.4, .35, .3]) == []


def test_examine_made_fg_with_one_batch_and_consistent_clock():
    def make_wp_state(*, base, possession_original, yardline_100, down,
                      ydstogo, elapsed_seconds, score_change_original, is_kickoff):
        assert possession_original is False
        assert score_change_original == 3
        return {"own": 100 - yardline_100, "time_cost": elapsed_seconds,
                "is_kickoff": is_kickoff, "score_diff": base["score_differential"] + 3}

    calls = []
    def predict(states, original_home, neutral):
        assert original_home is False
        assert neutral is False
        calls.append(states)
        return np.array([.45 if s["is_kickoff"] else
                         .5 - .002 * s["own"] - .001 * s["time_cost"]
                         for s in states])

    case = {
        "qtr": 4, "game_seconds_remaining": 299, "score_differential": -4,
        "_wp_original_on_home_axis": 0, "site": "AWAY",
    }
    d = validate_2024_q4_donors(pool())
    result = evaluate_made_fg_case({"make_wp_state": make_wp_state,
                                    "predict_original_team_wp": predict},
                                   case, d, touchback_rates=(.5,))
    assert len(calls) == 1
    assert calls[0][0]["time_cost"] == 5
    assert calls[0][-2]["time_cost"] == 10
    assert calls[0][-1]["time_cost"] == 12
    assert result["post_fg_score_diff"] == -1
    assert result["kickoff_pending_wp"] == pytest.approx(.45)
    assert result["mixed_wp"][.5] < .45
    assert result["reversals"] == []


def test_reject_non_q4_and_short_clocks():
    case = {"qtr": 3, "game_seconds_remaining": 299,
            "_wp_original_on_home_axis": 0, "site": "AWAY"}
    with pytest.raises(ValueError, match="Q4"):
        evaluate_made_fg_case({}, case, validate_2024_q4_donors(pool()))
