"""End-to-end 2025 ordinary + optional onside preview contract."""

import pandas as pd
import pytest

from src.streamlit_kickoff_preview import (
    preview_observed_2025_with_onside,
    preview_observed_2025_recommendation,
    observed_2025_onside_summary,
)


RAMS = {
    "qtr":4, "game_seconds_remaining":299.,
    "yardline_100":6., "ydstogo":4.,
    "score_differential":-4.,
    "posteam_timeouts_remaining":3.,
    "defteam_timeouts_remaining":3.,
    "site":"AWAY", "roof":"outdoors",
}


def eligible(result):
    return result["results"].loc[
        result["results"]["eligible"],
        ["action", "expected_win_probability", "mc_se"],
    ].sort_values("action").reset_index(drop=True)


def test_zero_onside_attempts_is_identical_to_frozen_2025_experiment():
    original=preview_observed_2025_recommendation(
        RAMS, simulations=240, action_seed=42
    )
    adjusted=preview_observed_2025_with_onside(
        RAMS, attempt_probability=0., recovery_probability=.10,
        simulations=240, action_seed=42,
    )
    pd.testing.assert_frame_equal(eligible(original),eligible(adjusted),
                                  check_exact=True)


def test_user_specified_onside_recovery_changes_only_made_fg_branch():
    recovered=preview_observed_2025_with_onside(
        RAMS, attempt_probability=1., recovery_probability=1.,
        simulations=240, action_seed=42,
    )
    failed=preview_observed_2025_with_onside(
        RAMS, attempt_probability=1., recovery_probability=0.,
        simulations=240, action_seed=42,
    )
    a=eligible(recovered).set_index("action")["expected_win_probability"]
    b=eligible(failed).set_index("action")["expected_win_probability"]
    assert a["FIELD_GOAL"] != b["FIELD_GOAL"]
    # GO touchdown/failed conversion use the unchanged ordinary kickoff pool.
    assert a["NORMAL_GO_PASS"] == b["NORMAL_GO_PASS"]


def test_small_sample_is_explicit_in_the_preview_api():
    data=observed_2025_onside_summary()
    assert data=={
        "attempts":41,
        "recovered":4,
        "matched":37,
        "success_donor_count":4,
        "failed_donor_count":33,
    }


def test_onside_sensitive_extension_rejects_nonapplicable_games():
    with pytest.raises(ValueError, match="regulation"):
        preview_observed_2025_with_onside(
            {**RAMS,"qtr":5},
            attempt_probability=.25,recovery_probability=.10,
        )
    with pytest.raises(ValueError,match="leave.*trailing"):
        preview_observed_2025_with_onside(
            {**RAMS,"score_differential":-3.},
            attempt_probability=.25,recovery_probability=.10,
        )
    with pytest.raises(ValueError,match="200"):
        preview_observed_2025_with_onside(
            RAMS,attempt_probability=.25,recovery_probability=.10,
            simulations=199,
        )
