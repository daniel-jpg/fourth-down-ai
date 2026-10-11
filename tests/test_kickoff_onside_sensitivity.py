"""Onside sensitivity must preserve possession, clocks, and official baseline."""

import numpy as np
import pandas as pd
import pytest

from src.kickoff_2025_empirical import load_frozen_pool
from src.kickoff_2025_onside_sensitivity import (
    load_onside_verified, resolve_q4_fg_onside_what_if,
    use_q4_onside_what_if, validate_what_if,
)
from src.streamlit_kickoff_preview import (
    isolated_engine_for_preview, preview_observed_2025_recommendation,
)


RAMS = {
    "qtr": 4, "game_seconds_remaining": 299.,
    "yardline_100": 6., "ydstogo": 4.,
    "score_differential": -4.,
    "posteam_timeouts_remaining": 3.,
    "defteam_timeouts_remaining": 3.,
    "site": "AWAY", "roof": "outdoors",
}


def prepared():
    return load_frozen_pool(), load_onside_verified()[0]


def pending():
    return {
        "qtr":4, "is_kickoff":1.,
        "_score_change_original":3., "_touchdown_original":0,
        "is_home_posteam":1., "yardline_100":np.nan,
        "down":np.nan,"ydstogo":np.nan,"goal_to_go":0.,
        "game_seconds_remaining":295., "half_seconds_remaining":295.,
        "final_five_minutes":1.,
    }


def test_real_onside_2025_verified_donor_count_and_possession_check():
    pool, report=load_onside_verified()
    assert len(pool)==37
    assert int(pool["official_kicker_recovered"].eq(1).sum())==4
    assert int(pool["official_kicker_recovered"].eq(0).sum())==33
    assert report["all_on_side_eligible"]==41
    assert report["official_kicker_recoveries"]==4


def test_forced_success_reverses_possession_not_score_and_preserves_clock():
    normal, onside=prepared()
    original=pending()
    result=resolve_q4_fg_onside_what_if(
        {"qtr":4,"score_differential":-4.},[original],
        normal,onside, rng_ordinary=np.random.default_rng(10),
        rng_special=np.random.default_rng(10),
        p_attempt=1.,p_recovery=1.,
    )[0]
    assert result["is_home_posteam"]==0.  # Rams recover, Seahawks no longer receive
    assert result["is_kickoff"]==0.
    assert result["down"]==1.
    assert result["_kickoff_transition_type"]=="onside_kicker_recovered"
    assert result["yardline_100"] in set(
        onside.loc[onside["official_kicker_recovered"].eq(1),
                   "yardline_100_snap"]
    )
    assert result["game_seconds_remaining"]<=original["game_seconds_remaining"]
    assert result["half_seconds_remaining"]==result["game_seconds_remaining"]
    assert original["is_kickoff"]==1. and np.isnan(original["yardline_100"])


def test_forced_failed_onside_preserves_receivers_possession():
    normal, onside=prepared()
    got=resolve_q4_fg_onside_what_if(
        {"qtr":4,"score_differential":-4.},[pending()],
        normal,onside,rng_ordinary=np.random.default_rng(13),
        rng_special=np.random.default_rng(13),
        p_attempt=1.,p_recovery=0.,
    )[0]
    assert got["is_home_posteam"]==1.
    assert got["_kickoff_transition_type"]=="onside_receiver_recovered"
    assert got["yardline_100"] in set(
        onside.loc[onside["official_kicker_recovered"].eq(0),
                   "yardline_100_snap"]
    )


def test_on_side_only_after_made_fg_still_trailing_and_clock_above_40():
    normal, onside=prepared()
    bad=pending()
    bad["game_seconds_remaining"]=39.
    long=resolve_q4_fg_onside_what_if(
        {"qtr":4,"score_differential":-4.},[bad],
        normal,onside,rng_ordinary=np.random.default_rng(1),
        rng_special=np.random.default_rng(1),
        p_attempt=1.,p_recovery=1.,
    )[0]
    assert not str(long.get("_kickoff_transition_type")).startswith("onside")
    tied=resolve_q4_fg_onside_what_if(
        {"qtr":4,"score_differential":-3.},[pending()],
        normal,onside,rng_ordinary=np.random.default_rng(1),
        rng_special=np.random.default_rng(1),
        p_attempt=1.,p_recovery=1.,
    )[0]
    assert not str(tied.get("_kickoff_transition_type")).startswith("onside")
    td=pending()
    td["_score_change_original"]=6.
    not_fg=resolve_q4_fg_onside_what_if(
        {"qtr":4,"score_differential":-4.},[td],
        normal,onside,rng_ordinary=np.random.default_rng(1),
        rng_special=np.random.default_rng(1),
        p_attempt=1.,p_recovery=1.,
    )[0]
    assert not str(not_fg.get("_kickoff_transition_type")).startswith("onside")


@pytest.mark.parametrize("attempt,recovery",[
    (-.01,.1), (1.01,.1), (.2,-.01),(.3,1.01),(np.nan,.1),(.1,np.inf),
])
def test_reject_invalid_hypothetical_probabilities(attempt,recovery):
    with pytest.raises(ValueError,match="probabilities"):
        validate_what_if(attempt,recovery)


def test_no_onside_probability_exactly_matches_existing_empirical_preview():
    pool, onside=prepared()
    private=isolated_engine_for_preview()
    original=private["evaluate_post_play_states"]
    with use_q4_onside_what_if(
        private,pool,onside,p_attempt=0.,p_recovery=.10,
    ):
        now=private["recommend"](RAMS,n_simulations=200,seed=42)
    old=preview_observed_2025_recommendation(
        RAMS,simulations=200,action_seed=42,
    )
    pd.testing.assert_frame_equal(now["results"],old["results"],check_exact=True)
    assert original.__globals__["evaluate_post_play_states"] is original


def test_nested_engine_overrides_and_exceptions_restore_global_evaluator():
    pool, onside=prepared()
    private=isolated_engine_for_preview()
    original=private["evaluate_post_play_states"]
    with pytest.raises(RuntimeError,match="intentional"):
        with use_q4_onside_what_if(
            private,pool,onside,p_attempt=.25,p_recovery=.10
        ):
            with pytest.raises(RuntimeError,match="nested"):
                with use_q4_onside_what_if(
                    private,pool,onside,p_attempt=.25,p_recovery=.10
                ):
                    pass
            raise RuntimeError("intentional")
    assert original.__globals__["evaluate_post_play_states"] is original


def test_positive_onside_what_if_is_valid_and_nonproduction():
    pool,onside=prepared()
    private=isolated_engine_for_preview()
    original=private["evaluate_post_play_states"]
    with use_q4_onside_what_if(
        private,pool,onside,p_attempt=.25,p_recovery=.10,
    ):
        result=private["recommend"](RAMS,n_simulations=200,seed=42)
    assert original.__globals__["evaluate_post_play_states"] is original
    rows=result["results"].loc[result["results"]["eligible"]]
    assert rows["expected_win_probability"].between(0,1).all()
    assert set(rows["action"])=={"FIELD_GOAL","NORMAL_GO_PASS"}
