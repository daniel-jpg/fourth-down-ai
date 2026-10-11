"""Only verified score-neutral 2025 penalty first snaps join ordinary donors."""

import numpy as np
import pandas as pd
import pytest

from src.kickoff_2025_empirical import load_frozen_pool
from src.kickoff_2025_resolved import (
    NEW_TYPE, ORDINARY_COUNT, PENALTY_COUNT, TOTAL_COUNT,
    load_resolved_2025_pool, resolve_post_score_q4_resolved,
    use_resolved_2025_kickoffs, validate_resolved_pool,
)
from src.streamlit_kickoff_preview import (
    isolated_engine_for_preview, preview_observed_2025_recommendation,
    preview_observed_2025_resolved_recommendation,
    preview_observed_2025_with_onside, observed_2025_resolved_summary,
)


RAMS = {
    "qtr": 4, "game_seconds_remaining": 299.,
    "yardline_100": 6., "ydstogo": 4.,
    "score_differential": -4.,
    "posteam_timeouts_remaining": 3.,
    "defteam_timeouts_remaining": 3.,
    "site": "AWAY", "roof": "outdoors",
}


def pending(**extra):
    x = {
        "qtr":4,"is_kickoff":1., "game_seconds_remaining":295.,
        "half_seconds_remaining":295., "is_home_posteam":1.,
        "_score_change_original":3., "_kickoff_from_50":False,
        "down":np.nan,"ydstogo":np.nan, "yardline_100":np.nan,
        "goal_to_go":0., "home_score_differential":1.,
        "final_five_minutes":1.,
    }
    x.update(extra)
    return x


def eligible(result):
    return result["results"].loc[
        result["results"]["eligible"],
        ["action","expected_win_probability","mc_se"],
    ].sort_values("action").reset_index(drop=True)


def test_2025_ordinary_plus_penalty_pool_exactly_reconciles_real_coverage():
    raw = load_resolved_2025_pool()
    old = load_frozen_pool()
    summary = observed_2025_resolved_summary()
    assert len(raw) == TOTAL_COUNT == 538
    assert len(old) == ORDINARY_COUNT == 488
    assert raw["transition_type"].eq(NEW_TYPE).sum() == PENALTY_COUNT == 50
    assert raw["game_id"].notna().all()
    assert not raw.duplicated(["game_id","kick_play_id"]).any()
    pd.testing.assert_frame_equal(
        raw.iloc[:488].reset_index(drop=True),
        old.reset_index(drop=True),
        check_exact=True,
    )
    assert summary["kickoffs"] == 538
    assert summary["ordinary"] == 488
    assert summary["penalty_first_snaps"] == 50
    assert summary["late_verified_penalties"] == 14
    assert summary["excluded_rekicks"] == 2
    assert summary["excluded_return_touchdowns"] == 1
    assert summary["excluded_onside_attempts"] == 41
    assert summary["excluded_unmatched"] == 2


def test_false_2026_donors_or_duplicate_penalty_kicks_fail_closed():
    pool = load_resolved_2025_pool()
    with pytest.raises(ValueError, match="not verified"):
        validate_resolved_pool(pool.assign(season=2026))
    dup = pool.copy()
    dup.loc[488,["game_id","kick_play_id"]] = pool.loc[0,["game_id","kick_play_id"]]
    with pytest.raises(ValueError, match="not verified"):
        validate_resolved_pool(dup)


class StubRng:
    def __init__(self, index):
        self.index=index
    def integers(self, lo, hi, size=None):
        if size == 1:
            assert lo==0 and hi==538
            return np.array([self.index])
        assert lo==0 and hi==488
        return 0


def test_selected_real_penalty_first_snap_preserves_receiving_team_and_score():
    pool = load_resolved_2025_pool()
    old = pending()
    donor = pool.iloc[488]
    out = resolve_post_score_q4_resolved(
        [old], pool, rng=StubRng(488),
    )[0]
    assert out["_kickoff_transition_type"] == NEW_TYPE
    assert out["is_home_posteam"] == 1.
    assert out["home_score_differential"] == 1.
    assert out["yardline_100"] == pytest.approx(100. - donor["own_yardline"])
    assert out["game_seconds_remaining"] == pytest.approx(
        295. - donor["elapsed_seconds"]
    )
    assert out["down"] == 1.
    assert out["is_kickoff"] == 0.
    assert np.isnan(old["yardline_100"]) and old["is_kickoff"]==1.


def test_unverified_midfield_penalty_donor_is_not_applied_to_midfield_kick():
    pool = load_resolved_2025_pool()
    out = resolve_post_score_q4_resolved(
        [pending(_kickoff_from_50=True)],
        pool, rng=StubRng(488),
    )[0]
    assert out["_kickoff_transition_type"] != NEW_TYPE
    assert out["is_home_posteam"] == 1.


def test_new_default_scenario_preserves_production_engine_and_action_scope():
    private=isolated_engine_for_preview()
    orig=private["evaluate_post_play_states"]
    pool=load_resolved_2025_pool()
    with use_resolved_2025_kickoffs(private,pool):
        result=private["recommend"](RAMS,n_simulations=220,seed=42)
    assert orig.__globals__["evaluate_post_play_states"] is orig
    assert eligible(result)["expected_win_probability"].between(0,1).all()
    test=preview_observed_2025_resolved_recommendation(
        RAMS,simulations=220,action_seed=42,
    )
    pd.testing.assert_frame_equal(eligible(result),eligible(test),check_exact=True)


def test_on_side_zero_attempt_scenario_equals_penalty_expanded_baseline():
    baseline=preview_observed_2025_resolved_recommendation(
        RAMS,simulations=220,action_seed=42,
    )
    onside=preview_observed_2025_with_onside(
        RAMS,attempt_probability=0.,recovery_probability=.10,
        include_verified_penalties=True,simulations=220,action_seed=42,
    )
    pd.testing.assert_frame_equal(
        eligible(baseline),eligible(onside),check_exact=True
    )


def test_original_2025_ordinary_preview_is_still_available():
    previous=preview_observed_2025_recommendation(
        RAMS,simulations=220,action_seed=42
    )
    alternative=preview_observed_2025_with_onside(
        RAMS,attempt_probability=0.,recovery_probability=.10,
        include_verified_penalties=False,simulations=220,action_seed=42,
    )
    pd.testing.assert_frame_equal(
        eligible(previous),eligible(alternative),check_exact=True,
    )
