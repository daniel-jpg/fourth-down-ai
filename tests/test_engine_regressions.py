from pathlib import Path
import os
import runpy

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)

ENGINE = runpy.run_path(
    str(ROOT / "src" / "42_build_decision_engine.py")
)

normalize_state = ENGINE["normalize_state"]
action_eligible = ENGINE["action_eligible"]
field_goal_probabilities = ENGINE["field_goal_probabilities"]
blocked_fg_live_probabilities = ENGINE["blocked_fg_live_probabilities"]
make_wp_state = ENGINE["make_wp_state"]
evaluate_post_play_states = ENGINE["evaluate_post_play_states"]
sample_ordinary_punt_state = ENGINE["sample_ordinary_punt_state"]
go_conversion_probability = ENGINE["go_conversion_probability"]


def base_state(**overrides):
    state = {
        "qtr": 4,
        "game_seconds_remaining": 300.0,
        "yardline_100": 40.0,
        "ydstogo": 2.0,
        "score_differential": 0.0,
        "is_home": 1,
        "site": "HOME",
        "roof": "outdoors",
        "posteam_timeouts_remaining": 3.0,
        "defteam_timeouts_remaining": 3.0,
    }
    state.update(overrides)
    return normalize_state(state)


def ot_state(
    *,
    phase="OPENING",
    clock=300.0,
    score=0.0,
    regular_season=True,
    ot_period=1,
    **overrides,
):
    state = {
        "qtr": 5,
        "game_seconds_remaining": float(clock),
        "yardline_100": 25.0,
        "ydstogo": 2.0,
        "score_differential": float(score),
        "is_home": 1,
        "site": "HOME",
        "roof": "outdoors",
        "ot_phase": phase,
        "ot_format": (
            "REGULAR_SEASON"
            if regular_season
            else "POSTSEASON"
        ),
        "ot_period": int(ot_period),
    }
    state.update(overrides)
    return normalize_state(state)


def make_state_after(
    base,
    *,
    possession_original,
    score_change=0.0,
    elapsed=1.0,
    opportunity_completed=False,
):
    state = make_wp_state(
        base,
        possession_original=possession_original,
        yardline_100=50.0,
        down=1.0,
        ydstogo=10.0,
        elapsed_seconds=float(elapsed),
        score_change_original=float(score_change),
        is_kickoff=False,
    )
    if opportunity_completed:
        state["_ot_opportunity_completed"] = True
    return state


@pytest.fixture
def fixed_raw_wp(monkeypatch):
    """Make OT rule tests deterministic and independent of the learned WP model."""
    globals_dict = evaluate_post_play_states.__globals__

    def fake_predict(states, *args, **kwargs):
        return np.full(len(states), 0.37, dtype=float)

    monkeypatch.setitem(
        globals_dict,
        "predict_original_team_wp",
        fake_predict,
    )
    return 0.37


# ---------------------------------------------------------------------------
# State validation / action support
# ---------------------------------------------------------------------------

def test_fourth_and_16_is_valid():
    state = base_state(
        yardline_100=52.0,
        ydstogo=16.0,
    )
    assert state["ydstogo"] == pytest.approx(16.0)
    assert state["yardline_100"] == pytest.approx(52.0)


def test_yards_to_go_cannot_exceed_distance_to_goal():
    with pytest.raises(ValueError):
        base_state(
            yardline_100=15.0,
            ydstogo=16.0,
        )


def test_fourth_and_16_pass_available_run_unsupported():
    state = base_state(
        yardline_100=52.0,
        ydstogo=16.0,
    )

    pass_ok, _ = action_eligible(
        state,
        "NORMAL_GO_PASS",
    )
    run_ok, run_reason = action_eligible(
        state,
        "NORMAL_GO_RUN",
    )

    assert pass_ok
    assert not run_ok
    assert "ydstogo > 3" in run_reason


def test_70_yard_fg_allowed_71_rejected():
    fg70 = base_state(
        yardline_100=52.0,
        ydstogo=10.0,
    )
    fg71 = base_state(
        yardline_100=53.0,
        ydstogo=10.0,
    )

    ok70, _ = action_eligible(
        fg70,
        "FIELD_GOAL",
    )
    ok71, reason71 = action_eligible(
        fg71,
        "FIELD_GOAL",
    )

    assert ok70
    assert not ok71
    assert "70-yard" in reason71


# ---------------------------------------------------------------------------
# Field-goal outcome model
# ---------------------------------------------------------------------------

def test_fg_probabilities_sum_to_one():
    for distance in range(19, 71):
        state = base_state(
            yardline_100=float(distance - 18),
            ydstogo=1.0,
        )
        probs = field_goal_probabilities(state)

        assert set(probs) == {
            "made",
            "missed",
            "blocked",
            "broken",
        }
        assert sum(probs.values()) == pytest.approx(
            1.0,
            abs=1e-12,
        )
        assert all(
            0.0 <= value <= 1.0
            for value in probs.values()
        )


def test_extreme_fg_tail_is_monotone():
    makes = []

    for distance in range(60, 71):
        state = base_state(
            yardline_100=float(distance - 18),
            ydstogo=1.0,
        )
        makes.append(
            field_goal_probabilities(state)["made"]
        )

    assert all(
        later < earlier
        for earlier, later in zip(
            makes,
            makes[1:],
        )
    )

    # Guard against the old unrealistic ~38% 70-yard value.
    assert makes[-1] < 0.10


def test_blocked_fg_live_ball_distribution():
    probs = blocked_fg_live_probabilities()

    assert set(probs) == {
        "opponent_scored",
        "kicking_team_muff_recovery",
        "opponent_ball_no_score",
    }
    assert sum(probs.values()) == pytest.approx(
        1.0,
        abs=1e-12,
    )

    # Rule 16 regression: the original kicking team must have
    # a nonzero live-ball recovery branch.
    assert probs["kicking_team_muff_recovery"] > 0.0


# ---------------------------------------------------------------------------
# Overtime rule layer
# ---------------------------------------------------------------------------

def test_ot_requires_phase():
    state = {
        "qtr": 5,
        "game_seconds_remaining": 300.0,
        "yardline_100": 25.0,
        "ydstogo": 2.0,
        "score_differential": 0.0,
        "is_home": 1,
        "site": "HOME",
        "roof": "outdoors",
    }

    with pytest.raises(
        ValueError,
        match="Overtime states require ot_phase",
    ):
        normalize_state(state)


def test_regular_ot_defaults_to_two_timeouts():
    state = ot_state(
        phase="OPENING",
        regular_season=True,
    )
    assert state["posteam_timeouts_remaining"] == pytest.approx(2.0)
    assert state["defteam_timeouts_remaining"] == pytest.approx(2.0)


def test_opening_score_is_not_automatically_terminal(
    fixed_raw_wp,
):
    base = ot_state(
        phase="OPENING",
        clock=300.0,
        score=0.0,
    )
    state = make_state_after(
        base,
        possession_original=False,
        score_change=3.0,
        elapsed=5.0,
    )

    value = evaluate_post_play_states(
        base,
        [state],
    )[0]

    assert value == pytest.approx(fixed_raw_wp)


def test_opening_defensive_score_is_loss(
    fixed_raw_wp,
):
    base = ot_state(
        phase="OPENING",
        clock=300.0,
        score=0.0,
    )
    state = make_state_after(
        base,
        possession_original=False,
        score_change=-6.0,
        elapsed=5.0,
    )

    value = evaluate_post_play_states(
        base,
        [state],
    )[0]

    assert value == pytest.approx(0.0)


def test_response_fg_while_tied_is_win(
    fixed_raw_wp,
):
    base = ot_state(
        phase="RESPONSE",
        clock=100.0,
        score=0.0,
    )
    state = make_state_after(
        base,
        possession_original=False,
        score_change=3.0,
        elapsed=5.0,
    )

    value = evaluate_post_play_states(
        base,
        [state],
    )[0]

    assert value == pytest.approx(1.0)


def test_response_miss_while_down_three_is_loss(
    fixed_raw_wp,
):
    base = ot_state(
        phase="RESPONSE",
        clock=100.0,
        score=-3.0,
    )
    state = make_state_after(
        base,
        possession_original=False,
        score_change=0.0,
        elapsed=5.0,
    )

    value = evaluate_post_play_states(
        base,
        [state],
    )[0]

    assert value == pytest.approx(0.0)


def test_response_tie_continues_to_sudden_death(
    fixed_raw_wp,
):
    base = ot_state(
        phase="RESPONSE",
        clock=100.0,
        score=-3.0,
    )
    state = make_state_after(
        base,
        possession_original=False,
        score_change=3.0,
        elapsed=5.0,
    )

    value = evaluate_post_play_states(
        base,
        [state],
    )[0]

    assert value == pytest.approx(fixed_raw_wp)


def test_sudden_death_score_is_terminal_win(
    fixed_raw_wp,
):
    base = ot_state(
        phase="SUDDEN_DEATH",
        clock=100.0,
        score=0.0,
    )
    state = make_state_after(
        base,
        possession_original=False,
        score_change=3.0,
        elapsed=5.0,
    )

    value = evaluate_post_play_states(
        base,
        [state],
    )[0]

    assert value == pytest.approx(1.0)


def test_regular_season_ot_expiring_tied_is_half_win(
    fixed_raw_wp,
):
    base = ot_state(
        phase="RESPONSE",
        clock=1.0,
        score=0.0,
        regular_season=True,
    )
    state = make_state_after(
        base,
        possession_original=True,
        score_change=0.0,
        elapsed=1.0,
    )

    value = evaluate_post_play_states(
        base,
        [state],
    )[0]

    assert value == pytest.approx(0.5)


def test_opportunity_completed_can_terminalize_response(
    fixed_raw_wp,
):
    base = ot_state(
        phase="RESPONSE",
        clock=100.0,
        score=-3.0,
    )
    state = make_state_after(
        base,
        possession_original=True,
        score_change=0.0,
        elapsed=1.0,
        opportunity_completed=True,
    )

    value = evaluate_post_play_states(
        base,
        [state],
    )[0]

    assert value == pytest.approx(0.0)



# ---------------------------------------------------------------------------
# GO conversion calibration
# ---------------------------------------------------------------------------

def test_normal_run_short_yardage_calibration(
    monkeypatch,
):
    class FakeGoModel:
        def predict_proba(
            self,
            X,
        ):
            return np.array(
                [[0.40, 0.60]],
                dtype=float,
            )

    monkeypatch.setitem(
        go_conversion_probability.__globals__,
        "go_model",
        FakeGoModel(),
    )

    short_state = base_state(
        ydstogo=1.0,
    )

    raw_probability = 0.60

    raw_logit = np.log(
        raw_probability
        /
        (1.0 - raw_probability)
    )

    expected = (
        1.0
        /
        (
            1.0
            +
            np.exp(
                -(
                    raw_logit
                    +
                    0.2925
                )
            )
        )
    )

    run_probability = (
        go_conversion_probability(
            short_state,
            "NORMAL_GO_RUN",
        )
    )

    raw_run_probability = (
        go_conversion_probability(
            short_state,
            "NORMAL_GO_RUN",
            apply_calibration=False,
        )
    )

    pass_probability = (
        go_conversion_probability(
            short_state,
            "NORMAL_GO_PASS",
        )
    )

    assert run_probability == pytest.approx(
        expected,
        abs=1e-12,
    )

    # Validation diagnostics must be able to request the
    # raw pre-calibration model probability.
    assert raw_run_probability == pytest.approx(
        raw_probability,
        abs=1e-12,
    )

    # Passing probabilities must remain untouched.
    assert pass_probability == pytest.approx(
        raw_probability,
        abs=1e-12,
    )


def test_normal_run_three_yards_is_recalibrated(
    monkeypatch,
):
    class FakeGoModel:
        def predict_proba(
            self,
            X,
        ):
            return np.array(
                [[0.40, 0.60]],
                dtype=float,
            )

    monkeypatch.setitem(
        go_conversion_probability.__globals__,
        "go_model",
        FakeGoModel(),
    )

    state = base_state(
        ydstogo=3.0,
    )

    probability = (
        go_conversion_probability(
            state,
            "NORMAL_GO_RUN",
        )
    )

    raw_probability = 0.60

    raw_logit = np.log(
        raw_probability
        /
        (1.0 - raw_probability)
    )

    expected = (
        1.0
        /
        (
            1.0
            +
            np.exp(
                -(
                    raw_logit
                    +
                    0.2925
                )
            )
        )
    )

    assert probability == pytest.approx(
        expected,
        abs=1e-12,
    )


# ---------------------------------------------------------------------------
# Punt ordinary-state sampling
# ---------------------------------------------------------------------------

def test_punt_touchback_donor_forces_own_20(monkeypatch):
    globals_dict = sample_ordinary_punt_state.__globals__

    pool = pd.DataFrame(
        {
            "start_yardline_bucket": [60.0],
            "yardline_residual": [-15.0],
            "pbp_touchback": [1.0],
            "seconds_to_state": [9.0],
        }
    )

    monkeypatch.setitem(
        globals_dict,
        "punt_pool",
        pool,
    )

    state = base_state(
        yardline_100=60.0,
        ydstogo=10.0,
    )

    result = sample_ordinary_punt_state(
        state,
        predicted_yardline=70.0,
        rng=np.random.default_rng(1),
    )

    assert result["yardline_100"] == pytest.approx(
        80.0
    )
    assert result["ydstogo"] == pytest.approx(
        10.0
    )


def test_punt_non_touchback_uses_prediction_plus_residual(
    monkeypatch,
):
    globals_dict = sample_ordinary_punt_state.__globals__

    pool = pd.DataFrame(
        {
            "start_yardline_bucket": [60.0],
            "yardline_residual": [4.5],
            "pbp_touchback": [0.0],
            "seconds_to_state": [9.0],
        }
    )

    monkeypatch.setitem(
        globals_dict,
        "punt_pool",
        pool,
    )

    state = base_state(
        yardline_100=60.0,
        ydstogo=10.0,
    )

    result = sample_ordinary_punt_state(
        state,
        predicted_yardline=70.0,
        rng=np.random.default_rng(2),
    )

    assert result["yardline_100"] == pytest.approx(
        74.5
    )
    assert result["ydstogo"] == pytest.approx(
        10.0
    )
