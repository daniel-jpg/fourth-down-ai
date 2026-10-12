"""Fabricated-only nflverse -> registered frozen policy end-to-end tests."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pandas as pd
import pytest

from src import nflverse_2027_frozen_policy_replay as replay
from src.nflverse_2027_reconstruction import MODE_SYNTHETIC
from src.frozen_2027_policy_bridge import (
    SNAPSHOT_KEYS, load_verified_engine, build_synthetic_policy_forecast,
)
from src.historical_nuisance_portable import (
    load_historical_parameters, assert_historical_source_integrity,
)


def play(**changes):
    obj = {
        "season": 2027, "season_type": "REG", "week": 3,
        "game_id": "synthetic_2027_03_ABC_DEF", "play_id": 101.,
        "down": 4., "qtr": 1., "game_seconds_remaining": 3100.,
        "yardline_100": 6., "ydstogo": 4., "goal_to_go": 0.,
        "score_differential": -4., "posteam_timeouts_remaining": 3.,
        "defteam_timeouts_remaining": 2., "posteam": "ABC",
        "home_team": "ABC", "away_team": "DEF",
        "location": "Home", "roof": "outdoors",
        # Deliberately poisonous hindsight fields that the engine MUST NOT see.
        "play_type": "punt", "desc": "post-snap punt result",
        "result": -17., "yards_gained": -1000., "epa": 1_000_000.,
        "offense_win": 0, "observed_action": "PUNT",
        "touchdown": 1, "field_goal_result": "made",
    }
    obj.update(changes)
    return obj


class FakeEngine:
    """Deterministic fabricated action engine, not an NFL model."""
    ACTIONS = ("NORMAL_GO_RUN", "NORMAL_GO_PASS", "PUNT", "FIELD_GOAL")

    def normalize_state(self, state):
        return dict(state)

    def action_eligible(self, state, action):
        return action in ("NORMAL_GO_PASS", "FIELD_GOAL"), ""

    def recommend(self, state, n_simulations, seed):
        assert seed in (42, 43)
        scores = {
            "NORMAL_GO_RUN": None, "NORMAL_GO_PASS": .55,
            "PUNT": None, "FIELD_GOAL": .58,
        }
        rows = [{
            "action": action, "eligible": scores[action] is not None,
            "expected_win_probability": scores[action],
            "mc_se": .01 if scores[action] is not None else float("nan"),
        } for action in self.ACTIONS]
        data = pd.DataFrame(rows)
        return {"results": pd.concat([
            data[data["eligible"]].sort_values(
                "expected_win_probability", ascending=False
            ),
            data[~data["eligible"]],
        ], ignore_index=True)}


@pytest.fixture(scope="module")
def nuisance():
    frozen = load_historical_parameters()
    assert_historical_source_integrity(frozen)
    return frozen


def exercise(rows, nuisance, *, n=24, engine=None):
    return replay.replay_fabricated_nflverse_through_frozen_policy(
        pd.DataFrame(rows), mode=MODE_SYNTHETIC,
        engine=engine if engine is not None else FakeEngine(),
        nuisance_snapshot=nuisance, n_simulations=n,
    )


def test_nflverse_fixture_flows_into_existing_frozen_policy_without_labels(nuisance):
    result = exercise([play()], nuisance)
    assert result["manifest"]["engine_used"] == "TEST_INJECTED_SYNTHETIC_ENGINE"
    assert result["manifest"]["policy_commit"] == replay.POLICY_COMMIT
    assert result["audit"]["source_fourth_down_candidates"] == 1
    assert result["audit"]["forecasted_fabricated_opportunities"] == 1
    assert result["audit"]["unscored_opportunities"] == 0
    assert result["audit"]["calibration_evaluated"] is False
    forecast = result["synthetic_forecasts"][0]
    assert forecast["policy_action"] == "FIELD_GOAL"
    assert forecast["q_wp_by_action"] == {
        "NORMAL_GO_PASS": .55, "FIELD_GOAL": .58
    }
    assert forecast["feature_asof"] == "pre_snap"  # synthetic adapter ONLY
    assert 0 <= forecast["mu_policy"] <= 1
    assert 0 <= forecast["propensity_policy"] <= 1
    assert len(result["audit"]["synthetic_prediction_sha256"]) == 64
    audit = result["per_forecast_engine_audit"][0]
    assert audit["simulation_count"] == 24
    assert audit["full_precision"] is False
    assert audit["kickoff_input_mode"] == "NO_Q4_KICKOFF_PATCH"
    assert all(result["audit"][k] is False for k in (
        "source_labels_accessed", "observed_actions_accessed",
        "final_game_results_accessed", "2026_reserved_data_accessed",
        "real_2027_outcomes_accessed", "future_season_scoring_authorized",
        "independent_pre_snap_provenance",
    ))


def test_future_postplay_labels_cannot_change_any_forecast(nuisance):
    first = exercise([play()], nuisance)
    changed = play(
        play_type="run", desc="fake FG, completed touchdown",
        result=45., yards_gained=99., epa=-600.,
        offense_win=1, observed_action="NORMAL_GO_RUN",
        field_goal_result="blocked", touchdown=0,
    )
    second = exercise([changed], nuisance)
    assert first["synthetic_forecasts"] == second["synthetic_forecasts"]
    assert (first["audit"]["synthetic_prediction_sha256"]
            == second["audit"]["synthetic_prediction_sha256"])
    assert first["skipped_opportunities"] == second["skipped_opportunities"]


def test_missing_q2_receiving_team_is_explicitly_skipped_before_engine_load():
    bad = play(qtr=2., game_seconds_remaining=2150.)
    out = replay.replay_fabricated_nflverse_through_frozen_policy(
        pd.DataFrame([bad]), mode=MODE_SYNTHETIC,
        # The default engine and historical parameters must not even be loaded.
    )
    assert not out["synthetic_forecasts"]
    assert out["manifest"]["engine_used"] == "NOT_RUN"
    assert out["audit"]["forecasted_fabricated_opportunities"] == 0
    assert out["audit"]["unscored_opportunities"] == 1
    assert out["skipped_opportunities"][0]["reason"] == "MODEL_CONTEXT_MISSING"
    assert "Q2_SECOND_HALF_RECEIVER_UNAVAILABLE" in out["skipped_opportunities"][0]["detail"]


def test_roof_and_goal_line_mismatches_are_not_imputed(nuisance):
    rows = [
        play(play_id=101., roof=None),
        play(play_id=202., goal_to_go=1.),
        play(play_id=303., roof="outdoors", goal_to_go=0.),
    ]
    out = exercise(rows, nuisance)
    assert out["audit"]["forecasted_fabricated_opportunities"] == 1
    assert out["audit"]["unscored_opportunities"] == 2
    assert out["audit"]["skip_reason_counts"] == {
        "FROZEN_POLICY_STATE_UNSUPPORTED": 1,
        "MODEL_CONTEXT_MISSING": 1,
    }
    assert {r["play_id"] for r in out["synthetic_forecasts"]} == {"303"}


def test_source_exclusions_are_preserved_in_full_decision_denominator(nuisance):
    rows = [
        play(play_id=101.),
        play(play_id=202., season_type="POST"),
        play(play_id=303., qtr=5., game_seconds_remaining=800.),
        play(play_id=404., yardline_100=-3.),
    ]
    out = exercise(rows, nuisance)
    assert out["audit"]["source_fourth_down_candidates"] == 4
    assert out["audit"]["source_exclusions"] == 3
    assert out["audit"]["unscored_opportunities"] == 3
    assert out["audit"]["forecasted_fabricated_opportunities"] == 1
    assert out["audit"]["skip_reason_counts"] == {
        "MISSING_OR_INVALID_PREPLAY_STATE": 1,
        "NOT_REGULAR_SEASON": 1,
        "OVERTIME": 1,
    }


@pytest.mark.parametrize("mode", [
    "prospective", "real_2027", "historical_2014_2025_development",
    "retrospective", None, "",
])
def test_real_holdout_is_refused_before_input_inspection(mode):
    class Trap:
        def __getattribute__(self, attr):
            raise AssertionError("Forbidden future data was inspected!")
    with pytest.raises(PermissionError, match="complete 2027 regular season"):
        replay.replay_fabricated_nflverse_through_frozen_policy(
            Trap(), mode=mode,
        )


def test_real_2027_pbp_identifiers_do_not_enter_a_fabricated_test(nuisance):
    with pytest.raises(ValueError, match="synthetic_"):
        exercise([play(game_id="2027_03_ABC_DEF")], nuisance)


def test_source_size_guard_and_duplicate_keys_prevent_silent_drops(nuisance):
    many = [play(play_id=float(i)) for i in range(101, 135)]
    with pytest.raises(ValueError, match="too large"):
        exercise(many, nuisance)
    with pytest.raises(ValueError, match="Duplicate"):
        exercise([play(), play()], nuisance)


def test_q4_refuses_unregistered_short_simulation_budget(nuisance):
    q4 = play(qtr=4., game_seconds_remaining=299.)
    with pytest.raises(ValueError, match="Q4 native kickoff policy"):
        exercise([q4], nuisance, n=24)


def test_polars_nflreadpy_frame_is_supported_without_postplay_access(nuisance):
    pl = pytest.importorskip("polars")
    source = pd.DataFrame([play()])
    p = replay.replay_fabricated_nflverse_through_frozen_policy(
        pl.from_pandas(source), mode=MODE_SYNTHETIC,
        engine=FakeEngine(), nuisance_snapshot=nuisance, n_simulations=24,
    )
    f = exercise([play()], nuisance)
    assert p["synthetic_forecasts"] == f["synthetic_forecasts"]
    assert p["audit"] == f["audit"]


def test_nonwhitelisted_snapshot_cannot_be_supplied_directly():
    row = {
        key: (0 if key not in ("game_id", "feature_asof") else "synthetic_bad")
        for key in replay.PREPLAY_OUTPUT_FIELDS
    }
    row["observed_action"] = "FIELD_GOAL"
    with pytest.raises(ValueError, match="label-free"):
        replay._synthetic_engine_snapshot(row)


def test_full_registered_q4_engine_from_nflverse_shape_matches_frozen_bridge(nuisance):
    """True registered 1200/seed43 Q4 path, but input is wholly fictional."""
    q4 = play(qtr=4., game_seconds_remaining=299.)
    result = replay.replay_fabricated_nflverse_through_frozen_policy(
        pd.DataFrame([q4]), mode=MODE_SYNTHETIC,
        nuisance_snapshot=nuisance,
    )
    assert result["manifest"]["engine_used"] == "PINNED_REGISTERED_ENGINE"
    assert result["manifest"]["full_precision_requested"] is True
    assert result["audit"]["forecasted_fabricated_opportunities"] == 1
    assert result["synthetic_forecasts"][0]["policy_action"] in {
        "NORMAL_GO_RUN", "NORMAL_GO_PASS", "PUNT", "FIELD_GOAL"
    }
    cfg = result["per_forecast_engine_audit"][0]
    assert cfg["full_precision"] is True
    assert cfg["simulation_count"] == 1200
    assert cfg["final_seed"] == 43
    assert cfg["kickoff_input_mode"] == "Q4_NATIVE_STANDARD_35"


def test_nonfrozen_seed_or_real_season_never_promoted(nuisance):
    with pytest.raises(ValueError, match="simulation budget"):
        exercise([play()], nuisance, n=1201)
    with pytest.raises(ValueError, match="forbidden season"):
        exercise([play(season=2026)], nuisance)


def test_adapter_does_not_download_games_or_change_frozen_model_assets():
    source = (Path(__file__).resolve().parents[1]
              / "src/nflverse_2027_frozen_policy_replay.py").read_text()
    assert "nfl.load_pbp(" not in source
    assert "nflreadpy.load_pbp(" not in source
    assert "joblib.dump(" not in source
    assert "2026_forward_kickoff_wp_final" not in source
    assert "real_2027_outcomes_accessed" in source
    assert "future_season_scoring_authorized" in source
