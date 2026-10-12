"""Synthetic only: verify frozen 2027 policy bridge, no coach/outcome lookahead."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.fourth_down_cohort_builder import REQUIRED_MANIFEST, build_synthetic_cohort
from src.fourth_down_policy_study import ACTIONS, POLICY_COMMIT
from src.frozen_2027_policy_bridge import (
    PINNED_ASSETS, PINNED_ENGINE_BLOB, RANDOM_SEED,
    build_synthetic_policy_forecast, check_pinned_engine,
    engine_state, load_verified_engine, validate_synthetic_snapshot,
)
from src.historical_nuisance_portable import (
    assert_historical_source_integrity, load_historical_parameters,
)

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = {
    "mode": "synthetic",
    "source_type": "fabricated_only",
    "policy_commit": POLICY_COMMIT,
}


def fabricated(**overrides):
    row = {
        "game_id": "synthetic_2027_08_ABC_DEF",
        "play_id": "3210",
        "season": 2027, "week": 8, "quarter": 1, "down": 4,
        "game_seconds_remaining": 3100.,
        "yardline_100": 6., "ydstogo": 4., "goal_to_go": 0.,
        "score_differential": -4.,
        "posteam_timeouts_remaining": 3,
        "defteam_timeouts_remaining": 2,
        "offense_team": "ABC", "home_team": "ABC", "away_team": "DEF",
        "site": "HOME", "site_advantage": 1, "feature_asof": "pre_snap",
        "roof": "outdoors", "second_half_receiver": None,
        "source_type": "fabricated_only",
    }
    row.update(overrides)
    return row


class SyntheticEngine:
    ACTION_ORDER = [
        "NORMAL_GO_RUN", "NORMAL_GO_PASS", "PUNT", "FIELD_GOAL",
        "FAKE_PUNT_RUN", "FAKE_PUNT_PASS", "FAKE_FG_RUN", "FAKE_FG_PASS",
    ]

    def normalize_state(self, state):
        return dict(state)

    def action_eligible(self, state, action):
        if action.startswith("FAKE_"):
            return False, "no empirical fake support"
        if action == "NORMAL_GO_RUN" and state["ydstogo"] > 3:
            return False, "run unsupported past 3 yards"
        if action == "FIELD_GOAL" and state["yardline_100"] + 18 > 70:
            return False, "FG too far"
        if action == "PUNT" and state["yardline_100"] < 20:
            return False, "punt not supported"
        return True, ""

    def recommend(self, state, n_simulations, seed):
        assert seed in (RANDOM_SEED, 43)
        assert 2 <= n_simulations <= 4000
        # Every stub result is computed from pre-snap inputs ONLY.
        weights = {
            "NORMAL_GO_RUN": .57, "NORMAL_GO_PASS": .55,
            "PUNT": .45, "FIELD_GOAL": .58,
        }
        result = []
        for act in self.ACTION_ORDER:
            available, detail = self.action_eligible(state, act)
            result.append({
                "action": act, "eligible": available,
                "expected_win_probability": weights.get(act, np.nan) if available else np.nan,
                "mc_se": .008 if available else np.nan,
                "detail": detail,
            })
        data = pd.DataFrame(result)
        return {
            "results": pd.concat([
                data[data["eligible"]].sort_values(
                    "expected_win_probability", ascending=False
                ), data[~data["eligible"]],
            ], ignore_index=True),
        }


@pytest.fixture(scope="module")
def snapshot():
    frozen = load_historical_parameters()
    assert_historical_source_integrity(frozen)
    return frozen


def test_all_pinned_engine_assets_match_registered_policy_commit():
    checked = check_pinned_engine(ROOT)
    assert checked["src/42_build_decision_engine.py"] == PINNED_ENGINE_BLOB
    assert set(PINNED_ASSETS) == set(checked) - {
        "src/42_build_decision_engine.py"
    }
    assert len(PINNED_ASSETS) >= 12


def test_synthetic_prediction_contains_every_supported_normal_action(snapshot):
    r = build_synthetic_policy_forecast(
        MANIFEST, fabricated(), engine=SyntheticEngine(),
        nuisance_snapshot=snapshot, n_simulations=24,
    )
    row = r["pre_snap"]
    assert row["policy_action"] == "FIELD_GOAL"
    assert row["eligible_actions"] == ["NORMAL_GO_PASS", "FIELD_GOAL"]
    assert row["q_wp_by_action"] == {
        "NORMAL_GO_PASS": .55, "FIELD_GOAL": .58
    }
    assert r["audit"]["frozen_policy_commit"] == POLICY_COMMIT
    assert r["audit"]["full_precision"] is False
    assert r["audit"]["prospectively_validated"] is False
    assert r["audit"]["observed_action_access"] is False
    assert r["audit"]["outcome_access"] is False
    assert 0 <= row["mu_policy"] <= 1
    assert 0 <= row["propensity_policy"] <= 1
    assert row["close_call"] is False


def test_exact_row_plugs_into_existing_synthetic_cohort_builder(snapshot):
    generated = build_synthetic_policy_forecast(
        MANIFEST, fabricated(), engine=SyntheticEngine(),
        nuisance_snapshot=snapshot, n_simulations=32,
    )["pre_snap"]
    # The observed choice and final result are joined later, NOT forecast inputs.
    co = build_synthetic_cohort({
        "manifest": dict(REQUIRED_MANIFEST),
        "pre_snap": [generated],
        "observed_plays": [{
            "game_id": generated["game_id"], "play_id": generated["play_id"],
            "observed_action": "NORMAL_GO_PASS", "no_play": False,
        }],
        "final_games": [{
            "game_id": generated["game_id"], "season": 2027,
            "week": generated["week"], "home_team": "ABC",
            "away_team": "DEF", "home_score_final": 27,
            "away_score_final": 31, "status": "final", "game_type": "REG",
        }],
    })
    assert co["included_decisions"] == 1
    row = co["records"][0]
    assert row["observed_action"] == "NORMAL_GO_PASS"
    assert row["policy_action"] == "FIELD_GOAL"
    assert row["offense_win"] == 0
    assert row["q_wp_observed"] == .55
    assert row["q_wp_policy"] == .58
    assert row["mu_policy"] == generated["mu_policy"]


def test_own5_and_q2_half_receiver_without_assuming_missing_data(snapshot):
    own5 = fabricated(
        quarter=1, game_seconds_remaining=3100., site="AWAY",
        offense_team="DEF", site_advantage=-1,
        yardline_100=95., ydstogo=3., goal_to_go=0.,
    )
    out = build_synthetic_policy_forecast(
        MANIFEST, own5, engine=SyntheticEngine(),
        nuisance_snapshot=snapshot, n_simulations=32,
    )
    assert out["pre_snap"]["policy_action"] == "NORMAL_GO_RUN"
    # RUN/PASS are nearly tied, which escalates the app's action-level
    # Monte Carlo budget; it must NOT flag the GO-vs-PUNT strategy
    # comparison as a close call.
    assert out["audit"]["adaptive_escalated"] is True
    assert out["pre_snap"]["close_call"] is False
    assert "PUNT" in out["pre_snap"]["eligible_actions"]
    assert "FIELD_GOAL" not in out["pre_snap"]["eligible_actions"]

    q2 = fabricated(
        quarter=2, game_seconds_remaining=1810.,
        second_half_receiver="DEFENSE", roof="dome",
        yardline_100=30., ydstogo=2.,
    )
    p = engine_state(q2)
    assert p["second_half_receiver"] == "DEFENSE"
    assert p["roof"] == "dome"
    assert p["goal_to_go"] == 0
    other = deepcopy(q2)
    other["second_half_receiver"] = None
    with pytest.raises(ValueError, match="second-half kickoff"):
        validate_synthetic_snapshot(other)


@pytest.mark.parametrize("key,value,error", [
    ("source_type", "nflverse_real", "not authorized"),
    ("season", 2026, "season"),
    ("down", 3, "down"),
    ("roof", None, "roof"),
    ("roof", "unknown", "roof"),
    ("goal_to_go", 1, "goal_to_go"),
    ("game_seconds_remaining", 2100, "Clock"),
    ("site_advantage", -1, "Site"),
    ("site", "AWAY", "Site"),
    ("posteam_timeouts_remaining", 4, "posteam_timeouts"),
])
def test_invalid_presnap_case_fails_closed(key, value, error):
    record = fabricated()
    record[key] = value
    with pytest.raises(ValueError, match=error):
        validate_synthetic_snapshot(record)


@pytest.mark.parametrize("sensitive", [
    "offense_win", "observed_action", "yards_gained",
    "converted", "result", "desc", "post_snap_epa",
    "final_home_margin", "coach_action",
])
def test_no_post_snap_information_can_enter_forecast_input(sensitive, snapshot):
    record = fabricated()
    record[sensitive] = 1
    with pytest.raises(ValueError, match="Exact synthetic pre-snap keys"):
        build_synthetic_policy_forecast(
            MANIFEST, record, engine=SyntheticEngine(),
            nuisance_snapshot=snapshot, n_simulations=16,
        )


def test_inconsistent_engine_rank_or_eligibility_rejected(snapshot):
    class WrongRank(SyntheticEngine):
        def recommend(self, state, n_simulations, seed):
            out = super().recommend(state, n_simulations, seed)
            out["results"] = out["results"].iloc[::-1].reset_index(drop=True)
            return out
    with pytest.raises(RuntimeError, match="max-WP"):
        build_synthetic_policy_forecast(
            MANIFEST, fabricated(), engine=WrongRank(),
            nuisance_snapshot=snapshot, n_simulations=8,
        )
    class WrongEligibility(SyntheticEngine):
        def recommend(self, state, n_simulations, seed):
            out = super().recommend(state, n_simulations, seed)
            out["results"].loc[
                out["results"]["action"] == "PUNT", "eligible"
            ] = True
            return out
    with pytest.raises(RuntimeError, match="eligibility inconsistent"):
        build_synthetic_policy_forecast(
            MANIFEST, fabricated(), engine=WrongEligibility(),
            nuisance_snapshot=snapshot, n_simulations=8,
        )


def test_exact_go_action_ties_and_strategy_ties_do_not_fabricate_winner(snapshot):
    class TieEngine(SyntheticEngine):
        def recommend(self, state, n_simulations, seed):
            out = super().recommend(state, n_simulations, seed)
            rows = out["results"].copy()
            both_go_fg = rows["action"].isin([
                "NORMAL_GO_RUN", "NORMAL_GO_PASS", "FIELD_GOAL",
            ])
            rows.loc[both_go_fg, "expected_win_probability"] = .58
            out["results"] = pd.concat([
                rows.loc[rows["eligible"]].sort_values(
                    "expected_win_probability", ascending=False,
                ),
                rows.loc[~rows["eligible"]],
            ], ignore_index=True)
            return out
    state = fabricated(ydstogo=2., goal_to_go=0.)
    result = build_synthetic_policy_forecast(
        MANIFEST, state, engine=TieEngine(), nuisance_snapshot=snapshot,
        n_simulations=24,
    )
    assert result["pre_snap"]["policy_action"] in {
        "NORMAL_GO_RUN", "NORMAL_GO_PASS", "FIELD_GOAL",
    }
    assert result["pre_snap"]["close_call"] is True
    assert result["pre_snap"]["q_wp_by_action"]["FIELD_GOAL"] == .58


def test_non_fabricated_manifest_or_nonfrozen_seed_rejected(snapshot):
    record = fabricated()
    for fake in [
        {**MANIFEST, "mode": "prospective"},
        {**MANIFEST, "source_type": "real"},
        {**MANIFEST, "offense_win": 1},
        {**MANIFEST, "policy_commit": "main"},
    ]:
        with pytest.raises(ValueError, match="synthetic fabricated pinned"):
            build_synthetic_policy_forecast(
                fake, record, engine=SyntheticEngine(),
                nuisance_snapshot=snapshot, n_simulations=8,
            )
    with pytest.raises(ValueError, match="seed"):
        build_synthetic_policy_forecast(
            MANIFEST, record, engine=SyntheticEngine(),
            nuisance_snapshot=snapshot, n_simulations=8, seed=43,
        )


def test_real_pinned_engine_smoke_q4_site_and_q2_halftime(snapshot):
    """Real frozen 2027 simulator integration, but fabricated inputs only."""
    engine = load_verified_engine(ROOT)
    examples = [
        fabricated(
            game_id="synthetic_2027_q4_backed_up",
            quarter=4, game_seconds_remaining=299.,
            score_differential=-4, yardline_100=6.,
            ydstogo=4., goal_to_go=0., roof="outdoors",
        ),
        fabricated(
            game_id="synthetic_2027_q2_halftime",
            quarter=2, game_seconds_remaining=1804.,
            yardline_100=31., ydstogo=2., goal_to_go=0.,
            second_half_receiver="DEFENSE", roof="dome",
        ),
        fabricated(
            game_id="synthetic_2027_neutral",
            site="NEUTRAL", site_advantage=0, roof="closed",
            offense_team="DEF", quarter=1, game_seconds_remaining=3075.,
            yardline_100=45., ydstogo=3., goal_to_go=0.,
        ),
    ]
    for sample in examples:
        n = 200 if sample["quarter"] == 4 else 32
        result = build_synthetic_policy_forecast(
            MANIFEST, sample, engine=engine, nuisance_snapshot=snapshot,
            n_simulations=n,
        )
        # Re-run EXACT registered Q4-native / Q1-3-adaptive branch with the
        # same frozen seed and actual budget, not an unpatched Q4 engine.
        if sample["quarter"] == 4:
            from src.streamlit_kickoff_preview import (
                isolated_engine_from_loaded_engine,
                preview_native_kickoff_feature_candidate,
            )
            private = isolated_engine_from_loaded_engine(engine)
            actual = preview_native_kickoff_feature_candidate(
                engine_state(sample), simulations=n, action_seed=43,
                isolated_engine=private,
            )
            assert result["audit"]["kickoff_input_mode"] == "Q4_NATIVE_STANDARD_35"
            assert result["audit"]["seed"] == 43
        else:
            meta = result["audit"]
            actual = engine.recommend(
                engine_state(sample),
                n_simulations=meta["simulation_count"], seed=meta["seed"],
            )
            assert meta["kickoff_input_mode"] == "NO_Q4_KICKOFF_PATCH"
            assert meta["pilot_simulations"] == 32
        computed = actual["results"]
        eligible = computed.loc[
            computed["eligible"] & computed["action"].isin(ACTIONS)
        ].set_index("action")["expected_win_probability"].to_dict()
        assert result["pre_snap"]["eligible_actions"]
        assert set(result["pre_snap"]["eligible_actions"]) == set(eligible)
        for action, p in eligible.items():
            assert result["pre_snap"]["q_wp_by_action"][action] == pytest.approx(p, abs=1e-12)
        assert result["pre_snap"]["policy_action"] in eligible
        assert result["audit"]["engine_blob_sha1"] == PINNED_ENGINE_BLOB
        assert result["audit"]["full_precision"] is False




def test_full_registered_q4_1200_trials_uses_native_patch_and_seed43(snapshot):
    """Full (not low-budget) fabricated Q4 path, exactly as pinned app."""
    engine = load_verified_engine(ROOT)
    sample = fabricated(
        game_id="synthetic_full_registered_q4",
        quarter=4, game_seconds_remaining=299.,
        yardline_100=6., ydstogo=4., goal_to_go=0.,
    )
    record = build_synthetic_policy_forecast(
        MANIFEST, sample, engine=engine, nuisance_snapshot=snapshot,
    )
    from src.streamlit_kickoff_preview import (
        isolated_engine_from_loaded_engine,
        preview_native_kickoff_feature_candidate,
    )
    direct = preview_native_kickoff_feature_candidate(
        engine_state(sample), simulations=1200, action_seed=43,
        isolated_engine=isolated_engine_from_loaded_engine(engine),
    )
    assert record["audit"]["full_precision"] is True
    assert record["audit"]["simulation_count"] == 1200
    assert record["audit"]["seed"] == 43
    assert record["audit"]["kickoff_input_mode"] == "Q4_NATIVE_STANDARD_35"
    actual = direct["results"]
    for action, score in record["pre_snap"]["q_wp_by_action"].items():
        q = float(actual.loc[
            actual["action"].eq(action), "expected_win_probability"
        ].iloc[0])
        assert score == pytest.approx(q, abs=1e-12)


def test_full_registered_q1_adaptive_pilot_budget_is_300(snapshot):
    engine = load_verified_engine(ROOT)
    sample = fabricated(
        game_id="synthetic_full_registered_q1",
        quarter=1, game_seconds_remaining=3100.,
        yardline_100=95., ydstogo=3., goal_to_go=0.,
    )
    result = build_synthetic_policy_forecast(
        MANIFEST, sample, engine=engine, nuisance_snapshot=snapshot,
    )
    meta = result["audit"]
    assert meta["pilot_simulations"] == 300
    assert meta["full_precision"] is True
    assert meta["seed"] in (42, 43)
    assert meta["simulation_count"] in (300, 1200)
    assert meta["adaptive_escalated"] == (meta["seed"] == 43)
    # The app reruns from scratch with seed43 if pilot action gap is close.
    pilot = engine.recommend(engine_state(sample), n_simulations=300, seed=42)
    from src.frozen_2027_policy_bridge import _action_gap_z
    z = _action_gap_z(pilot)
    expected_escalation = not np.isfinite(z) or z < 1.96
    assert expected_escalation == meta["adaptive_escalated"]
    assert meta["pilot_action_gap_z"] == pytest.approx(z) if np.isfinite(z) else (
        meta["pilot_action_gap_z"] is None
    )


def test_q4_cannot_silently_fall_back_to_unpatched_low_budget(snapshot):
    q4 = fabricated(quarter=4, game_seconds_remaining=299.)
    with pytest.raises(ValueError, match="Q4 native kickoff patch"):
        build_synthetic_policy_forecast(
            MANIFEST, q4, engine=SyntheticEngine(),
            nuisance_snapshot=snapshot, n_simulations=32,
        )


def test_bridge_never_downloads_or_reads_future_outcomes():
    src = (ROOT / "src/frozen_2027_policy_bridge.py").read_text()
    assert "nflreadpy" not in src
    assert "load_pbp" not in src
    assert "2026 Week 6" not in src
    assert "joblib.dump" not in src
    assert "policy_study.evaluate" not in src
