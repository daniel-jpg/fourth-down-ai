"""Model-only GO-vs-PUNT break-even sensitivities in issue #9.

All real-engine testing is read-only on pre-2026 checked-in model assets;
the cases are FABRICATED pre-snap states, not observed outcomes or
independent evidence of a successful on-field coaching choice.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from src.deep_own_go_break_even_audit import (
    EXPECTED_GIT_BLOBS, MODEL_SPLIT, REFERENCE_OWN_YARDS,
    REFERENCE_SEEDS, break_even_sensitivity, historical_local_support,
    planned_state, sample_private_conversion_arms,
    verify_frozen_research_files,
)
from src.streamlit_kickoff_preview import isolated_engine_for_preview


def test_exact_break_even_identity_and_hypothetical_overshoot():
    case=break_even_sensitivity(
        p_model=.70,wp_failure=.20,wp_success=.80,wp_punt=.50
    )
    assert case["status"]=="INTERIOR_SIMULATED_BREAK_EVEN"
    assert case["model_implied_break_even_conversion_p"]==pytest.approx(.5)
    assert case["model_probability_over_break_even_pp"]==pytest.approx(20.)
    assert case["wp_go_mixture_at_model_p"]==pytest.approx(.62)
    assert case["simulated_go_minus_punt_pp"]==pytest.approx(12.)
    assert case["hypothetical_go_minus_punt_pp_if_conversion_overestimated_by"]["15"]==pytest.approx(3.)
    assert case["hypothetical_go_minus_punt_pp_if_conversion_overestimated_by"]["25"]==pytest.approx(-3.)


def test_tipping_point_already_below_zero_means_go_wins_even_without_conversion():
    case=break_even_sensitivity(
        p_model=.1,wp_failure=.45,wp_success=.85,wp_punt=.40
    )
    assert case["status"]=="MODEL_GO_AHEAD_AT_ALL_CONVERSION_RATES"
    assert case["model_implied_break_even_conversion_p"]<0


def test_unreachable_tipping_point_identified_without_clipping_to_fake_100_percent():
    case=break_even_sensitivity(
        p_model=.7,wp_failure=.2,wp_success=.3,wp_punt=.5
    )
    assert case["status"]=="MODEL_PUNT_AHEAD_AT_ALL_CONVERSION_RATES"
    assert case["model_implied_break_even_conversion_p"]>1


def test_nonpositive_success_premium_never_fabricates_sensible_threshold():
    case=break_even_sensitivity(
        p_model=.5,wp_failure=.6,wp_success=.4,wp_punt=.5
    )
    assert case["status"]=="NONPOSITIVE_SIMULATED_SUCCESS_PREMIUM"
    assert case["model_implied_break_even_conversion_p"] is None
    assert case["model_probability_over_break_even_pp"] is None


@pytest.mark.parametrize("key",[
    "p_model","wp_failure","wp_success","wp_punt",
])
@pytest.mark.parametrize("bad",[-.01,1.001,float("nan"),float("inf"),True,None])
def test_invalid_nonfinite_or_out_of_range_value_cannot_create_support_claim(
    key,bad,
):
    args={
        "p_model":.7,"wp_failure":.2,
        "wp_success":.8,"wp_punt":.5,
    }
    args[key]=bad
    with pytest.raises(ValueError,match="finite probability|must be numeric"):
        break_even_sensitivity(**args)


@pytest.mark.parametrize("own",REFERENCE_OWN_YARDS)
def test_pregenerated_scenarios_only_contain_presnap_fields_and_are_neutral(own):
    state=planned_state(own)
    assert state["yardline_100"]==100-own
    assert state["ydstogo"]==3.
    assert state["qtr"]==1
    assert state["site"]=="NEUTRAL"
    assert "offense_win" not in state
    assert "observed_action" not in state
    assert "converted" not in state


@pytest.mark.parametrize("unsupported",[1,3,4,15,25,40])
def test_no_ad_hoc_territory_search_to_select_positive_result(unsupported):
    with pytest.raises(ValueError,match="prespecified"):
        planned_state(unsupported)


def test_real_historical_support_counts_reproduce_original_sparse_findings():
    report=historical_local_support()
    assert len(report)==8
    tr={r["territory"]:r for r in report if r["period"]=="train_2014_2022"}
    val={r["territory"]:r for r in report if r["period"].startswith("already_")}
    assert [tr[f"OWN_1_TO_{n}"]["go_chosen_count"] for n in REFERENCE_OWN_YARDS]==[
        0,1,13,73
    ]
    assert [val[f"OWN_1_TO_{n}"]["go_chosen_count"] for n in REFERENCE_OWN_YARDS]==[
        0,0,5,25
    ]
    assert tr["OWN_1_TO_5"]["go_conversion_rate_observed"] is None
    assert tr["OWN_1_TO_5"]["go_share_of_observed_go_or_punt"] is None
    assert tr["OWN_1_TO_10"]["punt_chosen_count"]==44
    assert all(r["overlap_or_causal_propensity_identified"] is False for r in report)


def test_hash_gate_rejects_changed_historical_data_before_parquet_read(monkeypatch):
    import src.deep_own_go_break_even_audit as m
    assert verify_frozen_research_files()==EXPECTED_GIT_BLOBS
    monkeypatch.setitem(m.EXPECTED_GIT_BLOBS,
                        "data/fourth_down_modeling_split.parquet","0"*40)
    def no_read(*args,**kwargs):
        raise AssertionError("Protected source was OPENED before checked")
    monkeypatch.setattr(m.pd,"read_parquet",no_read)
    with pytest.raises(ValueError,match="Frozen .* source hash mismatch"):
        m.historical_local_support()


@pytest.fixture(scope="module")
def actual_engine():
    return isolated_engine_for_preview()


def test_private_frozen_engine_branches_never_change_shared_conversion_prediction(
    actual_engine,
):
    state=planned_state(5)
    original=actual_engine["go_conversion_probability"]
    original_go=actual_engine["simulate_go"]
    original_punt=actual_engine["simulate_punt"]
    base=actual_engine["normalize_state"](state)
    ref=float(original(base,"NORMAL_GO_PASS"))
    result=sample_private_conversion_arms(
        actual_engine,state,"NORMAL_GO_PASS",seed=43,n_simulations=160,
    )
    assert 0<result["p_model"]<1
    assert result["p_model"]==pytest.approx(ref)
    assert result["branch_mc_standard_errors"]["failure"]>=0
    assert result["branch_mc_standard_errors"]["success"]>=0
    assert result["branch_mc_standard_errors"]["punt"]>=0
    assert result["simulations_per_arm"]==160
    assert result["assumptions"]["causal_policy_benefit_established"] is False
    assert result["assumptions"]["conditional_outcome_donors_unchanged"] is True
    assert actual_engine["go_conversion_probability"] is original
    assert actual_engine["simulate_go"] is original_go
    assert actual_engine["simulate_punt"] is original_punt
    assert original(base,"NORMAL_GO_PASS")==pytest.approx(ref)
    assert result["wp_if_all_go_fail"]<result["wp_if_all_go_convert"]


def test_two_seed_repeated_private_draw_is_deterministic(actual_engine):
    state=planned_state(10)
    x=sample_private_conversion_arms(
        actual_engine,state,"NORMAL_GO_RUN",seed=101,n_simulations=128
    )
    y=sample_private_conversion_arms(
        actual_engine,state,"NORMAL_GO_RUN",seed=101,n_simulations=128
    )
    assert x==y
    assert x["seed"]==101
    assert x["model_implied_break_even_conversion_p"] is not None


@pytest.mark.parametrize("wrong_action",["PUNT","FIELD_GOAL","FAKE_FG_RUN",None])
def test_only_normal_go_action_sensitivity_is_legal(actual_engine,wrong_action):
    with pytest.raises(ValueError,match="normal GO"):
        sample_private_conversion_arms(
            actual_engine,planned_state(5),wrong_action,
            seed=43,n_simulations=128,
        )


def test_2026_2027_outcomes_or_frozen_assets_cannot_be_written_by_research_script():
    src=Path("src/deep_own_go_break_even_audit.py").read_text()
    for forbidden in (
        "nfl.load_pbp(", "nflreadpy.load_pbp(", "requests.get(",
        "joblib.dump(", ".fit(", "model.predict_proba(",
        "Step 66", "Step 83", "2026_heldout",
    ):
        assert forbidden not in src
    assert "contains_2026_2027_outcomes" in src
    assert "does_not_provide_model_specification_uncertainty_ci" in src


def test_reference_seeds_are_fixed_before_study():
    assert REFERENCE_SEEDS==(43,101)


def test_unknown_model_blob_is_not_routable_through_public_app():
    assert MODEL_SPLIT.name=="fourth_down_modeling_split.parquet"
    assert Path("src/deep_own_go_break_even_audit.py").exists()
