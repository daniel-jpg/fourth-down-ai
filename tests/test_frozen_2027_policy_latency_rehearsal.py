"""Full-budget registered-policy compute timing only, synthetic 2027-shaped states."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.fourth_down_policy_study import POLICY_COMMIT
from src.frozen_2027_policy_latency_rehearsal import (
    CASES, fabricated_state, measure_fabricated_policy,
)
from src.frozen_2027_policy_bridge import validate_synthetic_snapshot


@pytest.fixture(scope="module")
def result():
    return measure_fabricated_policy(repeats=1)


def test_all_fixed_benchmark_states_are_safe_fabricated_fourth_down_only():
    assert len(CASES)==4
    for scenario in CASES:
        state=fabricated_state(**scenario["overrides"])
        validate_synthetic_snapshot(state)
        assert state["season"]==2027
        assert state["source_type"]=="fabricated_only"
        assert state["feature_asof"]=="pre_snap"
        assert state["down"]==4
        assert state["quarter"] in (1,2,3,4)


def test_exact_registered_policy_budget_and_independent_native_q4(result):
    assert result["registered_policy_commit"]==POLICY_COMMIT
    assert result["status"]=="FABRICATED_COMPUTE_LATENCY_ONLY_NO_REAL_SNAP_DEADLINE"
    assert result["full_registered_precision"] is True
    assert result["startup_and_model_load_ms"] >= 0
    assert len(result["cases"])==4
    for row in result["cases"]:
        assert row["median_ms"]>=0
        assert row["max_ms"]>=row["median_ms"]
        assert row["seed"] in (42,43)
        assert row["simulation_count"] in (300,1200)
        assert row["normal_action_count"]>0
        if row["quarter"]==4:
            assert row["simulation_count"]==1200
            assert row["seed"]==43
            assert row["kickoff_input_mode"]=="Q4_NATIVE_STANDARD_35"
        else:
            assert row["kickoff_input_mode"]=="NO_Q4_KICKOFF_PATCH"


def test_latency_report_does_not_claim_real_nfl_or_external_timeliness(result):
    assert result["source_type"]=="fabricated_only"
    for name in (
        "includes_network_latency",
        "includes_vendor_created_to_collector_receipt_latency",
        "includes_external_witness_receipt_latency",
        "includes_trusted_real_snap_timestamps",
        "establishes_2027_prospective_ready",
    ):
        assert result[name] is False
    assert "not p95 live end-to-end" in result["interpretation"]
    assert len(json.dumps(result))>200


@pytest.mark.parametrize("bad",[-1,0,6,1.2,False,"3"])
def test_cannot_silently_run_invalid_latency_sample_budget(bad):
    with pytest.raises(ValueError,match="repeats"):
        measure_fabricated_policy(repeats=bad)


def test_benchmark_has_no_reservation_or_live_nfl_fetch():
    source=Path("src/frozen_2027_policy_latency_rehearsal.py").read_text()
    for forbidden in (
        "requests.get(", "urlopen(", "nflreadpy.load",
        "2026_native_kickoff", "Step 66", "Step 83",
        "2026" + "_scores",
    ):
        assert forbidden not in source
    assert "fabricated_only" in source
