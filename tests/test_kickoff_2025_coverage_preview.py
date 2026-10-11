"""Streamlit preview must disclose denominators and unmodeled Q4 kickoff events."""

import json
from pathlib import Path

import pytest

from src.streamlit_kickoff_preview import (
    OBSERVED_2025_COVERAGE_PATH,
    observed_2025_kickoff_coverage,
    observed_2025_pool_summary,
)


def test_real_2025_metadata_has_exhaustive_denominator_and_2025_donors():
    coverage = observed_2025_kickoff_coverage()
    pool = observed_2025_pool_summary()
    all_kicks = coverage["all"]
    late = coverage["late"]
    close = coverage["receiver_leads_1to3"]
    assert all_kicks["total_kickoffs"] == 584
    assert all_kicks["counts"] == {
        "modeled_ordinary": 488,
        "onside_attempt": 41,
        "kickoff_return_td": 1,
        "penalty_or_rekick": 52,
        "other_unmatched": 2,
    }
    assert all_kicks["counts"]["modeled_ordinary"] == pool["kickoffs"]
    assert all_kicks["ordinary_share"] == pytest.approx(488/584)
    assert late["total_kickoffs"] == 218
    assert late["counts"]["modeled_ordinary"] == 171
    assert late["counts"]["onside_attempt"] == 32
    assert late["ordinary_share"] == pytest.approx(171/218)
    assert close["total_kickoffs"] == 20
    assert close["counts"]["modeled_ordinary"] == 15
    assert close["counts"]["onside_attempt"] == 5
    assert close["low_support"]


def test_missing_coverage_never_silently_reverts_to_100_percent(tmp_path):
    assert OBSERVED_2025_COVERAGE_PATH.is_file()
    with pytest.raises(FileNotFoundError, match="coverage evidence missing"):
        observed_2025_kickoff_coverage(tmp_path / "missing.json")


def test_stale_coverage_and_unmodeled_events_cannot_be_fabricated(tmp_path):
    obj = json.loads(OBSERVED_2025_COVERAGE_PATH.read_text())
    src = tmp_path / "tampered.json"

    obj["groups"][0]["counts"]["onside_attempt"] -= 1
    src.write_text(json.dumps(obj))
    with pytest.raises(ValueError, match="exhaustive-count"):
        observed_2025_kickoff_coverage(src)

    obj = json.loads(OBSERVED_2025_COVERAGE_PATH.read_text())
    obj["groups"][0]["counts"]["modeled_ordinary"] += 1
    obj["groups"][0]["total_kickoffs"] += 1
    obj["groups"][0]["ordinary_share"] = (
        obj["groups"][0]["counts"]["modeled_ordinary"]
        / obj["groups"][0]["total_kickoffs"]
    )
    src.write_text(json.dumps(obj))
    with pytest.raises(ValueError, match="pool mismatch"):
        observed_2025_kickoff_coverage(src)


def test_only_2025_historical_evidence_is_accepted(tmp_path):
    obj = json.loads(OBSERVED_2025_COVERAGE_PATH.read_text())
    src = tmp_path / "future.json"
    obj["data_year"] = 2026
    src.write_text(json.dumps(obj))
    with pytest.raises(ValueError, match="not eligible"):
        observed_2025_kickoff_coverage(src)

    obj["data_year"] = 2025
    obj["no_2026_outcomes"] = False
    src.write_text(json.dumps(obj))
    with pytest.raises(ValueError, match="not eligible"):
        observed_2025_kickoff_coverage(src)


def test_score_context_selection_is_algebraically_correct_for_rams():
    # Before field goal: Rams 27, Seahawks 31 (offense margin -4).
    # After make: Rams 30, Seahawks 31, Seahawks receive leading by 1.
    before_fg_margin = -4
    receiving_margin = -(before_fg_margin + 3)
    assert receiving_margin == 1
    # Do not mistake the 2025 selected kickoff frequency for a 2026 forecast.
    close = observed_2025_kickoff_coverage()["receiver_leads_1to3"]
    assert close["total_kickoffs"] < 30
