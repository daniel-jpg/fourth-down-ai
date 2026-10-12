"""Retained plain-JSON historical nuisance parameter snapshot, NOT production."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.historical_fourth_down_cohort import PRESNAP_COLUMNS
from src.historical_nuisance_portable import (
    PARAMETER_FILE, assert_historical_source_integrity,
    load_historical_parameters, predict_portable_policy,
)

ROOT = Path(__file__).resolve().parents[1]


def example():
    states = [
        (4, 299, 6, 4, 0, -4, 3, 3, -1, "AWAY"),
        (1, 3100, 95, 3, 0, 0, 3, 3, 1, "HOME"),
        (4, 120, 30, 2, 0, -3, 2, 1, 0, "NEUTRAL"),
        (2, 2000, 50, 7, 0, 7, 1, 2, -1, "AWAY"),
    ]
    rows = []
    for i, (q, sec, yard, togo, goal, margin, own_to, other_to, adv, site) in enumerate(states):
        rows.append({
            "game_id": f"synthetic-2027-{i}", "play_id": str(i+1),
            "season": 2027, "week": i+1, "quarter": q,
            "game_seconds_remaining": sec, "yardline_100": yard,
            "ydstogo": togo, "goal_to_go": goal,
            "score_differential": margin,
            "posteam_timeouts_remaining": own_to,
            "defteam_timeouts_remaining": other_to,
            "site_advantage": adv, "site": site,
            "feature_asof": "pre_snap",
        })
    return pd.DataFrame(rows, columns=PRESNAP_COLUMNS)


def test_portable_snapshot_matches_saved_2025_sklearn_artifact_reference_cases():
    # These numeric references were computed directly from CI's separately
    # archived sklearn-1.9.1 joblib models; no pickle needed in ordinary CI.
    snap = load_historical_parameters()
    assert_historical_source_integrity(snap)
    calls = ["NORMAL_GO_PASS", "NORMAL_GO_RUN", "FIELD_GOAL", "PUNT"]
    out = predict_portable_policy(snap, example(), calls)
    assert out["policy_action"].tolist() == calls
    assert np.allclose(
        out["propensity_policy"].to_numpy(float),
        [0.09131282651103, 0.00004002802656, 0.18931658887984, 0.82211644548696],
        rtol=1e-8, atol=1e-10,
    )
    assert np.allclose(
        out["mu_policy"].to_numpy(float),
        [0.29696629552821, 0.4989705353501, 0.50058857848551, 0.75988644894571],
        rtol=1e-8, atol=1e-10,
    )
    assert np.allclose(
        out[[f"coach_probability_{a}" for a in snap["hypothetical_action_order"]]]
        .sum(axis=1).to_numpy(float),
        1., atol=1e-12,
    )
    assert "offense_win" not in out
    assert "observed_action" not in out


def test_portable_outputs_do_not_depend_on_future_coach_choices_or_labels():
    snapshot = load_historical_parameters()
    f = example()
    a = predict_portable_policy(snapshot, f, ["PUNT"]*len(f))
    extended = f.copy(deep=True)
    extended["offense_win"] = 1
    extended["observed_action"] = "NORMAL_GO_PASS"
    with pytest.raises(ValueError, match="EXACT feature-only schema"):
        predict_portable_policy(snapshot, extended, ["PUNT"]*len(f))
    b = predict_portable_policy(snapshot, f, ["PUNT"]*len(f))
    pd.testing.assert_frame_equal(a, b)


def test_bad_coefficients_or_unapproved_status_cannot_be_scored():
    s = load_historical_parameters()
    wrong = deepcopy(s)
    wrong["status"] = "PRODUCTION"
    with pytest.raises(ValueError, match="Incompatible"):
        predict_portable_policy(wrong, example(), ["PUNT"]*4)
    wrong = deepcopy(s)
    wrong["models"]["propensity"]["scaler_scale"][0] = 0
    with pytest.raises(ValueError, match="Corrupt"):
        predict_portable_policy(wrong, example(), ["PUNT"]*4)


def test_pinned_source_sha256_and_model_hash_metadata_survive_actions_expiry():
    s = load_historical_parameters()
    assert s["schema_version"] == 1
    assert len(s["source_sha256"]) == 64
    assert len(s["artifact_sha256"]["propensity"]) == 64
    assert len(s["artifact_sha256"]["conditional_win_reward"]) == 64
    assert (ROOT / "data/fourth_downs_labeled.parquet").exists()
    assert PARAMETER_FILE.is_file()
    assert all(len(s[key]) == 64 for key in (
        "extractor_sha256", "trainer_sha256", "pre_snap_sha256"
    ))
