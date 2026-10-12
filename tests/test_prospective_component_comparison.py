"""Only fabricated inputs for the sealed future component audit.

The production model artifacts and reserved 2026 validation data are never
read. Synthetic success is software QA, not independent NFL evidence.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from src.prospective_component_comparison import (
    BOOTSTRAP_SEED, GO_KEYS, TRAINING_END, WP_KEYS,
    evaluate, forecast_sha256,
)
from src.fourth_down_policy_study import POLICY_COMMIT

ROOT = Path(__file__).resolve().parents[1]


def input_fixture():
    base = {
        "season": 2027, "quarter": 4, "site": "HOME",
        "yardline_100": 95., "ydstogo": 3.,
        "game_seconds_remaining": 299.,
        "offense_score_margin": 4., "feature_asof": "pre_snap",
    }
    wp = []
    go = []
    wp_final = []
    go_final = []
    for i in range(3):
        k = {"game_id": f"synthetic_game_{i}", "play_id": "88"}
        wp.append({
            **base, **k, "site": "HOME" if i != 1 else "AWAY",
            "frozen_wp": .38 + i*.1, "candidate_wp": .40 + i*.1,
        })
        go.append({
            **base, **k, "observed_go_action":
                "NORMAL_GO_PASS" if i != 1 else "NORMAL_GO_RUN",
            "frozen_conversion": .5, "candidate_conversion": .6,
        })
        wp_final.append({**k, "offense_win": int(i == 2)})
        go_final.append({**k, "converted": int(i > 0)})
    return {
        "manifest": {
            "mode": "synthetic",
            "source_type": "fabricated_only",
            "holdout_season": 2027,
            "training_end_season": 2022,
            "feature_policy": "pre_snap_only",
            "decision_policy_commit": POLICY_COMMIT,
        },
        "predictions": {"wp": wp, "go": go},
        "outcomes": {"wp": wp_final, "go": go_final},
    }


def test_schema_and_separate_factual_outcomes_score_without_promotion():
    obj = input_fixture()
    r = evaluate(obj["predictions"], obj["outcomes"], obj["manifest"], draws=60)
    assert r["mode"] == "synthetic"
    assert r["n_wp_states"] == 3
    assert r["n_go_attempts"] == 3
    assert len(r["forecasts_sha256"]) == 64
    assert r["wp_calibration"]["Q4"]["states"] == 3
    assert r["wp_calibration"]["Q4_FINAL_FIVE_OWN10_LEADING4"]["games"] == 3
    assert r["go_conversion_calibration"]["OWN10_FOURTH_2_OR_3"]["states"] == 3
    assert r["wp_calibration"]["AWAY"]["status"] == "SPARSE_DESCRIPTIVE_ONLY"
    assert r["model_review_gate"]["passes_prespecified_component_coverage_and_gain"] is False
    assert r["model_review_gate"]["automatic_promotion_allowed"] is False
    assert r["model_review_gate"]["causal_coaching_claim_allowed"] is False


def test_forecast_seal_excludes_labels_and_is_stable():
    obj = input_fixture()
    first = forecast_sha256(obj["predictions"])
    second = forecast_sha256(json.loads(json.dumps(obj["predictions"])))
    assert first == second
    obj["outcomes"]["go"][0]["converted"] = 1
    assert forecast_sha256(obj["predictions"]) == first


@pytest.mark.parametrize("field", [
    "offense_win", "converted", "winner", "home_score_final",
    "post_snap_epa", "yards_gained", "result",
])
def test_post_snap_outcome_field_cannot_enter_predictions(field):
    obj = input_fixture()
    obj["predictions"]["wp"][0][field] = 1
    with pytest.raises(ValueError, match="EXACT approved keys"):
        evaluate(obj["predictions"], obj["outcomes"], obj["manifest"], draws=20)


def test_future_or_protected_season_never_accepted_in_fixture():
    obj = input_fixture()
    obj["predictions"]["go"][0]["season"] = 2026
    with pytest.raises(ValueError, match="season"):
        evaluate(obj["predictions"], obj["outcomes"], obj["manifest"], draws=20)


@pytest.mark.parametrize("part,where,what", [
    ("predictions", "wp", "go"),
    ("outcomes", "wp", "go"),
    ("predictions", "go", "go"),
    ("outcomes", "go", "go"),
])
def test_unmatched_keys_fail_closed(part, where, what):
    obj = input_fixture()
    obj[part][where][0]["play_id"] = "unmatched"
    with pytest.raises(ValueError, match="Unmatched"):
        evaluate(obj["predictions"], obj["outcomes"], obj["manifest"], draws=20)


def test_same_game_play_duplicate_fails_closed_not_double_counted():
    obj = input_fixture()
    obj["predictions"]["wp"].append(deepcopy(obj["predictions"]["wp"][0]))
    with pytest.raises(ValueError, match="Duplicate"):
        evaluate(obj["predictions"], obj["outcomes"], obj["manifest"], draws=20)


@pytest.mark.parametrize("field,value", [
    ("training_end_season", 2026),
    ("holdout_season", 2026),
    ("decision_policy_commit", "HEAD"),
    ("feature_policy", "post_game"),
])
def test_manifest_pinning_is_mandatory(field, value):
    obj = input_fixture()
    obj["manifest"][field] = value
    with pytest.raises(ValueError, match="manifest mismatch"):
        evaluate(obj["predictions"], obj["outcomes"], obj["manifest"], draws=20)


def test_unsealed_or_early_prospective_requests_rejected():
    obj = input_fixture()
    obj["manifest"]["mode"] = "prospective"
    with pytest.raises(ValueError, match="completed 2027"):
        evaluate(obj["predictions"], obj["outcomes"], obj["manifest"], draws=20)
    obj["manifest"]["final_2027_regular_season_complete"] = True
    with pytest.raises(ValueError, match="Unreviewed prospective sources"):
        evaluate(obj["predictions"], obj["outcomes"], obj["manifest"], draws=20)
    obj["manifest"]["source_type"] = "independently_verified_sealed"
    with pytest.raises(ValueError, match="sealed before"):
        evaluate(obj["predictions"], obj["outcomes"], obj["manifest"], draws=20)
    obj["manifest"]["sealed_before_outcomes"] = True
    with pytest.raises(ValueError, match="Missing independently pinned"):
        evaluate(obj["predictions"], obj["outcomes"], obj["manifest"], draws=20)


def test_prospective_provenance_hash_match_and_size_gates():
    obj = input_fixture()
    man = obj["manifest"]
    man.update(mode="prospective",
               source_type="independently_verified_sealed",
               final_2027_regular_season_complete=True,
               sealed_before_outcomes=True,
               go_candidate_artifact_sha256="a"*64,
               wp_candidate_artifact_sha256="b"*64,
               extractor_source_sha256="c"*64,
               prediction_archive_sha256="d"*64)
    with pytest.raises(ValueError, match="SHA-256 does not match"):
        evaluate(obj["predictions"], obj["outcomes"], man, draws=20)
    man["prediction_archive_sha256"] = forecast_sha256(obj["predictions"])
    with pytest.raises(ValueError, match="100 distinct games"):
        evaluate(obj["predictions"], obj["outcomes"], man, draws=20)


def test_invalid_probability_rejected_not_silently_clamped():
    obj = input_fixture()
    obj["predictions"]["wp"][0]["candidate_wp"] = float("nan")
    with pytest.raises(ValueError, match="candidate_wp"):
        evaluate(obj["predictions"], obj["outcomes"], obj["manifest"], draws=20)
    obj = input_fixture()
    obj["predictions"]["go"][1]["candidate_conversion"] = 1.1
    with pytest.raises(ValueError, match="candidate_conversion"):
        evaluate(obj["predictions"], obj["outcomes"], obj["manifest"], draws=20)


def test_synthetic_cli_accepts_separate_prediction_and_label_sources(tmp_path):
    source = tmp_path / "fake_components.json"
    destination = tmp_path / "fake_audit.json"
    source.write_text(json.dumps(input_fixture()))
    subprocess.run(
        [sys.executable, "-m", "src.prospective_component_comparison",
         str(source), "--output", str(destination)],
        cwd=ROOT, check=True,
    )
    result = json.loads(destination.read_text())
    assert result["mode"] == "synthetic"
    assert "Synthetic results are fabricated" in result["warning"]


def test_component_evaluator_has_no_retrieval_or_training_side_effects():
    source = (ROOT / "src/prospective_component_comparison.py").read_text()
    assert "import nflreadpy" not in source
    assert "joblib.dump" not in source
    assert "requests.get" not in source
    assert "git push" not in source
    assert "forward_2026_validation_once" not in source
    assert WP_KEYS and GO_KEYS
    assert TRAINING_END == 2022 and BOOTSTRAP_SEED == 2027
