"""Connect nflreadpy-shaped pre-play reconstruction to the PINNED NFL policy.

This is a fully offline, FABRICATED-ONLY rehearsal. It accepts a supplied
pandas/Polars play-by-play frame and does not download NFL data. No true
2026/2027 outcomes, observed coaching choices or future final-score labels
may be admitted to the predictor.

The future *real* 2027 cohort remains fail-closed. Completion of the 2027
regular season, independent final-game checks, source-version hashes and a
separately reviewed one-time authorization must precede ANY future real-data
scoring code. This module never unlocks that evaluation.

The frozen bridge already owns the 300/1200 Q1-Q3 adaptive policy and the
Q4 1200-trial native kickoff correction. DO NOT implement those again here.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from typing import Any, Mapping

import pandas as pd

from src.fourth_down_policy_study import POLICY_COMMIT
from src.frozen_2027_policy_bridge import (
    APP_HIGH_SIMULATIONS, PINNED_ENGINE_BLOB, SNAPSHOT_KEYS,
    build_synthetic_policy_forecast, load_verified_engine,
    validate_synthetic_snapshot,
)
from src.historical_fourth_down_cohort import PRESNAP_COLUMNS
from src.historical_nuisance_portable import (
    assert_historical_source_integrity, load_historical_parameters,
)
from src.nflverse_2027_reconstruction import (
    MODE_SYNTHETIC, PREPLAY_OUTPUT_FIELDS, reconstruct_source_preplay,
)

MAX_FABRICATED_OPPORTUNITIES = 32
BRIDGE_MANIFEST = {
    "mode": "synthetic",
    "source_type": "fabricated_only",
    "policy_commit": POLICY_COMMIT,
}
FAKE_ASOF = "POSTGAME_RECONSTRUCTED_NOT_WITNESSED"
FINAL_2027_GATE_STATUS = "NOT_AUTHORIZED_OR_COMPLETED"

# Explicitly omit labels, result, coach action, classification, derived
# post-play state, and source context gap markers from the engine input.
ENGINE_CONTEXT_FIELDS = (
    "down", "offense_team", "home_team", "away_team", "roof",
)
NON_ASOF_FIELDS = {
    "play_type", "desc", "action", "observed_action", "result", "epa",
    "yards_gained", "touchdown", "first_down", "offense_win",
    "conversion", "field_goal_result", "final_home_score",
}


def _reject_real_holdout_before_access(mode: Any) -> None:
    # Check the selection before evaluating input type, columns, IDs,
    # downloading data, loading labels or importing any model.
    if mode != MODE_SYNTHETIC:
        raise PermissionError(
            "Real 2027 outcome/holdout evaluation is NOT authorized. "
            "Wait for the complete 2027 regular season, independently verify "
            "final games and provenance, and obtain a separately reviewed "
            "one-time season-admission gate. This adapter is fabricated-only."
        )


def _synthetic_engine_snapshot(row: Mapping[str, Any]) -> dict[str, Any]:
    """Translate one canonical nflverse *projection*, not the source row."""
    if set(row) != set(PREPLAY_OUTPUT_FIELDS):
        raise ValueError("Only the exact label-free nflverse preplay projection is allowed")
    if (row["season"] != 2027 or not isinstance(row["game_id"], str)
            or not row["game_id"].startswith("synthetic_")):
        raise ValueError("Only artificial 2027 game identifiers are allowed")
    if row["frozen_policy_commit"] != POLICY_COMMIT:
        raise ValueError("Source projection disagrees with frozen policy")
    if row["feature_asof"] != FAKE_ASOF:
        raise ValueError("Source projection must disclose postgame reconstruction")
    # Explicit refusal rather than guessing a Q2 receiver from the Q3 kickoff,
    # inventing roof status, or using "last play" state from future columns.
    if row["model_context_complete"] is not True:
        raise ValueError("Missing required model context; do not impute")
    if row["model_context_gaps"] != "":
        raise ValueError("A source row cannot clear documented context gaps")
    fields = {k: row[k] for k in PRESNAP_COLUMNS if k != "feature_asof"}
    fields.update({k: row[k] for k in ENGINE_CONTEXT_FIELDS})
    fields.update({
        "feature_asof": "pre_snap",  # synthetic fixture contract ONLY
        "second_half_receiver": None,  # Q2 always excluded for this source
        "source_type": "fabricated_only",
    })
    if set(fields) != SNAPSHOT_KEYS or NON_ASOF_FIELDS.intersection(fields):
        raise RuntimeError("Synthetic engine input whitelist/schema changed")
    validate_synthetic_snapshot(fields)  # goal line, clock, site, Q2, roof
    return fields


def _digest(obj: Any) -> str:
    data = json.dumps(
        obj, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(data).hexdigest()


def replay_fabricated_nflverse_through_frozen_policy(
    raw: Any, *, mode: str, engine: Any = None,
    nuisance_snapshot: Mapping[str, Any] | None = None,
    n_simulations: int = APP_HIGH_SIMULATIONS,
) -> dict[str, Any]:
    """Pre-play-like PBP -> frozen forecasts, with explicit non-scored skips.

    Injected engines are allowed ONLY for fabricated unit tests. The default
    loads the original verified registered policy and its unchanged
    historical 2014-25 coach/reward nuisance parameter set.

    Labels are not even projected. The generated "pre_snap" fields only meet
    the SYNTHETIC bridge schema; they do NOT indicate the predictions were
    calculated before actual NFL snaps.
    """
    _reject_real_holdout_before_access(mode)
    if (type(n_simulations) is not int
            or not 2 <= n_simulations <= APP_HIGH_SIMULATIONS):
        raise ValueError("Synthetic simulation budget must be 2..1200")
    reconstruction = reconstruct_source_preplay(raw, mode=MODE_SYNTHETIC)
    source_audit = reconstruction["audit"]
    if (source_audit["source_labels_accessed"] is not False
            or source_audit["real_2027_outcomes_accessed"] is not False
            or source_audit["frozen_policy_commit"] != POLICY_COMMIT):
        raise RuntimeError("Reconstruction must not access future outcomes")
    total = source_audit["source_fourth_down_candidates"]
    if total > MAX_FABRICATED_OPPORTUNITIES:
        raise ValueError("Synthetic batch is too large; real-season scoring is blocked")
    ready, skipped = [], [
        {"game_id": row.game_id, "play_id": row.play_id, "reason": row.reason}
        for row in reconstruction["excluded"].itertuples(index=False)
    ]
    for row in reconstruction["preplay_features"].to_dict(orient="records"):
        key = {k: row[k] for k in ("game_id", "play_id")}
        if not row["model_context_complete"]:
            skipped.append({
                **key, "reason": "MODEL_CONTEXT_MISSING",
                "detail": row["model_context_gaps"],
            })
            continue
        try:
            snapshot = _synthetic_engine_snapshot(row)
        except ValueError:
            skipped.append({**key, "reason": "FROZEN_POLICY_STATE_UNSUPPORTED"})
            continue
        ready.append(snapshot)
    if ready and n_simulations < 200 and any(s["quarter"] == 4 for s in ready):
        raise ValueError("Q4 native kickoff policy requires >=200 synthetic trials")
    using_registered_engine = engine is None
    if ready:
        # The snapshot has been validated before any heavy model operation.
        if nuisance_snapshot is None:
            nuisance_snapshot = load_historical_parameters()
        assert_historical_source_integrity(nuisance_snapshot)
        if engine is None:
            engine = load_verified_engine()
        else:
            # Injected mocks never receive real game data and results must not
            # be represented as official engine output.
            for name in ("recommend", "normalize_state", "action_eligible"):
                if not callable(getattr(engine, name, None)):
                    raise ValueError("Injected synthetic engine lacks required interface")
    predictions, replay_audit = [], []
    for snapshot in ready:
        result = build_synthetic_policy_forecast(
            BRIDGE_MANIFEST, snapshot, engine=engine,
            nuisance_snapshot=nuisance_snapshot,
            n_simulations=n_simulations,
        )
        forecast = result["pre_snap"]
        if (forecast["game_id"] != snapshot["game_id"]
                or forecast["play_id"] != snapshot["play_id"]
                or forecast["feature_asof"] != "pre_snap"
                or result["audit"]["observed_action_access"] is not False
                or result["audit"]["outcome_access"] is not False):
            raise RuntimeError("Frozen policy forecast leaked labels or mismatched ID")
        predictions.append(forecast)
        replay_audit.append({
            "game_id": snapshot["game_id"],
            "play_id": snapshot["play_id"],
            "simulation_count": result["audit"]["simulation_count"],
            "final_seed": result["audit"]["seed"],
            "kickoff_input_mode": result["audit"]["kickoff_input_mode"],
            "full_precision": result["audit"]["full_precision"],
        })
    if len(predictions) + len(skipped) != total:
        raise RuntimeError("Source denominator lost decision opportunities")
    pred_keys = [(r["game_id"], r["play_id"]) for r in predictions]
    skip_keys = [(r["game_id"], r["play_id"]) for r in skipped]
    if len(pred_keys + skip_keys) != len(set(pred_keys + skip_keys)):
        raise RuntimeError("Duplicate game/play between forecasts and skipped rows")
    return {
        "manifest": {
            **BRIDGE_MANIFEST,
            "reconstruction_asof": FAKE_ASOF,
            "real_2027_holdout_gate": FINAL_2027_GATE_STATUS,
            "engine_used": (
                "NOT_RUN" if not predictions else
                "PINNED_REGISTERED_ENGINE" if using_registered_engine else
                "TEST_INJECTED_SYNTHETIC_ENGINE"
            ),
            "policy_engine_blob_sha1": PINNED_ENGINE_BLOB,
            "full_precision_requested": n_simulations == APP_HIGH_SIMULATIONS,
            "actual_pre_snap_forecasts_observed": False,
        },
        "synthetic_forecasts": predictions,
        "skipped_opportunities": skipped,
        "per_forecast_engine_audit": replay_audit,
        "audit": {
            "source_fourth_down_candidates": total,
            "source_exclusions": source_audit["excluded_states"],
            "missing_model_context_counts": source_audit["missing_model_context_counts"],
            "reconstructed_core_states": source_audit["reconstructed_core_states"],
            "forecasted_fabricated_opportunities": len(predictions),
            "unscored_opportunities": len(skipped),
            "skip_reason_counts": dict(sorted(Counter(
                r["reason"] for r in skipped
            ).items())),
            "source_labels_accessed": False,
            "observed_actions_accessed": False,
            "final_game_results_accessed": False,
            "2026_reserved_data_accessed": False,
            "real_2027_outcomes_accessed": False,
            "future_season_scoring_authorized": False,
            "independent_pre_snap_provenance": False,
            "calibration_evaluated": False,
            "synthetic_prediction_sha256": _digest(predictions),
            "interpretation": (
                "FABRICATED ONLY: policy forecasts computed using "
                "retrospectively projected play-by-play shapes. NOT "
                "live pre-snap evidence, real 2027 validation or coaching benefit."
            ),
        },
    }


def main() -> None:
    """One deliberately invented nflreadpy-shaped play; NO network access."""
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fabricated-demo-only", action="store_true")
    args = parser.parse_args()
    if not args.fabricated_demo_only:
        parser.error("Only --fabricated-demo-only is permitted")
    from src.nflverse_2027_reconstruction import PREPLAY_SOURCE_FIELDS
    fake = {
        "season": 2027, "season_type": "REG", "week": 2,
        "game_id": "synthetic_2027_frozen_policy_demo", "play_id": 101.,
        "down": 4., "qtr": 4., "game_seconds_remaining": 299.,
        "yardline_100": 6., "ydstogo": 4., "goal_to_go": 0.,
        "score_differential": -4., "posteam_timeouts_remaining": 2.,
        "defteam_timeouts_remaining": 3., "posteam": "ABC",
        "home_team": "ABC", "away_team": "DEF",
        "location": "Home", "roof": "outdoors",
    }
    if set(fake) != set(PREPLAY_SOURCE_FIELDS):
        raise RuntimeError("Demo input no longer matches source features")
    result = replay_fabricated_nflverse_through_frozen_policy(
        pd.DataFrame([fake]), mode=MODE_SYNTHETIC,
    )
    print(json.dumps({
        "manifest": result["manifest"],
        "audit": result["audit"],
        "synthetic_action": result["synthetic_forecasts"][0]["policy_action"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
