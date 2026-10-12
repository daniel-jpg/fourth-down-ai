"""One-shot POSTSEASON evaluation rehearsal with irreversible synthetic seal.

The only supported input is fabricated 2027-shaped score and play records.
This module never calls nflreadpy, ESPN, the network, model trainers, or
protected 2026/real-2027 holdout experiments. It cannot be unlocked by
passing final_2027_regular_season_complete=True.

Stage 1: validate and EXCLUSIVELY WRITE a forecast commitment, without
reading observed coach actions or any final scores.
Stage 2: validate a *fabricated* full 272-game ESPN-shaped reference,
then join separate observed actions and terminal winners, evaluate ONCE,
and exclusively write outcome report and terminal completion marker.

An existing output directory (including a partially failed attempt) always
refuses a rerun. This is a local filesystem rehearsal ONLY; ephemeral CI
runners cannot substitute for a durable, human-reviewed future release lock.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from src.fourth_down_cohort_builder import (
    PRE_SNAP_KEYS, REQUIRED_MANIFEST, build_synthetic_cohort,
)
from src.fourth_down_policy_study import POLICY_COMMIT, evaluate
from src.postseason_2027_admission import (
    MODE, YEAR, _canonical_hash, refuse_future_data,
    verify_fabricated_complete_season,
)

TOP_LEVEL = frozenset({
    "forecast_bundle", "observed_plays", "pbp_games", "reference_games",
})
BUNDLE_KEYS = frozenset({
    "manifest", "synthetic_forecasts", "skipped_opportunities",
    "per_forecast_engine_audit", "audit",
})
FAKE_ASOF = "POSTGAME_RECONSTRUCTED_NOT_WITNESSED"
NO_REAL_STUDY = "REAL_2027_STUDY_NOT_AUTHORIZED"
ACCEPTED_ENGINES = {
    "PINNED_REGISTERED_ENGINE",
    "TEST_INJECTED_SYNTHETIC_ENGINE",
    "SYNTHETIC_PRECOMPUTED_FIXTURE_NOT_REGISTERED_MODEL",
}
FALSE_AUDIT_FLAGS = (
    "source_labels_accessed", "observed_actions_accessed",
    "final_game_results_accessed", "2026_reserved_data_accessed",
    "real_2027_outcomes_accessed", "future_season_scoring_authorized",
    "independent_pre_snap_provenance", "calibration_evaluated",
)


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(
        value, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False,
    ) + "\n").encode("ascii")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _keys(rows: list, label: str) -> list[tuple[str, str]]:
    result = []
    for item in rows:
        if not isinstance(item, dict):
            raise ValueError(f"Malformed {label} row")
        game, play = item.get("game_id"), item.get("play_id")
        if (not isinstance(game, str) or not game.startswith("synthetic_")
                or not isinstance(play, str) or not play):
            raise ValueError(f"Only uniquely keyed fabricated {label} rows allowed")
        result.append((game, play))
    if len(set(result)) != len(result):
        raise ValueError(f"Duplicate {label} game/play keys")
    return result


def validate_label_free_forecasts(bundle: Any) -> dict:
    if not isinstance(bundle, dict) or set(bundle) != BUNDLE_KEYS:
        raise ValueError("A complete registered synthetic forecast bundle is required")
    manifest = bundle["manifest"]
    if (not isinstance(manifest, dict)
            or manifest.get("mode") != "synthetic"
            or manifest.get("source_type") != "fabricated_only"
            or manifest.get("policy_commit") != POLICY_COMMIT
            or manifest.get("reconstruction_asof") != FAKE_ASOF
            or manifest.get("real_2027_holdout_gate") != "NOT_AUTHORIZED_OR_COMPLETED"
            or manifest.get("actual_pre_snap_forecasts_observed") is not False
            or manifest.get("engine_used") not in ACCEPTED_ENGINES):
        raise ValueError("Wrong or unapproved forecast provenance")
    audit = bundle["audit"]
    if (not isinstance(audit, dict)
            or any(audit.get(name) is not False for name in FALSE_AUDIT_FLAGS)):
        raise ValueError("Forecast bundle accessed labels or future outcomes")
    forecasts, skips = bundle["synthetic_forecasts"], bundle["skipped_opportunities"]
    if (not isinstance(forecasts, list) or not isinstance(skips, list)
            or not forecasts):
        raise ValueError("Need at least one synthetic forecast and explicit omissions")
    ids = _keys(forecasts, "forecast")
    skip_ids = _keys(skips, "skipped")
    if set(ids).intersection(skip_ids):
        raise ValueError("Forecast overlaps an explicitly skipped fourth down")
    if (type(audit.get("source_fourth_down_candidates")) is not int
            or audit["source_fourth_down_candidates"] != len(ids)+len(skip_ids)
            or audit.get("forecasted_fabricated_opportunities") != len(ids)
            or audit.get("unscored_opportunities") != len(skip_ids)):
        raise ValueError("Source opportunity denominator does not reconcile")
    skip_counts = Counter()
    for item in skips:
        if not isinstance(item.get("reason"), str) or not item["reason"]:
            raise ValueError("An abstention must have an explicit reason")
        skip_counts[item["reason"]] += 1
    if dict(sorted(skip_counts.items())) != audit.get("skip_reason_counts"):
        raise ValueError("Missing or inaccurate source omission counts")
    for item in forecasts:
        if not isinstance(item, dict) or set(item) != PRE_SNAP_KEYS:
            raise ValueError("Unexpected forecast input or post-play label leakage")
    if _canonical_hash(forecasts) != audit.get("synthetic_prediction_sha256"):
        raise ValueError("Synthetic forecast commitment hash mismatch")
    # Validate with the existing cohort builder on label-FREE empty suffixes
    # before ANY observed action or scoreboard input may be read.
    build_synthetic_cohort({
        "manifest": dict(REQUIRED_MANIFEST),
        "pre_snap": forecasts,
        "observed_plays": [], "final_games": [],
    })
    return {
        "policy_commit": POLICY_COMMIT,
        "engine_used": manifest["engine_used"],
        "source_reconstruction": FAKE_ASOF,
        "synthetic_forecast_sha256": _canonical_hash(forecasts),
        "synthetic_forecast_count": len(ids),
        "missing_opportunities": len(skip_ids),
        "full_precision_requested": manifest.get("full_precision_requested") is True,
        "data_labels_inspected_before_seal": False,
        "genuine_future_study_authorized": False,
    }


def run_fabricated_postseason_once(
    payload: Any, output_dir: Path, *, mode: Any, bootstrap_draws: int = 20,
) -> dict:
    """Exclusive one-time local study. Never evaluates genuine season data."""
    refuse_future_data(mode)  # MUST occur before dereferencing a caller object
    if not isinstance(payload, dict) or set(payload) != TOP_LEVEL:
        raise ValueError("Expected separate forecast, PBP roster, action and scoreboard")
    # No observed play or final result is touched during this phase.
    seal = validate_label_free_forecasts(payload["forecast_bundle"])
    base = Path(output_dir)
    if base.exists() or base.is_symlink():
        raise FileExistsError("A sealed evaluation already exists: never rescore")
    base.parent.mkdir(parents=True, exist_ok=True)
    base.mkdir(exist_ok=False)
    seal_bytes = _json_bytes({
        "trial": "SYNTHETIC_ONLY_NO_REAL_2027_OUTCOMES",
        "seal": seal,
    })
    with (base / "forecast_seal.json").open("xb") as fd:
        fd.write(seal_bytes)
    # From this line onward, only labels may be accessed. If any subsequent
    # verification fails, the sealed output directory remains forever blocked.
    try:
        season = verify_fabricated_complete_season(
            payload["pbp_games"], payload["reference_games"], mode=mode,
        )
        roster = {row["game_id"] for row in season["final_games"]}
        predicted = payload["forecast_bundle"]["synthetic_forecasts"]
        skipped = payload["forecast_bundle"]["skipped_opportunities"]
        all_keys = set(_keys(predicted, "forecast")+_keys(skipped, "skipped"))
        if any(game_id not in roster for game_id, _ in all_keys):
            raise ValueError("Forecast belongs to game outside verified season")
        observed = payload["observed_plays"]
        if not isinstance(observed, list) or set(_keys(observed, "observed")) != all_keys:
            raise ValueError("Observed opportunities differ from sealed denominator")
        cohort = build_synthetic_cohort({
            "manifest": dict(REQUIRED_MANIFEST),
            "pre_snap": predicted,
            "observed_plays": observed,
            "final_games": season["final_games"],
        })
        if cohort["included_decisions"] + len(cohort["excluded_decisions"]) != len(all_keys):
            raise RuntimeError("Final joined denominator does not equal source roster")
        study = evaluate(
            cohort["records"], cohort["manifest"],
            bootstrap_draws=bootstrap_draws,
        )
        if study["mode"] != "synthetic" or study["n_decisions"] != cohort["included_decisions"]:
            raise RuntimeError("Unexpected evaluator mode or label population")
        result = {
            "status": "SYNTHETIC_POSTSEASON_REHEARSAL_ONLY",
            "real_2027_evaluated": False,
            "actual_2027_regular_season_verified": False,
            "real_holdout_release_authorized": False,
            "forecast_seal_sha256": _sha256(seal_bytes),
            "source_crosscheck": season["audit"],
            "source_fourth_down_candidates": len(all_keys),
            "joined_decisions": cohort["included_decisions"],
            "exclusion_counts": cohort["exclusion_counts"],
            "observed_play_count": cohort["total_observed_plays"],
            "frozen_policy_commit": POLICY_COMMIT,
            "evaluated_synthetic_only": study,
            "interpretation": (
                "Single-use LOCAL fabricated-data arithmetic rehearsal; "
                "no real 2027 outcomes, independent data authentication, "
                "live forecasts or causal NFL coaching advantage."
            ),
        }
        scored_bytes = _json_bytes(result)
        with (base / "synthetic_evaluation.json").open("xb") as fd:
            fd.write(scored_bytes)
        with (base / "done.json").open("xb") as fd:
            fd.write(_json_bytes({
                "status": "FABRICATED_ONCE_COMPLETE",
                "forecast_seal_sha256": _sha256(seal_bytes),
                "evaluation_sha256": _sha256(scored_bytes),
                "real_release_approved": False,
            }))
        return result
    except Exception:
        with (base / "failed_after_seal.txt").open("xb") as fd:
            fd.write(b"SEALED_REHEARSAL_FAILED_NO_RETRY\n")
        raise
