"""Isolated, versioned SHADOW training-data intake, HISTORICAL 2014-2025 ONLY.

Before reading ANY play rows, the original 2014-2025 Git blob SHA-1 must
match. An updated nflverse file containing 2026/2027 (even at the old path)
is rejected BEFORE opening the parquet. No downloads, no future outcomes,
no model training, no frozen policy changes, no automatic promotion.

Outputs in local artifacts/shadow_historical_2014_2025/ deliberately keep
pre-snap feature facts, subsequently observed choices, and final labels
in separate tables. This is development-stage evidence, NOT a 2027
pre-snap timestamp witness or a new untouched evaluation sample.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.historical_fourth_down_cohort import (
    ACTION_COLUMNS, FEATURE_COLUMNS, KEY, OUTCOME_COLUMNS,
    PRESNAP_COLUMNS, SOURCE_COLUMNS, extract_historical_cohort,
    fingerprint_frame,
)
from src.independent_espn_margin_archive import replay_pinned_source

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/fourth_downs_labeled.parquet"
DESTINATION = ROOT / "artifacts/shadow_historical_2014_2025"
SOURCE_GIT_BLOB = "d7230df63cad1c24579ee40cabaacd0b330e4784"
FROZEN_POLICY = "31176d50af25db9dd42dae07e9b17cb5c2b930b4"
TRAIN_YEARS = tuple(range(2014, 2023))
VALIDATION_YEARS = (2023, 2024)
SEEN_DEV_YEARS = (2025,)
ALLOWED = frozenset((*TRAIN_YEARS, *VALIDATION_YEARS, *SEEN_DEV_YEARS))
FORBIDDEN_PRESNAP = frozenset({
    "action", "observed_action", "offense_win", "result", "converted",
    "yards_gained", "epa", "play_type", "desc", "touchdown", "field_goal_result",
    "post_snap", "winner",
})


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      allow_nan=False, ensure_ascii=True).encode("ascii")


def git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode("ascii") + data).hexdigest()


def require_pinned_source(path: Path = SOURCE) -> dict:
    """Compare complete source bytes before parsing; no path or season overrides."""
    if (Path(path).is_symlink() or Path(path).resolve() != SOURCE.resolve()
            or not SOURCE.is_file()):
        raise ValueError("Only original checked-in historical source path is approved")
    raw = SOURCE.read_bytes()
    blob = git_blob_sha1(raw)
    if blob != SOURCE_GIT_BLOB:
        raise ValueError(
            "Source Git blob differs from approved 2014-2025 original. "
            "Refusing to parse any 2026 or 2027 play-by-play outcomes."
        )
    return {
        "file": "data/fourth_downs_labeled.parquet",
        "git_blob_sha1": blob,
        "file_sha256": sha256_bytes(raw),
        "approved_seasons": list(range(2014, 2026)),
    }


def historical_role(season: Any) -> str:
    if isinstance(season, (bool, np.bool_)) or not isinstance(
        season, (int, np.integer)
    ):
        raise ValueError("Historical season must be an exact integer")
    year = int(season)
    if year in TRAIN_YEARS:
        return "train_2014_2022"
    if year in VALIDATION_YEARS:
        return "validation_2023_2024"
    if year in SEEN_DEV_YEARS:
        return "previously_seen_2025_development"
    raise ValueError("Future-model shadow intake blocks unapproved 2026/2027 seasons")


def require_same_keys(reference: pd.DataFrame, other: pd.DataFrame, label: str) -> None:
    if len(reference) != len(other):
        raise ValueError(f"{label} count must equal pre-snap count")
    if reference.duplicated(KEY).any() or other.duplicated(KEY).any():
        raise ValueError(f"Duplicate game/play in {label}")
    ref = set(reference[KEY].itertuples(index=False, name=None))
    keys = set(other[KEY].itertuples(index=False, name=None))
    if ref != keys:
        raise ValueError(f"{label} game/play IDs differ from pre-snap features")


def plan_historical_snapshot(cohort: dict) -> dict:
    """Pure deterministic development split; labels NEVER enter X."""
    needed = {"pre_snap", "observed_actions", "final_labels", "excluded", "audit"}
    if not isinstance(cohort, dict) or not needed.issubset(cohort):
        raise ValueError("Historical extractor has incomplete separate outputs")
    pre, actions, final, excluded = (
        cohort[k] for k in ("pre_snap", "observed_actions", "final_labels", "excluded")
    )
    if (list(pre.columns) != PRESNAP_COLUMNS
            or list(actions.columns) != ACTION_COLUMNS
            or list(final.columns) != OUTCOME_COLUMNS
            or list(excluded.columns) != [*KEY, "reason"]):
        raise ValueError("Exact source feature, action and terminal-label schemas required")
    if FORBIDDEN_PRESNAP & set(pre.columns) or FORBIDDEN_PRESNAP & set(FEATURE_COLUMNS):
        raise ValueError("Post-snap label/action leaked into feature matrix")
    if pre.empty or not pre["feature_asof"].eq("pre_snap").all():
        raise ValueError("No valid pre-snap-only historical development rows")
    require_same_keys(pre, actions, "observed_action")
    require_same_keys(pre, final, "final_label")
    if not final["offense_win"].isin([0, 1]).all():
        raise ValueError("Historical terminal outcome not Boolean 0/1")
    if actions["observed_action"].isna().any():
        raise ValueError("Missing observed action cannot enter historical development")

    memberships = pre[["game_id", "play_id", "season", "week"]].copy()
    memberships["development_role"] = [
        historical_role(y) for y in memberships["season"]
    ]
    memberships = memberships.sort_values(KEY, kind="stable").reset_index(drop=True)
    require_same_keys(pre, memberships, "chronological split")
    counts = dict(sorted(Counter(memberships["development_role"]).items()))
    if sum(counts.values()) != len(pre):
        raise RuntimeError("Chronological split dropped candidate decisions")
    if len(excluded) + len(pre) != cohort["audit"]["source_candidates"]:
        raise RuntimeError("Historical normal action omissions are unaccounted for")

    return {
        "membership": memberships, "counts_by_role": counts,
        "years": sorted(set(int(y) for y in pre["season"])),
        "source_candidates": int(cohort["audit"]["source_candidates"]),
        "included_decisions": len(pre), "excluded_decisions": len(excluded),
        "exclusion_counts": dict(cohort["audit"]["exclusion_counts"]),
        "distinct_included_games": int(pre["game_id"].nunique()),
        "observed_action_counts": dict(cohort["audit"]["observed_action_counts"]),
        "row_digests": {
            "features": fingerprint_frame(pre),
            "actions": fingerprint_frame(actions),
            "outcomes": fingerprint_frame(final),
            "excluded": fingerprint_frame(excluded),
            "chronological_roles": fingerprint_frame(memberships),
        },
    }


def version_manifest(plan: dict, source: dict, external: dict) -> dict:
    """The complete historical source is expected, not a cherry-picked subset."""
    if (source.get("git_blob_sha1") != SOURCE_GIT_BLOB
            or external.get("status") != "ALL_HISTORICAL_FINAL_SCORES_MATCH"
            or external.get("source_game_count") != 3151
            or external.get("source_fourth_down_rows") != 47942
            or plan["source_candidates"] != 47942
            or plan["years"] != list(range(2014, 2026))):
        raise ValueError("All frozen 2014-2025 games and independent score agreement required")
    return {
        "schema_version": 1,
        "status": "SHADOW_HISTORICAL_DEVELOPMENT_NOT_A_NEW_EVALUATION",
        "source": source,
        "registered_2027_policy_commit_UNCHANGED": FROZEN_POLICY,
        "independent_final_scores": {
            "status": external["status"],
            "games":external["source_game_count"],
            "source_fourth_down_rows":external["source_fourth_down_rows"],
            "game_keys_sha256":external["game_keys_sha256"],
            "game_margins_sha256":external["game_margins_sha256"],
        },
        "training_years": list(TRAIN_YEARS),
        "validation_years": list(VALIDATION_YEARS),
        "previously_seen_development_years": list(SEEN_DEV_YEARS),
        "pre_snap_columns": list(PRESNAP_COLUMNS),
        "pre_snap_feature_columns": list(FEATURE_COLUMNS),
        "observed_action_columns": list(ACTION_COLUMNS),
        "postgame_outcome_columns": list(OUTCOME_COLUMNS),
        "source_fourth_down_candidates": plan["source_candidates"],
        "included_historical_decisions": plan["included_decisions"],
        "excluded_historical_decisions": plan["excluded_decisions"],
        "exclusion_counts": plan["exclusion_counts"],
        "by_development_role": plan["counts_by_role"],
        "by_historical_action": plan["observed_action_counts"],
        "distinct_included_games": plan["distinct_included_games"],
        "table_canonical_sha256": plan["row_digests"],
        "contains_2026_or_2027_rows": False,
        "2025_is_an_untouched_holdout": False,
        "approved_for_production": False,
        "approved_for_new_future_training": False,
        "approved_for_2027_prospective_evaluation": False,
        "real_time_decision_provenance_proved": False,
        "parquet_open_only_after_source_blob_check": True,
    }


def write_snapshot(path: Path, cohort: dict, plan: dict, manifest: dict) -> dict:
    """Create a NEW local stage directory; write file hashes and manifest LAST."""
    directory = Path(path)
    if directory.exists() or directory.is_symlink():
        raise FileExistsError("Existing shadow intake must never be silently overwritten")
    tables = {
        "presnap_features.parquet": cohort["pre_snap"],
        "observed_actions.parquet": cohort["observed_actions"],
        "postgame_outcomes.parquet": cohort["final_labels"],
        "excluded_candidates.parquet": cohort["excluded"],
        "chronological_roles.parquet": plan["membership"],
    }
    directory.mkdir(parents=True, exist_ok=False)
    metadata = {}
    for filename, frame in tables.items():
        target = directory / filename
        frame.to_parquet(target, index=False)
        metadata[filename] = {
            "rows": len(frame), "sha256": sha256_bytes(target.read_bytes()),
        }
    result = {**manifest, "staged_tables": metadata}
    result["manifest_content_sha256"] = sha256_bytes(canonical_json(result))
    (directory / "manifest.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+"\n"
    )
    return result


def run() -> dict:
    source = require_pinned_source()      # Must precede ALL parquet reads.
    external = replay_pinned_source()     # Offline frozen ESPN score margin archive.
    raw = pd.read_parquet(SOURCE, columns=SOURCE_COLUMNS)
    cohort = extract_historical_cohort(raw)
    plan = plan_historical_snapshot(cohort)
    manifest = version_manifest(plan, source, external)
    result = write_snapshot(DESTINATION, cohort, plan, manifest)
    return {
        "status": "HISTORICAL_ONLY_SHADOW_INTAKE_COMPLETE",
        "source_candidates": result["source_fourth_down_candidates"],
        "included_decisions": result["included_historical_decisions"],
        "excluded_decisions": result["excluded_historical_decisions"],
        "development_roles": result["by_development_role"],
        "independently_verified_final_games": result["independent_final_scores"]["games"],
        "manifest_sha256": sha256_bytes((DESTINATION/"manifest.json").read_bytes()),
        "output_directory": DESTINATION.relative_to(ROOT).as_posix(),
        "production_models_changed": False,
        "2026_2027_game_rows_accessed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Research-only shadow staging of frozen 2014-2025 PBP"
    )
    parser.parse_args()  # NO source path or season override, intentionally.
    print(json.dumps(run(), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
