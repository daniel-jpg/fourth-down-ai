"""Read-only, same-source 2025 NFLverse + independently archived ESPN gate.

This is the dry-run collector contract for the future 2027 post-season
nflreadpy / ESPN workflow. The only permitted *real* season today is 2025,
using original version-pinned 2014-2025 archives already in this repository.

The CLI may separately ask the EXISTING ESPN collector to refresh only 2025,
but it never downloads 2026, never downloads 2027, and never modifies models,
the production app, or the protected 2026 Step 66/83 validation.

No field labeled "pre_snap" in a completed-game source proves real-time
availability. This module certifies final-game *historical* crosschecks only.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.historical_scoreboard_verification import (
    GAME_COLUMNS, pbp_game_census, team_code,
)
from src.independent_espn_margin_archive import (
    ARCHIVE, EXPECTED_GAMES, EXPECTED_KEYS_SHA256,
    EXPECTED_MARGINS_SHA256, PBP, SOURCE_PBP_SHA1, SOURCE_PBP_SHA256,
    replay_pinned_source,
)
from src.nflverse_2027_reconstruction import (
    HISTORICAL_GIT_BLOB, HISTORICAL_SOURCE, PREPLAY_SOURCE_FIELDS,
)

HISTORICAL_YEAR = 2025
HISTORICAL_EXPECTED_REG_GAMES = 272
FROZEN_POLICY = "31176d50af25db9dd42dae07e9b17cb5c2b930b4"
MODE = "verified_2025_archive_only"
ALLOWED_SCHEMA_SOURCE = "LOCKED_2014_2025_NFLVERSE_PBP_ARCHIVE"
OUTPUT_FIELDS = frozenset({
    "status", "source", "season", "policy_commit", "pbp_original_sha256",
    "pbp_original_git_blob_sha1", "labeled_original_sha256",
    "archived_independent_scores", "reg_game_count", "weeks",
    "season_2025_matchup_sha256", "preplay_source_field_names",
    "preplay_schema_sha256", "quarterback_and_action_labels_accessed",
    "independent_historical_score_margin_verified",
    "actual_2027_outcomes_accessed", "real_2027_collection_authorized",
    "timestamped_pre_snap_claim_authorized",
})


def _checked_year(year: Any, mode: Any) -> None:
    # This is deliberately FIRST, before consulting any data object or URL.
    if type(year) is not int or year != HISTORICAL_YEAR or mode != MODE:
        raise PermissionError(
            "Historical archive evidence is restricted to original 2025 only. "
            "Genuine 2027 collection needs separate post-2027-season review "
            "and is NOT enabled by flags or by the arrival of Week 18."
        )


def _read_pinned(path: Path, expected_git_blob: str, expected_sha256: str | None) -> bytes:
    if not path.is_file() or path.is_symlink():
        raise ValueError("Pinned historical data file missing or symlinked")
    raw = path.read_bytes()
    sha1 = hashlib.sha1(
        f"blob {len(raw)}\0".encode("ascii")+raw
    ).hexdigest()
    sha256 = hashlib.sha256(raw).hexdigest()
    if sha1 != expected_git_blob or (
        expected_sha256 is not None and sha256 != expected_sha256
    ):
        raise ValueError("Historical NFLverse input differs from original frozen source")
    return raw


def _hash_json(obj: Any) -> str:
    return hashlib.sha256(json.dumps(
        obj, sort_keys=True, separators=(",", ":"), allow_nan=False,
        ensure_ascii=True,
    ).encode("ascii")).hexdigest()


def historical_2025_source_evidence(*, year: Any, mode: Any) -> dict[str, Any]:
    """Source/file and 3151-game independent ESPN verification, offline.

    No network; no provider calls; no 2026 or 2027 outcomes.
    """
    _checked_year(year, mode)
    raw = _read_pinned(HISTORICAL_SOURCE, HISTORICAL_GIT_BLOB, None)
    labeled = _read_pinned(PBP, SOURCE_PBP_SHA1, SOURCE_PBP_SHA256)
    if not ARCHIVE.is_file() or ARCHIVE.is_symlink():
        raise ValueError("Original independently archived ESPN scores unavailable")
    archive = ARCHIVE.read_bytes()
    if not archive:
        raise ValueError("Original independently archived ESPN scores unavailable")

    # replay_pinned_source re-checks the WHOLE canonical matchup vector and
    # all 3151 historical final margins against independent ESPN evidence.
    espn_audit = replay_pinned_source()
    if (espn_audit["status"] != "ALL_HISTORICAL_FINAL_SCORES_MATCH"
            or espn_audit["source_game_count"] != EXPECTED_GAMES
            or espn_audit["game_keys_sha256"] != EXPECTED_KEYS_SHA256
            or espn_audit["game_margins_sha256"] != EXPECTED_MARGINS_SHA256):
        raise RuntimeError("Independent archived ESPN full-score audit failed")

    source = pd.read_parquet(PBP, columns=list(GAME_COLUMNS))
    if (source["season"].isna().any()
            or not source["season"].between(2014, 2025).all()):
        raise ValueError("Mixed/future seasons in historical NFLverse source")
    census = pbp_game_census(source)
    season = census[census["season"].eq(HISTORICAL_YEAR)].copy()
    if (len(season) != HISTORICAL_EXPECTED_REG_GAMES
            or season["pbp_home_margin"].isna().any()
            or set(season["week"]) != set(range(1, 19))):
        raise ValueError("2025 game census incomplete or missing final margins")

    # Still no 2025 forecast/outcome scorer. This census only uses final
    # matches and original hashes; source action/future columns are excluded.
    weekly = Counter(int(w) for w in season["week"].tolist())
    if (weekly[18] != 16 or sum(weekly.values()) != 272):
        raise ValueError("Expected Week 18 16 games and all 272 2025 finals")
    occurrences = Counter()
    ordered_keys = []
    for row in season.sort_values(["week", "home", "away"]).itertuples(index=False):
        home, away = team_code(row.home), team_code(row.away)
        if home == away:
            raise ValueError("Corrupt team abbreviations")
        occurrences[home] += 1
        occurrences[away] += 1
        ordered_keys.append([2025, int(row.week), home, away])
    if len(occurrences) != 32 or set(occurrences.values()) != {17}:
        raise ValueError("Incomplete 2025 team participation")

    return {
        "status": "HISTORICAL_2025_SOURCE_PROVENANCE_VERIFIED",
        "source": ALLOWED_SCHEMA_SOURCE,
        "season": 2025,
        "policy_commit": FROZEN_POLICY,
        "pbp_original_sha256": hashlib.sha256(raw).hexdigest(),
        "pbp_original_git_blob_sha1": HISTORICAL_GIT_BLOB,
        "labeled_original_sha256": hashlib.sha256(labeled).hexdigest(),
        "archived_independent_scores": {
            "provider": "ESPN historical score archive (2014-2025)",
            "archive_sha256": hashlib.sha256(archive).hexdigest(),
            "reference_games_verified": EXPECTED_GAMES,
            "original_keys_sha256": EXPECTED_KEYS_SHA256,
            "original_margins_sha256": EXPECTED_MARGINS_SHA256,
        },
        "reg_game_count": len(season),
        "weeks": {str(w): weekly[w] for w in sorted(weekly)},
        "season_2025_matchup_sha256": _hash_json(ordered_keys),
        "preplay_source_field_names": sorted(PREPLAY_SOURCE_FIELDS),
        "preplay_schema_sha256": _hash_json(sorted(PREPLAY_SOURCE_FIELDS)),
        "quarterback_and_action_labels_accessed": False,
        "independent_historical_score_margin_verified": True,
        "actual_2027_outcomes_accessed": False,
        "real_2027_collection_authorized": False,
        "timestamped_pre_snap_claim_authorized": False,
    }


def save_historical_2025_evidence(
    dest: Path, *, year: Any, mode: Any,
) -> dict[str, Any]:
    _checked_year(year, mode)
    if not isinstance(dest, Path) or dest.is_symlink() or dest.exists():
        raise FileExistsError("Historical evidence path must be new and non-symlink")
    packet = historical_2025_source_evidence(year=year, mode=mode)
    if set(packet) != OUTPUT_FIELDS:
        raise RuntimeError("Historical collector output schema changed")
    dest.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(packet, indent=2, sort_keys=True, allow_nan=False)+"\n"
    ).encode("utf-8")
    with dest.open("xb") as handle:
        handle.write(encoded)
    return packet
