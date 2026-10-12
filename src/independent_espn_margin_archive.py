"""Offline replay of the *independently sourced* ESPN final-score archive.

The repository permanently stores the 2014-2025 home-margin vector and a
SHA-256 of its normalized game matchup keys. This intentionally small data
file survives GitHub Actions artifact expiration. It is labeled post-game
research evidence; NEVER use external finals inside pre-snap predictions.

This module makes zero network calls, loads zero models, and touches no
2026/2027 NFL outcome data or model promotion state.
"""
from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
import struct
from typing import Any, Mapping
import zlib

import numpy as np
import pandas as pd

from src.historical_scoreboard_verification import (
    FIRST, LAST, GAME_COLUMNS, pbp_game_census,
)

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "data/independent_espn_2014_2025_margin_archive.json"
PBP = ROOT / "data/fourth_downs_labeled.parquet"
SOURCE_PBP_SHA1 = "d7230df63cad1c24579ee40cabaacd0b330e4784"
SOURCE_PBP_SHA256 = "07fb573961131e02340b1403e9b03edec306523d528136e438f269447dbba903"
EXPECTED_GAMES = 3151
EXPECTED_KEYS_SHA256 = "58f90a4a6a61ad934573c75fe6e80490df906489e8100d62e55057b40828ac76"
EXPECTED_MARGINS_SHA256 = "4801e7df90243eecdd2d1e69bf22c71b1f7ffffc47cf59970071b6ff29809580"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _game_keys_sha256(games: pd.DataFrame) -> str:
    keys = games[["season","week","home","away"]].to_numpy().tolist()
    keys = [[int(y),int(w),str(h),str(a)] for y,w,h,a in keys]
    return _sha256(json.dumps(
        keys, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8"))


def load_archived_margins(path: Path = ARCHIVE) -> tuple[dict[str, Any], tuple[int,...]]:
    """Validate immutable vector and dictionary schema before using any labels."""
    metadata = json.loads(Path(path).read_text())
    if (not isinstance(metadata, dict) or metadata.get("schema_version") != 1
            or metadata.get("archive_status") !=
            "RESEARCH_ONLY_INDEPENDENT_HISTORICAL_FINAL_SCORES_NOT_PRESNAP"
            or metadata.get("years") != list(range(FIRST,LAST+1))
            or metadata.get("source_game_count") != EXPECTED_GAMES
            or metadata.get("source_week_count") != 209
            or metadata.get("source_pbp_git_blob_sha1") != SOURCE_PBP_SHA1
            or metadata.get("source_pbp_sha256") != SOURCE_PBP_SHA256
            or metadata.get("ordered_matchup_key_sha256") != EXPECTED_KEYS_SHA256
            or metadata.get("margin_vector_raw_sha256") != EXPECTED_MARGINS_SHA256
            or metadata.get("nuisance_training_model_changed") is not False
            or metadata.get("independent_final_score_matches") != EXPECTED_GAMES):
        raise ValueError("External historical score archive metadata/freeze mismatch")
    value = metadata.get("margin_vector_base64")
    if not isinstance(value, str) or len(value) > 10000:
        raise ValueError("External score vector is absent or oversized")
    try:
        compressed = base64.b64decode(value, validate=True)
        # The compressed frame is from the code-reviewed repo. Reject zip
        # bombs and trailing frame bytes before allocating any large arrays.
        decoder = zlib.decompressobj()
        packed = decoder.decompress(compressed, EXPECTED_GAMES * 2 + 1)
        if (len(packed) != EXPECTED_GAMES*2 or not decoder.eof
                or decoder.unused_data or decoder.unconsumed_tail):
            raise ValueError("External score vector has incorrect byte count")
    except (ValueError, zlib.error) as exc:
        raise ValueError("Malformed independent ESPN score archive") from exc
    if _sha256(packed) != EXPECTED_MARGINS_SHA256:
        raise ValueError("Independent ESPN score vector SHA-256 mismatch")
    margins = struct.unpack(f"<{EXPECTED_GAMES}h", packed)
    if not all(-150 <= score <= 150 for score in margins):
        raise ValueError("Implausible NFL final-game margin")
    return metadata, margins


def replay_score_audit(census: pd.DataFrame, path: Path = ARCHIVE) -> dict:
    """Compare *archived ESPN labels* with separately supplied PBP game census.

    Sorting by the exact same canonical (year, week, normalized home/away)
    and requiring a precommitted matchup-key SHA prevents accidental
    reordering, renamed-team drift, or silent mismatches.
    """
    manifest, margins = load_archived_margins(path)
    if not isinstance(census,pd.DataFrame) or not set({
        "season","week","home","away","pbp_home_margin","pbp_fourth_down_rows",
    }).issubset(census.columns):
        raise ValueError("Missing historical source game-census columns")
    if (len(census) != EXPECTED_GAMES
            or census.duplicated(["season","week","home","away"]).any()
            or census["pbp_home_margin"].isna().any()
            or set(census["season"]) != set(range(FIRST,LAST+1))):
        raise ValueError("PBP games are incomplete or have ambiguous final margins")
    ordered = census.sort_values(
        ["season","week","home","away"], kind="stable"
    ).reset_index(drop=True)
    key_hash = _game_keys_sha256(ordered)
    if key_hash != EXPECTED_KEYS_SHA256:
        raise ValueError("PBP game identities/order differ from independently archived ESPN keys")
    pbp_margins = pd.to_numeric(
        ordered["pbp_home_margin"], errors="raise"
    ).to_numpy(dtype=float)
    if not np.isfinite(pbp_margins).all():
        raise ValueError("PBP final game margins are not finite")
    wrong = np.flatnonzero(pbp_margins != np.asarray(margins,dtype=float))
    if len(wrong):
        sample = [
            {"season":int(ordered.loc[i,"season"]),
             "week":int(ordered.loc[i,"week"]),
             "home":str(ordered.loc[i,"home"]),
             "away":str(ordered.loc[i,"away"]),
             "pbp_margin":float(pbp_margins[i]),
             "espn_margin":int(margins[i])}
            for i in wrong[:10]
        ]
        raise ValueError(
            f"Independent ESPN-vs-PBP final scores differ in {len(wrong)} games: {sample}"
        )
    return {
        "status":"ALL_HISTORICAL_FINAL_SCORES_MATCH",
        "provider":"ESPN separately retrieved archived 2014–2025 REG scoreboards",
        "source_game_count":len(ordered),
        "source_fourth_down_rows":int(ordered["pbp_fourth_down_rows"].sum()),
        "seasons":list(range(FIRST,LAST+1)),
        "game_keys_sha256":key_hash,
        "game_margins_sha256":EXPECTED_MARGINS_SHA256,
        "provenance_run_url":manifest["audit_source"],
        "prospective_or_causal_claims_authorized":False,
    }


def replay_pinned_source() -> dict:
    raw = PBP.read_bytes()
    if _sha256(raw) != SOURCE_PBP_SHA256 or hashlib.sha1(
        f"blob {len(raw)}\0".encode("ascii")+raw
    ).hexdigest() != SOURCE_PBP_SHA1:
        raise RuntimeError("Original historical PBP Git artifact changed")
    frame = pd.read_parquet(PBP, columns=list(GAME_COLUMNS))
    return replay_score_audit(pbp_game_census(frame))


def main() -> None:
    print(json.dumps(replay_pinned_source(),sort_keys=True,indent=2))


if __name__ == "__main__":
    main()
