"""Secondary, offline ESPN-vs-nflverse historical neutral-venue audit.

Score-matching a final game does not prove the physical site information
available when a fourth-down call happened. This compares two RETROSPECTIVE
venue indicators and reports discrepancies for review. It NEVER modifies
PBP inputs, source labels, model weights or a coaching recommendation.
"""
from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
import zlib

import pandas as pd

from src.historical_scoreboard_verification import GAME_COLUMNS, pbp_game_census
from src.independent_espn_margin_archive import (
    ARCHIVE, EXPECTED_GAMES, EXPECTED_KEYS_SHA256,
    PBP, _game_keys_sha256, load_archived_margins,
    replay_pinned_source,
)

NEUTRAL_FLAGS_SHA = "cfb3406242e2c30baf09736bda9ab88de974f238fc043a31d4a9880ee7c3a76b"


def load_archived_neutral_flags(archive: Path = ARCHIVE) -> tuple[int,...]:
    meta,_ = load_archived_margins(archive)
    entry = meta.get("neutral_site_provenance")
    if (not isinstance(entry,dict)
            or entry.get("status") != "SECONDARY_STADIUM_SITE_DIAGNOSTIC_NOT_SNAP_TIME_VERIFIED"
            or entry.get("espn_neutral_site_games") != 49
            or entry.get("binary_sha256") != NEUTRAL_FLAGS_SHA):
        raise ValueError("Independent venue-data metadata mismatch")
    data = entry.get("flag_vector_base64")
    if not isinstance(data,str) or len(data) > 1000:
        raise ValueError("Independent venue flags missing or oversized")
    try:
        decoder = zlib.decompressobj()
        raw = decoder.decompress(base64.b64decode(data,validate=True),EXPECTED_GAMES+1)
    except (ValueError,zlib.error) as exc:
        raise ValueError("Invalid independent ESPN neutral-site flags") from exc
    if (len(raw) != EXPECTED_GAMES or not decoder.eof
            or decoder.unused_data or decoder.unconsumed_tail
            or hashlib.sha256(raw).hexdigest() != NEUTRAL_FLAGS_SHA
            or any(n not in (0,1) for n in raw)
            or sum(raw) != 49):
        raise ValueError("Independent ESPN neutral-site flags failed integrity")
    return tuple(raw)


def independent_neutral_site_audit(source: pd.DataFrame) -> dict:
    """Requires documented location from one REG PBP game, not inferred home/away."""
    if not isinstance(source,pd.DataFrame) or "location" not in source.columns:
        raise ValueError("Historical PBP game-site location column required")
    census = pbp_game_census(source)
    if len(census) != EXPECTED_GAMES:
        raise ValueError("Incomplete historical game census for site verification")
    grouped = source[source["season_type"]=="REG"].groupby("game_id")
    venue = {}
    for gid,plays in grouped:
        values=plays["location"].drop_duplicates().tolist()
        if len(values)!=1:
            raise ValueError(f"Conflicting or ambiguous physical site for game {gid}")
        venue[str(gid)]=values[0]
    ordered=census.sort_values(
        ["season","week","home","away"],kind="stable"
    ).reset_index(drop=True)
    if _game_keys_sha256(ordered) != EXPECTED_KEYS_SHA256:
        raise ValueError("Independent venue archive game identity hash mismatch")
    external = load_archived_neutral_flags()
    mismatch,missing,pbp_neutral = [],[],0
    for index,row in ordered.iterrows():
        name=venue.get(str(row["game_id"]))
        if name not in {"Home","Neutral"}:
            missing.append({
                "game_id":str(row["game_id"]),"location_value":str(name)
            })
            continue
        inferred=int(name=="Neutral")
        pbp_neutral+=inferred
        if inferred != external[index]:
            mismatch.append({
                "game_id":str(row["game_id"]),"season":int(row["season"]),
                "week":int(row["week"]),"home":str(row["home"]),
                "away":str(row["away"]),
                "pbp_neutral":bool(inferred),"espn_neutral":bool(external[index]),
            })
    return {
        "audit":"2014-2025 retrospective ESPN neutralSite vs PBP location",
        "status":(
            "ALL_RETROSPECTIVE_SITE_FLAGS_MATCH"
            if not mismatch and not missing
            else "SITE_DISAGREEMENTS_REQUIRE_INDEPENDENT_REVIEW"
        ),
        "games":len(ordered),
        "espn_neutral_site_games":sum(external),
        "pbp_neutral_site_games":pbp_neutral,
        "source_location_missing_or_unrecognized":missing,
        "mismatch_games":mismatch,
        "mismatch_count":len(mismatch),
        "offline_archive_verified":True,
        "not_certified":[
            "stadium roof open/closed at the actual decision",
            "weather or kicker ability at the decision",
            "physical home-field advantage magnitude",
            "actual snap-time or model forecast timestamps",
        ],
        "changes_to_frozen_policy":False,
    }


def audit_pinned_source() -> dict:
    # Require the previously frozen PBP SHA256 and 3,151-game final
    # score replay BEFORE inspecting venue attributes from that source.
    replay_pinned_source()
    columns=list(GAME_COLUMNS)+["location"]
    raw=pd.read_parquet(PBP,columns=columns)
    return independent_neutral_site_audit(raw)


def main() -> None:
    import argparse
    parser=argparse.ArgumentParser(description="Offline ESPN-vs-PBP neutral venue diagnosis")
    parser.add_argument("--out",type=Path)
    args=parser.parse_args()
    result=audit_pinned_source()
    content=json.dumps(result,sort_keys=True,indent=2)+"\n"
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(content)
    print(json.dumps({
        "status":result["status"],"games":result["games"],
        "espn_neutral":result["espn_neutral_site_games"],
        "pbp_neutral":result["pbp_neutral_site_games"],
        "mismatches":result["mismatch_count"],
        "missing_locations":len(result["source_location_missing_or_unrecognized"]),
        "examples":result["mismatch_games"][:12],
        "result_path":str(args.out) if args.out else None,
    },sort_keys=True))
    # This is a DIAGNOSTIC: if retrospective sources disagree, print
    # exact matches and do not silently rewrite either pre-snap source.


if __name__=="__main__":
    main()
