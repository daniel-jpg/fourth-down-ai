"""Research-only 2014-2025 retrospective fourth-down source-field inventory.

Inventory is deliberately NOT an as-of-snap certification. A populated
historical PBP field can have been backfilled after the play or game,
and game-clock information is not a trusted wall-clock receipt time.

Reads only the already-pinned 2014-2025 raw source parquet, projects
only approved identifying/state/location columns, no coach action or
final result. Never downloads newer football data, predicts, trains,
or authorizes a future 2027 prospective study.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[1]
PBP=ROOT/"data/fourth_downs_raw.parquet"
EXPECTED_BLOB_SHA1="41ae23de12913f0bf06ac608ad4716ae48d75df8"
MIN_YEAR,MAX_YEAR=2014,2025

STATE_FIELDS=(
    "game_id","play_id","season","week","down","qtr",
    "game_seconds_remaining","yardline_100","ydstogo",
    "goal_to_go","score_differential",
    "posteam_timeouts_remaining","defteam_timeouts_remaining",
    "posteam","home_team","away_team",
)
VENUE_FIELDS=("location","roof","stadium","stadium_id","surface")
KICKOFF_DIRECTION_CANDIDATES=(
    "second_half_receiver","second_half_receiving_team",
    "second_half_kickoff_receiver","coin_toss_winner",
    "coin_toss_choice",
)
KICKER_CANDIDATES=("kicker_player_id","kicker_player_name","kicker","kicker_name")
HISTORICAL_WALL_CLOCK_HINTS=(
    "game_date","game_time","time","start_time",
)
TRUSTED_CAPTURE_FIELDS=(
    "source_observed_at_utc","source_received_at_utc",
    "model_forecast_issued_at_utc","independent_witnessed_at_utc",
    "independently_attested_snap_at_utc",
)
PROHIBITED_AS_PRESNAP=(
    "play_type","desc","result","epa","yards_gained",
    "converted","first_down","touchdown",
    "kicker_player_id","kicker_player_name",
)
AUDITED_FIELDS=tuple(dict.fromkeys(
    STATE_FIELDS+VENUE_FIELDS+KICKOFF_DIRECTION_CANDIDATES+
    KICKER_CANDIDATES+HISTORICAL_WALL_CLOCK_HINTS+TRUSTED_CAPTURE_FIELDS
))
SCOPE_FIELDS=("season","season_type","down","qtr")


def _git_blob_sha1(path: Path) -> str:
    raw=path.read_bytes()
    return hashlib.sha1(f"blob {len(raw)}\0".encode("ascii")+raw).hexdigest()


def _nonmissing(series: pd.Series) -> int:
    filled=series.notna()
    if pd.api.types.is_string_dtype(series):
        filled &= series.astype("string").str.strip().ne("").fillna(False)
    return int(filled.sum())


def _row_coverage(frame: pd.DataFrame, names: Iterable[str]) -> dict:
    summary={}
    for name in names:
        if name not in frame:
            summary[name]={
                "source_column_present":False,
                "nonmissing":0,"missing":len(frame),
                "coverage_fraction":0.0,
                "availability_certified_before_snap":False,
            }
        else:
            count=_nonmissing(frame[name])
            summary[name]={
                "source_column_present":True,
                "nonmissing":count,
                "missing":int(len(frame)-count),
                "coverage_fraction":round(count/len(frame),6),
                "availability_certified_before_snap":False,
            }
    return summary


def inventory_retrospective_fields(frame: pd.DataFrame, *, raw_columns: Iterable[str]) -> dict:
    """Pure, deterministic metadata census; no user action/outcome labels."""
    if not isinstance(frame,pd.DataFrame) or frame.empty:
        raise ValueError("Historical raw fourth-down frame must be nonempty")
    supplied=set(raw_columns)
    if not set(SCOPE_FIELDS).issubset(frame):
        raise ValueError("Historical frame has no reliable date/down/quarter scope")
    years=pd.to_numeric(frame["season"],errors="coerce")
    if years.isna().any() or not years.between(MIN_YEAR,MAX_YEAR).all():
        raise ValueError("Forbidden 2026 or later / unapproved historical seasons")
    if not set(frame.columns).issubset(
        supplied.intersection(set(AUDITED_FIELDS)|set(SCOPE_FIELDS))
    ):
        raise ValueError("Projected feature inventory received extra, potentially post-snap columns")
    reg=frame.loc[
        frame["season_type"].eq("REG")
        & frame["down"].eq(4)
        & frame["qtr"].isin((1,2,3,4)),
    ].copy()
    if reg.empty:
        raise ValueError("No regular-season regulation fourth downs in archive")
    if set(reg["season"]) - set(range(MIN_YEAR,MAX_YEAR+1)):
        raise ValueError("Future data appeared in research frame")
    counters=_row_coverage(reg,AUDITED_FIELDS)
    # Game-clock is in seconds remaining. It never establishes a *UTC
    # arrival timestamp*, and cannot prove this row existed pre-snap.
    by_year={}
    for year,subset in reg.groupby("season",sort=True):
        y=int(year)
        by_year[str(y)]={
            "regulation_fourth_down_rows":int(len(subset)),
            "venue_coverage":_row_coverage(
                subset,VENUE_FIELDS
            ),
            "play_clock_coverage":_row_coverage(
                subset,("game_seconds_remaining",)
            ),
        }
    return {
        "status":"RETROSPECTIVE_FIELD_COVERAGE_ONLY_NOT_DECISION_TIME_CERTIFIED",
        "source":"pinned 2014-2025 historical raw PBP",
        "years":[int(x) for x in sorted(reg["season"].unique())],
        "regulation_fourth_down_rows":int(len(reg)),
        "distinct_games":int(reg["game_id"].nunique()) if "game_id" in reg else None,
        "scope":"REG Q1-Q4 down=4 only",
        "fields":counters,
        "by_season":by_year,
        "risk_summary":{
            "snap_time_proof":"NOT AVAILABLE FROM RETROSPECTIVE PBP",
            "realtime_provider_observed_receipt":"NOT VERIFIED",
            "roof_open_closed_at_snap":"NOT VERIFIED BY STATIC GAME METADATA",
            "coin_toss_receiver_at_snap":"NOT VERIFIED EVEN IF HISTORICAL CANDIDATE EXISTS",
            "kicker_player_identified_before_the_attempt":
                "NOT VERIFIED; HISTORICAL kicker_player_id MAY BE POST-SNAP",
            "physical_stadium_identity":"RETROSPECTIVE ONLY",
            "trusted_independent_witness":"NOT AVAILABLE",
            "source_facts_with_independent_utc_signatures":0,
        },
        "post_snap_excluded_from_projected_frame":[
            k for k in PROHIBITED_AS_PRESNAP if k in supplied
        ],
        "safe_for_model_input":False,
        "permits_2026_gate_or_2027_prospective_evaluation":False,
    }


def main() -> None:
    import argparse
    p=argparse.ArgumentParser(description="Historical-only availability gap inventory")
    p.add_argument("--out",type=Path,default=ROOT/"artifacts/2027_prespnap_gap_inventory.json")
    args=p.parse_args()
    if not PBP.is_file() or _git_blob_sha1(PBP)!=EXPECTED_BLOB_SHA1:
        raise RuntimeError("Historical raw PBP source hash changed")
    columns=set(pq.ParquetFile(PBP).schema.names)
    if not set(SCOPE_FIELDS).issubset(columns):
        raise RuntimeError("Historical PBP lacks core REG/down/quarter/year fields")
    to_read=sorted(columns & (set(AUDITED_FIELDS)|set(SCOPE_FIELDS)))
    source=pd.read_parquet(PBP,columns=to_read)
    report=inventory_retrospective_fields(source,raw_columns=columns)
    report["source_git_blob_sha1"]=EXPECTED_BLOB_SHA1
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2,sort_keys=True,
                                   allow_nan=False)+"\n")
    print(json.dumps({
        "status":report["status"],
        "rows":report["regulation_fourth_down_rows"],
        "games":report["distinct_games"],
        "venue_fields":{
            k:{"present":report["fields"][k]["source_column_present"],
               "coverage":report["fields"][k]["coverage_fraction"]}
            for k in VENUE_FIELDS
        },
        "coin_toss_receiving_fields":{
            k:report["fields"][k]["source_column_present"]
            for k in KICKOFF_DIRECTION_CANDIDATES
        },
        "trustworthy_source_timestamp_fields":{
            k:report["fields"][k]["source_column_present"]
            for k in TRUSTED_CAPTURE_FIELDS
        },
        "post_snap_fields_explicitly_excluded":
            report["post_snap_excluded_from_projected_frame"],
        "forecast_ready":False,
    },sort_keys=True),flush=True)


if __name__=="__main__":
    main()
