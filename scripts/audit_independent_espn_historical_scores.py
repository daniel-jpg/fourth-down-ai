"""Read-only external ESPN-vs-PBP historical game-result audit.

Explicit 2014–2025 only. Historical ESPN SCOREBOARDS are post-game truth,
never forecast features; do not call this from prediction/training or use
it to access 2026 held-out Step 66/83 or 2027 games.

Run:
    PYTHONPATH=. python scripts/audit_independent_espn_historical_scores.py \
        --seasons 2025 --out artifacts/espn_historical_2025_score_audit.json
    PYTHONPATH=. python scripts/audit_independent_espn_historical_scores.py \
        --seasons 2014 2015 2016 2017 2018 2019 2020 2021 2022 2023 2024 2025

ESPN's public scoreboard API is an independently sourced game-result
provider; availability/schema/rate limits are NOT under project control.
HTTP errors, unfinished games, ambiguous joins, and mismatched labels
fail CLOSED. Never substitute historical PBP results for ESPN results.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

from src.historical_scoreboard_verification import (
    FIRST, LAST, GAME_COLUMNS, parse_espn_week,
    pbp_game_census, season_int, verify_independent_margins,
)

ROOT = Path(__file__).resolve().parents[1]
PBP = ROOT / "data/fourth_downs_labeled.parquet"
SOURCE_GIT_BLOB = "d7230df63cad1c24579ee40cabaacd0b330e4784"
URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
USER_AGENT = "FourthDownAI/1.0 (+https://github.com/daniel-jpg/fourth-down-ai; historical final score research)"
TIMEOUT = 24
MAX_ATTEMPTS = 3
MAX_RESPONSE_BYTES = 4_000_000


def file_git_blob(path: Path) -> str:
    b = path.read_bytes()
    return hashlib.sha1(f"blob {len(b)}\0".encode("ascii")+b).hexdigest()


def fetch_one(yr: int, week: int) -> tuple[list[dict], dict]:
    """Only validated historical weeks; never hit a 2026+ URL."""
    season_int(yr)
    query = urlencode({
        "dates": yr, "seasontype": 2, "week": week, "limit": 100,
    })
    from src.historical_scoreboard_verification import week_int
    week_int(week, yr)
    url = f"{URL}?{query}"
    last_error = None
    for attempt in range(MAX_ATTEMPTS):
        try:
            request = Request(url, headers={
                "User-Agent": USER_AGENT, "Accept": "application/json",
            })
            with urlopen(request, timeout=TIMEOUT) as res:
                if res.status != 200:
                    raise RuntimeError(f"Unexpected HTTP {res.status}")
                data = res.read(MAX_RESPONSE_BYTES+1)
            if len(data) > MAX_RESPONSE_BYTES:
                raise RuntimeError("ESPN response exceeded maximum JSON size")
            payload = json.loads(data)
            games = parse_espn_week(payload, yr, week)
            checksum = hashlib.sha256(data).hexdigest()
            return games, {
                "season": yr, "week": week, "source_url": url,
                "raw_response_sha256": checksum, "final_events": len(games),
            }
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError,
                RuntimeError) as err:
            last_error = err
            if attempt+1 < MAX_ATTEMPTS:
                time.sleep(.6*(attempt+1))
    raise RuntimeError(
        f"Independent ESPN source UNAVAILABLE for {yr} week {week}: "
        f"{type(last_error).__name__}: {last_error}. "
        "No game result has been inferred or substituted."
    )


def run(seasons: list[int], output: Path) -> dict:
    if not seasons or len(seasons) != len(set(seasons)):
        raise ValueError("Select unique explicit historical seasons")
    years = sorted(season_int(y) for y in seasons)
    if not PBP.is_file() or file_git_blob(PBP) != SOURCE_GIT_BLOB:
        raise RuntimeError("Pinned historical source parquet was modified")
    raw = pd.read_parquet(PBP, columns=list(GAME_COLUMNS))
    census = pbp_game_census(raw)
    if set(census["season"]).difference(set(range(FIRST, LAST+1))):
        raise RuntimeError("Forbidden season in historical source")
    tasks = [(year, week) for year in years for week in range(
        1, (17 if year <= 2020 else 18)+1
    )]
    games, week_provenance = [], []
    # Bounded concurrency (no ESPN bombardment); each response and its hash
    # remain separately inspectable in the saved report.
    with ThreadPoolExecutor(max_workers=3) as pool:
        pending = {pool.submit(fetch_one, year, week):(year, week)
                   for year, week in tasks}
        for future in as_completed(pending):
            yr, week = pending[future]
            records, metadata = future.result()
            games.extend(records)
            week_provenance.append(metadata)
            print(f"ESPN {yr} W{week:02d}: {len(records)} verified final games",
                  flush=True)
    week_provenance.sort(key=lambda x:(x["season"], x["week"]))
    games.sort(key=lambda r:(r["season"],r["week"],r["home"],r["away"]))
    for year in years:
        n = sum(r["season"]==year for r in games)
        # 2022 BUF-CIN game was cancelled (no final score) so is not
        # allowed into the final-score dataset; other seasons have 256/272.
        expected = 256 if year <= 2020 else 271 if year == 2022 else 272
        if n != expected:
            raise RuntimeError(
                f"ESPN {year} completed-game coverage={n}, expected={expected}; "
                "must investigate missing/canceled events, not silently accept"
            )
    out = verify_independent_margins(
        census, games, expected_seasons=years
    )
    report = {
        **out,
        "source_parquet_git_blob_sha1": SOURCE_GIT_BLOB,
        "source_parquet_sha256": hashlib.sha256(PBP.read_bytes()).hexdigest(),
        "provider": "ESPN historical final NFL REG scoreboard Site API",
        "retrieval_type": "LIVE_HISTORICAL_POSTGAME_AUDIT_NOT_REALTIME",
        "source_response_provenance": week_provenance,
        "source_week_requests": len(tasks),
        "source_completed_games": len(games),
        "archived_espn_scores_file": "separate sidecar output",
        "source_score_snapshot_sha256": hashlib.sha256(json.dumps(
            games, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")).hexdigest(),
        "source_independence_limitations": [
            "Cross-source score agreement does not prove statistical calibration",
            "ESPN live historical responses are mutable until independently archived",
            "Only final game margins are verified, not play formation or time-of-snap",
            "No timestamps, roof context, kickoff receiving team or kicker priors verified",
            "No 2026-27 data inspected, no policy evaluation or training run",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True,
                                 allow_nan=False)+"\n")
    sidecar = output.with_name(output.stem+"_espn_scores.json")
    sidecar.write_text(json.dumps(games, indent=2, sort_keys=True,
                                  allow_nan=False)+"\n")
    print(json.dumps({
        "status":report["status"],"years":years,
        "source_week_requests":len(tasks),"espn_final_games":len(games),
        "pbp_game_count":report["pbp_games"],
        "matched_games":report["matched_final_scores"],
        "fourth_down_rows_verified":report["total_fourth_down_rows_in_verified_games"],
        "mismatched_or_unmatched":report["reason_counts"],
        "missing_pbp_scores":len(report["missing_pbp_scores"]),
        "archived_scores_sha256":report["source_score_snapshot_sha256"],
        "output":str(output),"sidecar":str(sidecar),
    }, sort_keys=True),flush=True)
    if report["status"] != "ALL_MATCHED":
        raise RuntimeError(
            "Independent scoreboard cross-check NOT COMPLETE. "
            f"See {output} for unverified or mismatched game labels."
        )
    return report


def main() -> None:
    parser=argparse.ArgumentParser(description="Historical-only independent ESPN final scores")
    parser.add_argument("--seasons", type=int, nargs="+", required=True)
    parser.add_argument("--out", type=Path,
                        default=ROOT/"artifacts/independent_espn_score_audit.json")
    arguments=parser.parse_args()
    run(arguments.seasons, arguments.out)


if __name__ == "__main__":
    main()
