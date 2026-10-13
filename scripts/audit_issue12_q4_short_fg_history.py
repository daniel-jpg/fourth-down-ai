"""Issue #12: historical *support counts* for final-40s Q4 post-made-FG kicks.

This does NOT fit models, use game outcomes, estimate causal action values, or
open protected 2026+ seasons. Completed 2023-25 records have been examined
before; these counts are descriptive research/development evidence only.
"""
from __future__ import annotations

import json
from pathlib import Path

import nflreadpy as nfl
import pandas as pd

YEARS = tuple(range(2014, 2026))
TRAIN_YEARS = tuple(range(2014, 2023))
DEV_YEARS = (2023, 2024, 2025)
OUTPUT = Path("artifacts/issue12_q4_short_fg_historical_support.json")
ACTIONS = {
    "run", "pass", "qb_kneel", "qb_spike", "punt",
    "field_goal", "extra_point", "two_point_conversion",
}
NEEDED = {
    "season", "season_type", "game_id", "play_id", "play_type", "qtr",
    "posteam", "home_team", "away_team", "field_goal_result",
    "game_seconds_remaining", "yardline_100", "posteam_score",
    "defteam_score", "desc",
}


def select_made_fg_kickoffs(raw: pd.DataFrame) -> pd.DataFrame:
    """Select pre-kickoff evidence with exact, within-game preceding made FG."""
    if missing := NEEDED.difference(raw.columns):
        raise ValueError("Missing source fields: " + repr(sorted(missing)))
    if raw.empty:
        raise ValueError("Empty season.")
    df = raw.sort_values(["game_id", "play_id"]).reset_index(drop=True).copy()
    if df.duplicated(["game_id", "play_id"]).any():
        raise ValueError("Duplicate game/play ID.")
    games = df["game_id"]
    substantive = df["play_type"].isin(ACTIONS)

    def prior(col: str) -> pd.Series:
        return df[col].where(substantive).groupby(games).transform(
            lambda x: x.ffill().shift(1)
        )

    last_kind = prior("play_type")
    last_fg = prior("field_goal_result")
    last_team = prior("posteam")
    last_yardline = pd.to_numeric(prior("yardline_100"), errors="coerce")
    last_id = pd.to_numeric(prior("play_id"), errors="coerce")
    last_kick_id = pd.to_numeric(
        df["play_id"].where(df["play_type"].eq("kickoff"))
        .groupby(games).transform(lambda x: x.ffill().shift(1)),
        errors="coerce",
    )
    clock = pd.to_numeric(df["game_seconds_remaining"], errors="coerce")
    kickoff_yardline = pd.to_numeric(df["yardline_100"], errors="coerce")
    recv_lead = (
        pd.to_numeric(df["posteam_score"], errors="coerce")
        - pd.to_numeric(df["defteam_score"], errors="coerce")
    )
    is_made_fg = (
        last_kind.eq("field_goal")
        & last_fg.fillna("").astype(str).str.lower().eq("made")
    )
    is_first = last_kick_id.isna() | last_id.gt(last_kick_id)
    receiving_team_valid = df["posteam"].eq(df["home_team"]) | df["posteam"].eq(df["away_team"])
    selection = (
        df["season"].isin(YEARS)
        & df["season_type"].eq("REG")
        & df["qtr"].eq(4)
        & df["play_type"].eq("kickoff")
        & clock.ge(0) & clock.le(120)
        & kickoff_yardline.eq(35)
        & is_made_fg & is_first
        & df["posteam"].notna() & last_team.notna()
        & df["posteam"].ne(last_team)
        & receiving_team_valid
    )
    selected = df.loc[selection, [
        "season", "game_id", "play_id", "home_team", "away_team", "posteam", "desc",
    ]].copy()
    selected["clock"] = clock[selection].to_numpy(float)
    selected["receiver_lead_after_fg"] = recv_lead[selection].to_numpy(float)
    selected["prior_fg_yardline_100"] = last_yardline[selection].to_numpy(float)
    selected["kicker_site"] = [
        "HOME" if p == h else "AWAY"
        for p, h in zip(last_team[selection], df.loc[selection, "home_team"])
    ]
    selected["declared_onside_text"] = (
        selected["desc"].fillna("").astype(str)
        .str.contains("onside", case=False, regex=False)
    )
    return selected


def count_group(frame: pd.DataFrame) -> dict:
    return {
        "plays": int(len(frame)),
        "games": int(frame["game_id"].nunique()),
        "onside_description": int(frame["declared_onside_text"].sum()),
        "fg_from_opp_1_to_10": int(frame["prior_fg_yardline_100"].le(10).sum()),
        "fg_from_opp_1_to_10_games": int(
            frame.loc[frame["prior_fg_yardline_100"].le(10), "game_id"].nunique()
        ),
    }


def summarize(selected: pd.DataFrame) -> dict:
    result = {}
    for label, seasons in (
        ("2014_2022_training_era", TRAIN_YEARS),
        ("2023_2025_examined_development", DEV_YEARS),
    ):
        subset = selected.loc[selected["season"].isin(seasons)]
        result[label] = {}
        for low, high, bucket in (
            (0, 40, "0_to_40s"),
            (40, 60, "gt40_to_60s"),
            (60, 120, "gt60_to_120s"),
        ):
            window = subset.loc[
                subset["clock"].gt(low if low != 0 else -1)
                & subset["clock"].le(high)
            ]
            result[label][bucket] = {
                "all_after_made_fg": count_group(window),
                "receiving_team_leads_exactly_one": count_group(
                    window.loc[window["receiver_lead_after_fg"].eq(1)]
                ),
                "receiving_team_leads_one_to_three": count_group(
                    window.loc[window["receiver_lead_after_fg"].between(1, 3)]
                ),
                "receiving_leads_exactly_one_kicker_HOME": count_group(
                    window.loc[
                        window["receiver_lead_after_fg"].eq(1)
                        & window["kicker_site"].eq("HOME")
                    ]
                ),
                "receiving_leads_exactly_one_kicker_AWAY": count_group(
                    window.loc[
                        window["receiver_lead_after_fg"].eq(1)
                        & window["kicker_site"].eq("AWAY")
                    ]
                ),
            }
    return result


def main() -> None:
    rows = []
    print("ISSUE12_HISTORICAL_ONLY years=2014..2025 REG, Q4, <=120s", flush=True)
    for year in YEARS:
        print("ISSUE12_HISTORICAL_LOADING", year, flush=True)
        raw = nfl.load_pbp([year]).to_pandas()
        if raw.empty or not raw["season"].eq(year).all():
            raise ValueError("Untrusted season selection " + str(year))
        selected = select_made_fg_kickoffs(raw)
        rows.append(selected)
        print(
            "ISSUE12_HISTORICAL_SEASON", year,
            "all_short_post_FG_kicks", len(selected), flush=True,
        )
    all_rows = pd.concat(rows, ignore_index=True)
    all_rows["game_id"] = all_rows["season"].astype(str) + "_" + all_rows["game_id"].astype(str)
    if all_rows.duplicated(["game_id", "play_id"]).any():
        raise ValueError("Duplicate cross-year game/play.")
    groups = summarize(all_rows)
    for era, slots in groups.items():
        for bucket, counts in slots.items():
            print(
                "ISSUE12_HISTORICAL_COUNT", era, bucket,
                "all", counts["all_after_made_fg"],
                "receiver_lead_1", counts["receiving_team_leads_exactly_one"],
                flush=True,
            )
    report = {
        "issue": 12,
        "seasons": list(YEARS),
        "2026_or_later_loaded": False,
        "season_type": "REG",
        "game_labels_or_future_outcomes_used": False,
        "pre_kickoff_selector": (
            "Q4, normal own-35 kickoff <=120s, first kick after last "
            "substantive made FG, verified receiving team; no outcome filtering"
        ),
        "onside_identification": (
            "desc contains literal 'onside' (incomplete proxy; all kicks retained)"
        ),
        "not_counterfactual_validation": True,
        "already_examined_2023_2025": True,
        "qualified": (
            "Post-FG score margin and kickoff time are observational. "
            "These are broad analog counts, not exact 4th-and-1, own-1, "
            "three-score context matches or calibrated FG/GO benefits."
        ),
        "groups": groups,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
    print("ISSUE12_HISTORICAL_REPORT", OUTPUT, flush=True)


if __name__ == "__main__":
    main()
