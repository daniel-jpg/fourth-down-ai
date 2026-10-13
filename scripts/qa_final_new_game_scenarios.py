"""Post-release real-NFL-state + new-edge stress QA, never model validation.

Reads ONLY existing 2014-2025 archived NFL play-by-play from the repository.
Selects observed 2024-25 regular-season pre-snap 4th-down states; the
observed coach action/result are NOT used as model inputs or scoring labels.
All Q4 scenarios use the public demo's per-request native-35 + exact 0:00
guard, and all other periods use the public adaptive 300/42 -> 1200/43 rule.
Separately examines unprecedented constructed boundary states. No 2026/2027
files, no training, no releases and no hosting changes.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.qa_fourth_down_scenarios import Case, NORMAL, check_result
from src.streamlit_kickoff_preview import (
    isolated_engine_for_preview,
    isolated_engine_from_loaded_engine,
    preview_native_kickoff_feature_candidate,
)
from src.q4_demo_terminal_guard import use_q4_demo_terminal_guard
from src.situational_support import (
    deep_own_short_go_notice,
    late_q4_trailing_after_fg_notice,
    supported_recommendation_display,
)
from src.weather_fg_sensitivity import (
    compare_weather_scenario,
    validate_weather_input,
)

SOURCE = Path("data/fourth_downs_labeled.parquet")
REPORT = Path("artifacts/final_new_scenario_stress_2026_10.json")
ALLOWED_ROOFS = {"outdoors", "dome", "open", "closed"}
ALLOWED = {"NORMAL_GO_PASS", "NORMAL_GO_RUN", "FIELD_GOAL", "PUNT"}

# All fresh synthetic states: not previous 21-case anchors and not claimed
# to be specific historical NFL plays. Distinct game periods/clock/score/roof.
NEW_EDGES = [
    Case("NEW Q1 14:08 own12 4th7 tied dome", 1, 848, 7, 88, 0, "HOME", roof="dome"),
    Case("NEW Q1 00:41 opp42 4th1 down3", 1, 41, 1, 42, -3, "AWAY"),
    Case("NEW Q2 00:22 opp27 4th8 down3", 2, 22, 8, 27, -3, "HOME",
         receiver="OFFENSE", offense_to=1, defense_to=2),
    Case("NEW Q2 09:53 own44 4th4 up6", 2, 593, 4, 56, 6, "NEUTRAL",
         receiver="DEFENSE"),
    Case("NEW Q3 11:12 own4 4th2 tied", 3, 672, 2, 96, 0, "AWAY"),
    Case("NEW Q3 04:31 opp2 4th1 down5", 3, 271, 1, 2, -5, "HOME"),
    Case("NEW Q3 00:46 opp48 4th12 up1", 3, 46, 12, 48, 1, "AWAY"),
    Case("NEW Q4 14:22 own20 4th8 down2", 4, 862, 8, 80, -2, "AWAY"),
    Case("NEW Q4 05:19 opp43 4th2 tied", 4, 319, 2, 43, 0, "HOME"),
    Case("NEW Q4 02:24 own10 4th13 up3", 4, 144, 13, 90, 3, "AWAY",
         offense_to=0, defense_to=2),
    Case("NEW Q4 01:11 opp26 4th8 down2", 4, 71, 8, 26, -2, "AWAY",
         offense_to=1, defense_to=1),
    Case("NEW Q4 00:38 opp1 4th1 down6", 4, 38, 1, 1, -6, "HOME",
         offense_to=1, defense_to=0, expected="GO FOR IT"),
    Case("NEW Q4 00:25 opp2 4th2 down4", 4, 25, 2, 2, -4, "AWAY",
         offense_to=1, defense_to=1),
    Case("NEW Q4 00:03 opp18 4th2 tied", 4, 3, 2, 18, 0, "HOME",
         offense_to=0, defense_to=0),
    Case("NEW Q4 00:02 opp18 4th2 down2", 4, 2, 2, 18, -2, "HOME",
         offense_to=0, defense_to=0, expected="FIELD GOAL"),
    Case("NEW Q4 00:18 own40 4th10 down4", 4, 18, 10, 60, -4, "AWAY",
         offense_to=0, defense_to=0),
    Case("NEW OT regular opening 06:33 opp18", 5, 393, 3, 18, 0, "HOME",
         offense_to=2, defense_to=2,
         ot_format="REGULAR_SEASON", ot_phase="OPENING"),
    Case("NEW OT regular sudden death 02:09 own12", 5, 129, 9, 88, 0, "AWAY",
         offense_to=1, defense_to=0,
         ot_format="REGULAR_SEASON", ot_phase="SUDDEN_DEATH"),
    Case("NEW OT postseason response 07:14 opp20", 5, 434, 2, 20, -3, "AWAY",
         offense_to=2, defense_to=2,
         ot_format="POSTSEASON", ot_phase="RESPONSE"),
]


def _finite_series(frame: pd.DataFrame, column: str):
    return pd.to_numeric(frame[column], errors="coerce")


def historical_cases() -> tuple[list[Case], list[dict]]:
    """Select identifiable observed REG 2024-25 plays, no outcome information."""
    if not SOURCE.is_file():
        raise FileNotFoundError("Pinned historical development play archive missing")
    columns = [
        "season", "season_type", "game_id", "play_id", "qtr",
        "game_seconds_remaining", "yardline_100", "ydstogo",
        "score_differential", "posteam_timeouts_remaining",
        "defteam_timeouts_remaining", "posteam", "home_team",
        "away_team", "location", "action", "roof",
    ]
    # Missing source columns must be explicitly diagnosed, not filled with
    # invented stadium/clock/team conditions.
    data = pd.read_parquet(SOURCE)
    missing = sorted(set(columns).difference(data.columns))
    if missing:
        raise ValueError("Archived PBP lacks verifiable fields: " + repr(missing))
    values = data.loc[data["season"].isin([2024, 2025])
                      & data["season_type"].eq("REG")
                      & data["qtr"].isin([1, 3, 4])
                      & data["action"].isin(ALLOWED)].copy()
    for col in (
        "game_seconds_remaining", "yardline_100", "ydstogo",
        "score_differential", "posteam_timeouts_remaining",
        "defteam_timeouts_remaining", "play_id",
    ):
        values[col] = _finite_series(values, col)
    values = values.replace([np.inf, -np.inf], np.nan).dropna(subset=[
        "game_seconds_remaining", "yardline_100", "ydstogo",
        "score_differential", "posteam_timeouts_remaining",
        "defteam_timeouts_remaining", "game_id", "play_id", "posteam",
        "home_team", "away_team", "location", "roof",
    ])
    values = values.loc[
        values["yardline_100"].between(1, 99)
        & values["ydstogo"].between(1, 20)
        & values["ydstogo"].le(values["yardline_100"])
        & values["posteam_timeouts_remaining"].between(0, 3)
        & values["defteam_timeouts_remaining"].between(0, 3)
        & values["roof"].isin(ALLOWED_ROOFS)
        & values["posteam"].isin(values["home_team"].unique().tolist()
                                + values["away_team"].unique().tolist())
        & values["location"].isin(["Home", "Away", "Neutral"])
    ]
    values = values.loc[
        values.apply(lambda x: x.posteam in (x.home_team, x.away_team), axis=1)
        & values.apply(
            lambda x: int(x.qtr) == 4
            and 0 < x.game_seconds_remaining <= 900
            or int(x.qtr) == 3
            and 900 < x.game_seconds_remaining <= 1800
            or int(x.qtr) == 1
            and 2700 < x.game_seconds_remaining <= 3600,
            axis=1,
        )
    ].sort_values(["season", "game_id", "play_id"], ascending=[False, True, True])
    assert not values.duplicated(["game_id", "play_id"]).any()
    # These are selection bins, NOT target outcomes or coach-action labels.
    buckets = [
        ("Q4 trailing one-to-three final 120 FG range",
         lambda f: (f.qtr == 4) & f.game_seconds_remaining.le(120)
         & f.score_differential.between(-3, -1) & f.yardline_100.le(45)),
        ("Q4 trailing four-to-eight final 300 deep redzone",
         lambda f: (f.qtr == 4) & f.game_seconds_remaining.le(300)
         & f.score_differential.between(-8, -4) & f.yardline_100.le(25)),
        ("Q4 leading one-to-seven final 180 backed up",
         lambda f: (f.qtr == 4) & f.game_seconds_remaining.le(180)
         & f.score_differential.between(1, 7) & f.yardline_100.ge(60)),
        ("Q4 tied final 300 FG range",
         lambda f: (f.qtr == 4) & f.game_seconds_remaining.le(300)
         & f.score_differential.eq(0) & f.yardline_100.le(50)),
        ("Q4 down at least seven final 180",
         lambda f: (f.qtr == 4) & f.game_seconds_remaining.le(180)
         & f.score_differential.le(-7) & f.yardline_100.le(65)),
        ("Q4 final 60 trailing territory",
         lambda f: (f.qtr == 4) & f.game_seconds_remaining.le(60)
         & f.score_differential.lt(0) & f.yardline_100.le(60)),
        ("Q4 opening twelve minutes own side tied",
         lambda f: (f.qtr == 4) & f.game_seconds_remaining.ge(720)
         & f.score_differential.eq(0) & f.yardline_100.ge(55)),
        ("Q4 4th and one near midfield",
         lambda f: (f.qtr == 4) & f.ydstogo.le(1)
         & f.yardline_100.between(40, 60)),
        ("Q3 4th and short own twenty",
         lambda f: (f.qtr == 3) & f.ydstogo.le(3)
         & f.yardline_100.ge(78)),
        ("Q3 4th and ten plus own side",
         lambda f: (f.qtr == 3) & f.ydstogo.ge(10)
         & f.yardline_100.ge(55)),
        ("Q3 trailing exactly four opponent forty",
         lambda f: (f.qtr == 3) & f.score_differential.eq(-4)
         & f.yardline_100.le(40)),
        ("Q3 longer attempt opponent 35-50",
         lambda f: (f.qtr == 3) & f.yardline_100.between(35, 50)),
        ("Q3 fourth and short opponent five",
         lambda f: (f.qtr == 3) & f.ydstogo.le(3)
         & f.yardline_100.le(6)),
        ("Q1 own thirty 4th and short",
         lambda f: (f.qtr == 1) & f.ydstogo.le(3)
         & f.yardline_100.ge(68)),
        ("Q1 4th and short near midfield",
         lambda f: (f.qtr == 1) & f.ydstogo.le(3)
         & f.yardline_100.between(40, 60)),
        ("Q1 deep opposing red zone tied",
         lambda f: (f.qtr == 1) & f.score_differential.eq(0)
         & f.yardline_100.le(22)),
        ("Q1 fourth and long in own territory",
         lambda f: (f.qtr == 1) & f.ydstogo.ge(10)
         & f.yardline_100.ge(60)),
    ]
    seen, cases, notes = set(), [], []
    for label, test in buckets:
        sub = values.loc[test(values)]
        sub = sub.loc[~sub.apply(lambda x: (x.game_id, int(x.play_id)) in seen, axis=1)]
        if sub.empty:
            notes.append({"stratum": label, "missing": True})
            print("HISTORICAL_STRATUM_EMPTY", label, flush=True)
            continue
        row = sub.iloc[0]
        key = (row.game_id, int(row.play_id))
        seen.add(key)
        qtr = int(row.qtr)
        quarter_seconds = float(row.game_seconds_remaining) - (4-qtr)*900
        assert 0 < quarter_seconds <= 900
        place = "NEUTRAL" if row.location == "Neutral" else (
            "HOME" if row.posteam == row.home_team else "AWAY"
        )
        case = Case(
            name=f"REAL {row.game_id} play{int(row.play_id)}",
            qtr=qtr, secs=int(round(quarter_seconds)),
            ydstogo=float(row.ydstogo),
            yardline=float(row.yardline_100),
            score=float(row.score_differential),
            site=place, offense_to=int(row.posteam_timeouts_remaining),
            defense_to=int(row.defteam_timeouts_remaining),
            roof=str(row.roof),
        )
        cases.append(case)
        notes.append({
            "stratum": label, "game_id": str(row.game_id),
            "play_id": int(row.play_id), "season": int(row.season),
            "offense": str(row.posteam), "home": str(row.home_team),
            "away": str(row.away_team), "observed_play_type": str(row.action),
            "input_from_archived_presnap_only": True,
        })
    if len(cases) < 10:
        raise AssertionError(
            "Need >=10 actual independent archived play states; found " + str(len(cases))
        )
    return cases, notes


def simulate(engine: dict, case: Case, *, seed=43, force_n=1200) -> dict:
    st = case.state()
    if case.qtr == 4:
        private = isolated_engine_from_loaded_engine(engine)
        original = private["evaluate_post_play_states"]
        with use_q4_demo_terminal_guard(private):
            result = preview_native_kickoff_feature_candidate(
                st, simulations=1200, action_seed=seed, isolated_engine=private,
            )
        assert private["evaluate_post_play_states"] is original
        assert "_q4_demo_unchanged_evaluator" not in private
    else:
        # Mirror the real public adaptive 300/42 --> 1200/43 selection.
        # The hypothetical alternate 101 run always uses 1200, diagnostic only.
        if seed != 43:
            result = engine["recommend"](st, n_simulations=force_n, seed=seed)
        else:
            initial = engine["recommend"](st, n_simulations=300, seed=42)
            frame = initial["results"].loc[initial["results"]["eligible"]].copy()
            frame = frame.loc[frame["action"].isin(ALLOWED)].sort_values(
                "expected_win_probability", ascending=False
            )
            if len(frame) >= 2:
                p, r = frame.iloc[0], frame.iloc[1]
                denom = math.hypot(float(p["mc_se"]), float(r["mc_se"]))
                z = (float(p["expected_win_probability"])
                     - float(r["expected_win_probability"])) / denom if denom else float("inf")
                escalate = not math.isfinite(z) or z < 1.96
            else:
                escalate = True
            result = (
                engine["recommend"](st, n_simulations=1200, seed=43)
                if escalate else initial
            )
    return result


def classify_model_sanity(case: Case, result: dict, checked: dict) -> list[dict]:
    """Rule-like plausibility warnings only; not historical accuracy labels."""
    notes = []
    row = result["results"]
    eligible = row.loc[row["eligible"]]
    strategies = {NORMAL[str(r["action"])] for _, r in eligible.iterrows()}
    displayed = supported_recommendation_display(case.state(), checked["winner"])
    if case.qtr == 4 and case.score == -4 and 0 < case.secs <= 120:
        assert late_q4_trailing_after_fg_notice(
            case.state(), strategies
        ) is not None or "FIELD GOAL" not in strategies
        if checked["winner"] == "FIELD GOAL":
            assert displayed == "INSUFFICIENT DATA"
        notes.append({
            "severity": "qualification", "reason": (
                "Trailing-by-four late-FG recovery still lacks historical analogs; "
                "shown WPs are extrapolations"
            )
        })
    if case.yardline >= 90 and case.ydstogo <= 3:
        if checked["winner"] == "GO FOR IT":
            assert displayed == "INSUFFICIENT DATA"
            notes.append({
                "severity": "qualification",
                "reason": "Deep own-10 short-yardage GO is unsupported by historic data",
            })
    if case.qtr == 4 and case.secs <= 35 and case.score < 0 and (
        case.yardline >= 45 or case.score <= -4
    ) and checked["winner"] == "PUNT":
        notes.append({
            "severity": "review",
            "reason": "Punt with almost no clock and trailing; check recovery assumptions",
        })
    if case.qtr == 4 and case.secs <= 35 and case.score <= -4 and (
        checked["winner"] == "FIELD GOAL"
    ):
        notes.append({
            "severity": "review",
            "reason": "Late field goal while still trailing by at least one possession",
        })
    if case.qtr <= 3 and checked["wp"] > 99.0:
        notes.append({
            "severity": "review",
            "reason": "Nearly certain modeled game win very early",
        })
    if case.qtr == 4 and case.secs <= 10 and case.score < 0:
        # Model may count successful last-snap score; any high ~100% deserves review.
        if checked["wp"] > 98.0:
            notes.append({
                "severity": "review",
                "reason": "Near-certain win from trailing late, inspect last-snap scoring",
            })
    return notes


def one_case(engine: dict, case: Case) -> dict:
    result = simulate(engine, case)
    checked = check_result(case, result)
    flags = classify_model_sanity(case, result, checked)
    raw = result["results"]
    all_actions = {
        str(x["action"]): {
            "eligible": bool(x["eligible"]),
            "wp_pct": (round(float(x["expected_win_probability"])*100, 3)
                       if bool(x["eligible"]) else None),
            "mc_se_pp": (round(float(x["mc_se"])*100, 3)
                         if bool(x["eligible"]) else None),
        } for _, x in raw.iterrows() if str(x["action"]) in ALLOWED
    }
    model = {
        "qtr": case.qtr, "clock": case.secs, "ydstogo": case.ydstogo,
        "yards_from_opponent_endzone": case.yardline, "margin": case.score,
        "site": case.site, "roof": case.roof,
        "offense_timeouts": case.offense_to, "defense_timeouts": case.defense_to,
    }
    if case.qtr == 2:
        model["halftime_receiver"] = case.receiver
    if case.qtr == 5:
        model["ot_format"], model["ot_phase"] = case.ot_format, case.ot_phase
    print("QA_CASE", json.dumps({
        **checked, **model, "displayed": supported_recommendation_display(
            case.state(), checked["winner"],
        ),
        "flags": flags,
    }, sort_keys=True), flush=True)
    return {"name": case.name, "input": model, "output": checked,
            "actions": all_actions, "flags": flags}


def diagnose_seeds(engine: dict, cases: list[Case]) -> list[dict]:
    """Different Monte Carlo samples can flip near ties; report them, don't fit."""
    results = []
    for case in cases:
        if case.qtr == 5:
            continue
        baseline = simulate(engine, case)
        other = simulate(engine, case, seed=101)
        b = check_result(case, baseline)
        a = check_result(case, other)
        flip = b["winner"] != a["winner"]
        large_flip = flip and min(b["edge_pp"], a["edge_pp"]) > 5.0
        record = {
            "name": case.name,
            "seed43": [b["winner"], b["wp"], b["edge_pp"]],
            "seed101": [a["winner"], a["wp"], a["edge_pp"]],
            "flip": flip, "large_flip": large_flip,
        }
        print("QA_SEED_CHECK", json.dumps(record), flush=True)
        results.append(record)
    return results


def weather_extremes(engine: dict) -> dict:
    """Independent weather comparison must not become the official action."""
    case = next(c for c in NEW_EDGES if c.name.startswith("NEW Q4 01:11"))
    st = case.state()
    scenarios = {}
    for key, temp, wind, conditions in (
        ("cold_windy", 15, 35, "Snow"),
        ("warm_calm", 75, 0, "Clear / dry"),
        ("cold_windy_clear", 15, 35, "Clear / dry"),
    ):
        weather = validate_weather_input(
            roof=st["roof"], temperature_f=temp,
            wind_mph=wind, conditions=conditions,
        )
        result = compare_weather_scenario(st, weather=weather, live_engine=engine)
        assert result["applicable"] is True
        assert result["official_model_unchanged"] is True
        assert result["validation_shows_brier_improvement"] is False
        for k in (
            "original_made_probability", "weather_made_probability",
            "weather_fg_win_probability",
        ):
            assert 0 <= result[k] <= 1 and math.isfinite(result[k])
        scenarios[key] = {
            "FG_make_pct": round(100*result["weather_made_probability"], 3),
            "FG_win_pct": round(100*result["weather_fg_win_probability"], 3),
        }
    assert scenarios["cold_windy"]["FG_make_pct"] < scenarios["warm_calm"]["FG_make_pct"]
    assert scenarios["cold_windy"] == scenarios["cold_windy_clear"], (
        "Rain/snow must remain descriptive, not silently change odds"
    )
    print("QA_WEATHER_EXTREMES", json.dumps(scenarios), flush=True)
    return scenarios


def main() -> None:
    existing = {x.name for x in __import__(
        "scripts.qa_fourth_down_scenarios", fromlist=["CASES"]
    ).CASES}
    assert len(NEW_EDGES) == 19 and not existing.intersection(x.name for x in NEW_EDGES)
    engine = isolated_engine_for_preview()
    original_recommend = engine["recommend"]
    original_eval = engine["evaluate_post_play_states"]

    historical, provenance = historical_cases()
    print("QA_NEW_CASES_START", json.dumps({
        "historical": len(historical), "constructed": len(NEW_EDGES),
        "archive": str(SOURCE),
        "protected_2026_2027_data_read": False,
    }), flush=True)
    all_cases = []
    for c in historical + NEW_EDGES:
        all_cases.append(one_case(engine, c))
    assert engine["recommend"] is original_recommend
    assert engine["evaluate_post_play_states"] is original_eval

    # Three historical plays plus five unusual synthetic states receive
    # a second seed at unchanged 1,200-draw capacity.
    seed_cases = historical[:3] + [
        NEW_EDGES[i] for i in (4, 11, 12, 14, 15)
    ]
    seed_results = diagnose_seeds(engine, seed_cases)
    weather = weather_extremes(engine)

    summary = {
        "observed_historical_cases": len(historical),
        "novel_synthetic_cases": len(NEW_EDGES),
        "total_cases": len(all_cases),
        "large_seed_flips": sum(s["large_flip"] for s in seed_results),
        "review_flags": sum(
            note["severity"] == "review"
            for case in all_cases for note in case["flags"]
        ),
        "support_qualifications": sum(
            note["severity"] == "qualification"
            for case in all_cases for note in case["flags"]
        ),
        "real_game_winners_used_as_labels": False,
        "full_independent_policy_validation": False,
        "frozen_artifacts_modified": False,
        "future_2026_2027_outcomes_accessed": False,
        "live_app_source_modified": False,
        "no_historical_data_downloads": True,
    }
    output = {
        "summary": summary,
        "selection_provenance": provenance,
        "cases": all_cases,
        "seeds": seed_results,
        "weather": weather,
        "limitations": (
            "2024-25 games are already-examined historical development data; "
            "the model could have seen these years during retrospective work. "
            "Matching coach actions is NOT proof of better coaching policy. "
            "Low-frequency own-territory, end-clock and onside extrapolations "
            "are qualified, not empirically validated. This test is finite/"
            "consistency/plausibility QA only."
        ),
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(output, indent=2, allow_nan=False)+"\n")
    print("QA_FINAL_SUMMARY", json.dumps(summary), flush=True)
    print("QA_FINAL_REPORT", REPORT, flush=True)
    # A football plausibility flag is an explicit FAIL FOR CLAIMING "NO CRAZY
    # RESULTS", not necessarily an implementation exception. Document review.
    if summary["large_seed_flips"] or summary["review_flags"]:
        print("QA_REVIEW_REQUIRED: football plausibility/model uncertainty flags", flush=True)
    else:
        print("QA_NO_NEW_SEVERE_PLAUSIBILITY_FLAGS", flush=True)
    assert summary["total_cases"] >= 29
    print("QA_STRUCTURAL_PASS: all modeled action numbers consistent", flush=True)


if __name__ == "__main__":
    main()
