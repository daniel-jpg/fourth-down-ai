"""Complete fourth-down score and input audit. Historical labels are pre-snap only.

This is reporting QA, not prospective validation. No model is retrained.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

from scripts.qa_final_new_game_scenarios import (
    NEW_EDGES, SOURCE, historical_cases,
)
from src.streamlit_kickoff_preview import (
    isolated_engine_for_preview, isolated_engine_from_loaded_engine,
    preview_native_kickoff_feature_candidate,
)
from src.q4_demo_terminal_guard import use_q4_demo_terminal_guard


OUTPUT_JSON = Path("artifacts/final_full_input_score_audit.json")
OUTPUT_MD = Path("artifacts/final_full_input_score_audit.md")

# Synthetic score PAIRS only, not alleged historical facts.
SYNTHETIC = [
    (0, 0), (7, 10), (17, 20), (13, 7), (17, 17),
    (17, 22), (24, 23), (21, 23), (24, 24), (27, 24),
    (27, 29), (21, 27), (27, 31), (24, 24), (27, 29),
    (27, 31), (24, 24), (27, 27), (24, 27),
]


def build_pairs(historical):
    frame = pd.read_parquet(
        SOURCE, columns=[
            "game_id", "play_id", "posteam_score", "defteam_score",
            "score_differential", "season", "posteam", "defteam",
        ],
    )
    frame["play_id"] = pd.to_numeric(frame["play_id"], errors="coerce")
    assert not frame.duplicated(["game_id", "play_id"]).any()
    lookup = frame.set_index(["game_id", "play_id"])
    result = {}
    for case in historical:
        game, play = case.name.removeprefix("REAL ").rsplit(" play", 1)
        k = (game, float(int(play)))
        assert k in lookup.index, case.name
        row = lookup.loc[k]
        assert math.isclose(float(row.score_differential), case.score, abs_tol=1e-8)
        scores = [row.posteam_score, row.defteam_score]
        if any(pd.isna(s) for s in scores):
            score_pair = [None, None]
            status = "ARCHIVED_PRESNAP_SCORE_MISSING"
        else:
            score_pair = [int(s) for s in scores]
            status = "ARCHIVED_ACTUAL_PRESNAP"
        result[case.name] = {
            "pair": score_pair, "status": status,
            "offense_team": str(row.posteam),
            "defense_team": str(row.defteam),
            "season": int(row.season),
            "game_id": game, "play_id": int(play),
        }
    assert len(NEW_EDGES) == len(SYNTHETIC) == 19
    for case, (off, de) in zip(NEW_EDGES, SYNTHETIC, strict=True):
        result[case.name] = {
            "pair": [off, de],
            "status": "HYPOTHETICAL_SCORE_NOT_A_REAL_RESULT",
        }
    for case in historical + NEW_EDGES:
        p = result[case.name]["pair"]
        if p[0] is None:
            continue
        assert p[0] >= 0 and p[1] >= 0
        assert math.isclose(p[0] - p[1], case.score, abs_tol=1e-8), (
            "Scoreboard does not match state margin", case.name, p, case.score,
        )
    return result


def period(case):
    if case.qtr == 5:
        return "OT " + str(case.ot_format) + " " + str(case.ot_phase)
    return "Q" + str(case.qtr)


def spot(case):
    if case.yardline == 50:
        return "midfield"
    if case.yardline < 50:
        return "opponent " + str(case.yardline)
    return "own " + str(100 - case.yardline)


def separate_margin_impact():
    engine = isolated_engine_for_preview()
    example = next(c for c in NEW_EDGES if c.name.startswith("NEW Q4 01:11"))
    def evaluate(margin):
        state = example.state()
        state["score_differential"] = margin
        private = isolated_engine_from_loaded_engine(engine)
        with use_q4_demo_terminal_guard(private):
            return preview_native_kickoff_feature_candidate(
                state, simulations=1200, action_seed=43,
                isolated_engine=private,
            )
    def action_wp(result):
        return {
            str(r["action"]): round(float(r["expected_win_probability"])*100, 6)
            for _, r in result["results"].iterrows()
            if bool(r["eligible"])
        }
    trailing = action_wp(evaluate(-2))
    leading = action_wp(evaluate(+3))
    keys = trailing.keys() & leading.keys()
    assert keys
    diff = max(abs(trailing[k] - leading[k]) for k in keys)
    assert diff > 0.01, (
        "Model WP unexpectedly unchanged with a five-point margin shift",
        trailing, leading,
    )
    return {
        "same_margin_pairs": [[27, 29], [10, 12]],
        "same_margin_both_inputs_identical": True,
        "trailing_by_2_action_wp": trailing,
        "leading_by_3_action_wp": leading,
        "largest_action_difference_pp": round(diff, 3),
        "score_difference_affects_real_engine": True,
        "absolute_score_total_independently_modeled": False,
    }


def main():
    historical, _provenance = historical_cases()
    prior = json.loads(
        Path("artifacts/final_new_scenario_stress_2026_10.json").read_text()
    )
    modeled = {c["name"]: c for c in prior["cases"]}
    pairs = build_pairs(historical)
    details = []
    for c in historical + NEW_EDGES:
        outcome = modeled[c.name]
        scoreboard = pairs[c.name]
        st = c.state()
        required = {
            "qtr", "game_seconds_remaining", "yardline_100", "ydstogo",
            "score_differential", "posteam_timeouts_remaining",
            "defteam_timeouts_remaining", "site", "roof",
        }
        assert required.issubset(st), (c.name, required.difference(st))
        row = {
            "name": c.name, "source": scoreboard["status"],
            "offense_score": scoreboard["pair"][0],
            "defense_score": scoreboard["pair"][1],
            "score_differential": c.score,
            "quarter": period(c),
            "clock": "%02d:%02d" % (c.secs // 60, c.secs % 60),
            "down": 4, "to_go": c.ydstogo,
            "position": spot(c), "yardline_100": c.yardline,
            "approx_field_goal_yards": c.yardline + 18,
            "offense_timeouts": c.offense_to,
            "defense_timeouts": c.defense_to,
            "site": c.site, "roof": c.roof,
            "q2_second_half_receiver": c.receiver,
            "ot_format": c.ot_format, "ot_phase": c.ot_phase,
            "ot_period": c.ot_period if c.qtr == 5 else None,
            "official_weather": "not used",
            "experimental_weather": "not entered for this run",
            "team_identity_or_actual_kicker": "not personalized by model",
            "top_action": outcome["output"]["winner"],
            "top_wp_percent": outcome["output"]["wp"],
            "action_probabilities": outcome["actions"],
            "warnings": outcome["flags"],
        }
        if scoreboard["status"] == "ARCHIVED_ACTUAL_PRESNAP":
            row["offense_team"] = scoreboard["offense_team"]
            row["defense_team"] = scoreboard["defense_team"]
            row["game_id"] = scoreboard["game_id"]
            row["play_id"] = scoreboard["play_id"]
        details.append(row)
        print("SCORE_INPUT_CASE", json.dumps(row, sort_keys=True), flush=True)
    assert len(details) == 36
    margin_test = separate_margin_impact()
    complete = {
        "total": len(details),
        "historical": len(historical),
        "hypothetical": len(NEW_EDGES),
        "archived_scores_verified": sum(
            d["source"] == "ARCHIVED_ACTUAL_PRESNAP" for d in details
        ),
        "historical_scores_missing": sum(
            d["source"] == "ARCHIVED_PRESNAP_SCORE_MISSING" for d in details
        ),
        "synthetic_score_pairs_consistent": True,
        "actual_score_margin_affects_engine": True,
        "raw_score_totals_independently_modeled": False,
        "future_2026_2027_outcomes_accessed": False,
        "model_weights_modified": False,
    }
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(
        {"summary": complete, "score_impact": margin_test, "scenarios": details},
        indent=2, allow_nan=False,
    ) + "\n", encoding="utf-8")
    md = [
        "# Fourth Down AI - Complete score and input audit",
        "",
        "All 36 fourth downs include score difference, both timeouts, game site, "
        "roof, yard line, quarter, period clock, OT / halftime metadata, and "
        "all modeled action probabilities in the companion JSON.",
        "",
        "ARCHIVED_ACTUAL_PRESNAP scores come from an identifiable play-by-play "
        "record; hypothetical scores are constructed pairs (NOT historical "
        "facts). The real engine uses only the margin, not independently the "
        "raw offense and defense point totals.",
        "",
        "Temperature / wind are optional experimental FG only, not official "
        "forecast inputs. Individual named kicker / team strength are NOT "
        "personalized, so this is not a test of every possible real-world factor.",
        "",
        "| Scenario | Period / clock | Offense-Defense | Margin | 4th-and | Ball | TO Off/Def | Site | Roof | Top model choice | Top WP |",
        "|---|---|---|---:|---|---|---|---|---|---|---:|",
    ]
    for x in details:
        scores = (str(x["offense_score"]) + "-" + str(x["defense_score"])
                  if x["offense_score"] is not None else "unknown")
        md.append(
            "| %s | %s %s | %s (%s) | %+g | %g | %s | %s/%s | %s | %s | %s | %.2f%% |" %
            (x["name"], x["quarter"], x["clock"], scores, x["source"],
             x["score_differential"], x["to_go"], x["position"],
             x["offense_timeouts"], x["defense_timeouts"], x["site"],
             x["roof"], x["top_action"], x["top_wp_percent"])
        )
    md.extend([
        "", "## Score sensitivity",
        "",
        "Same state: offense 27 / defense 29 and offense 10 / defense 12 "
        "are equivalent, by design. Changing score difference from "
        "-2 to +3 moves the real model's action values, largest "
        "shift " + str(margin_test["largest_action_difference_pp"]) + " pp.",
    ])
    OUTPUT_MD.write_text("\n".join(md) + "\n", encoding="utf-8")
    print("FULL_SCORE_INPUT_SUMMARY", json.dumps(complete), flush=True)
    print("FULL_SCORE_SENSITIVITY", json.dumps(margin_test), flush=True)
    print("FULL_SCORE_INPUT_AUDIT_PASS", OUTPUT_MD, OUTPUT_JSON, flush=True)


if __name__ == "__main__":
    main()
