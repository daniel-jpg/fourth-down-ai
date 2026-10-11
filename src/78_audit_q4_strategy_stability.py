"""Step 78: fixed Q4 fourth-down strategy robustness grid.

Question: does the apparent GO FOR IT preference near fourth-and-3 at the
opponent 30 (Q4 7:38, trailing three) survive nearby changes in clock,
distance, field position, score, timeouts, and game site?

This is a deterministic sensitivity study of a FROZEN model, not policy
validation. Comparing 2025 selected kickoff donors with an uncalibrated
kickoff-pending model cannot establish the optimal real-world decision.

The grid is chosen in advance. No tuning, no 2026 outcomes, no model
replacement, no change to the public Streamlit app. Normal actions only.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from src.streamlit_kickoff_preview import (
    STRATEGY_FOR_ACTION,
    isolated_engine_for_preview,
    preview_observed_2025_resolved_recommendation,
    preview_strategy_rank_with_uncertainty,
    preview_go_playcall_with_uncertainty,
    recommendation_sensitivity,
)

OUTPUT=ROOT/"artifacts/78_q4_fourth_down_local_strategy_grid.json"
SIMULATIONS=1200
SEED=42
BASE={
    "qtr":4, "game_seconds_remaining":458.0,
    "yardline_100":30.0, "ydstogo":3.0,
    "score_differential":-3.0,
    "posteam_timeouts_remaining":3.0,
    "defteam_timeouts_remaining":3.0,
    "site":"AWAY", "roof":"outdoors",
}

# Explicit, one-at-a-time perturbations; not optimized to favor a strategy.
# These are mathematical neighboring states, not observed 2026 plays.
CHANGES=(
    ("reference",{}),
    ("4th_and_2",{"ydstogo":2.0}),
    ("4th_and_4",{"ydstogo":4.0}),
    ("4th_and_5",{"ydstogo":5.0}),
    ("opponent_25",{"yardline_100":25.0}),
    ("opponent_35",{"yardline_100":35.0}),
    ("q4_05_38",{"game_seconds_remaining":338.0}),
    ("q4_09_38",{"game_seconds_remaining":578.0}),
    ("trailing_2",{"score_differential":-2.0}),
    ("trailing_4",{"score_differential":-4.0}),
    ("offense_one_timeout",{"posteam_timeouts_remaining":1.0}),
    ("defense_one_timeout",{"defteam_timeouts_remaining":1.0}),
    ("offense_home",{"site":"HOME"}),
    ("neutral_site",{"site":"NEUTRAL"}),
)


def build_states() -> list[tuple[str,dict]]:
    if len(set(label for label,_ in CHANGES)) != len(CHANGES):
        raise ValueError("Duplicate scenario label.")
    rows=[]
    for label,changes in CHANGES:
        state={**BASE,**changes}
        if (
            int(state["qtr"]) != 4
            or float(state["yardline_100"]) < float(state["ydstogo"])
            or float(state["game_seconds_remaining"]) <= 0
        ):
            raise ValueError("Invalid prespecified football state.")
        rows.append((label,state))
    return rows


def snapshot(result:dict)->dict:
    raw=result["results"]
    eligible=raw.loc[raw["eligible"]
                     & raw["action"].isin(STRATEGY_FOR_ACTION)].copy()
    if eligible["action"].duplicated().any():
        raise ValueError("Duplicate modeled normal action.")
    if eligible.empty:
        raise ValueError("No normal decisions eligible.")
    vals=dict(zip(eligible["action"],
                  eligible["expected_win_probability"]))
    err=dict(zip(eligible["action"],eligible["mc_se"]))
    go_actions=[a for a in ("NORMAL_GO_PASS","NORMAL_GO_RUN") if a in vals]
    if not go_actions or "FIELD_GOAL" not in vals:
        return {
            "supported":False,
            "reason":"GO or FG outside modeled eligibility",
            "eligible_actions":sorted(vals),
        }
    best_go=max(go_actions,key=lambda name:float(vals[name]))
    go=float(vals[best_go]);fg=float(vals["FIELD_GOAL"])
    go_se=float(err[best_go]);fg_se=float(err["FIELD_GOAL"])
    gap=100*(go-fg)
    near=abs(go-fg)<=1.96*float(np.hypot(go_se,fg_se))
    if not all(np.isfinite(x) for x in [go,fg,go_se,fg_se]):
        raise ValueError("Invalid WP or MC uncertainty.")
    if not (0<=go<=1 and 0<=fg<=1 and go_se>=0 and fg_se>=0):
        raise ValueError("Outside valid probability ranges.")
    strategy=preview_strategy_rank_with_uncertainty(result)
    playcall=preview_go_playcall_with_uncertainty(result)
    return {
        "supported":True,
        "top_strategy":strategy["leader"],
        "strategy_near_tie_mc_only":bool(strategy["near_tie"]),
        "go_best_action":best_go,
        "go_wp":go,
        "fg_wp":fg,
        "go_minus_fg_pp":gap,
        "go_vs_fg_near_tie_mc_only":bool(near),
        "go_mc_se":go_se,
        "fg_mc_se":fg_se,
        "run_vs_pass":(
            {
                "leader":playcall["leader"],
                "gap_pp":playcall["gap_pp"],
                "near_tie_mc_only":playcall["near_tie"],
            } if playcall is not None else None
        ),
    }


def summarize(scenarios:list[dict])->dict:
    rows=[r for r in scenarios if
          r["official"]["supported"] and r["expanded_2025"]["supported"]]
    if not rows:
        raise RuntimeError("No matched modeling scenarios for summary.")
    counts={}
    for kind in ("official","expanded_2025"):
        eligible=[r[kind] for r in rows]
        counts[kind]={
            "number_supported":len(eligible),
            "go_above_fg_point_count":sum(
                r["go_minus_fg_pp"]>0 for r in eligible
            ),
            "fg_above_go_point_count":sum(
                r["go_minus_fg_pp"]<0 for r in eligible
            ),
            "go_vs_fg_mc_near_ties":sum(
                r["go_vs_fg_near_tie_mc_only"] for r in eligible
            ),
            "top_strategy_counts":{
                strategy:sum(
                    r["top_strategy"]==strategy for r in eligible
                )
                for strategy in ("GO FOR IT","FIELD GOAL","PUNT")
            },
        }
    return {
        "prespecified_states":len(scenarios),
        "paired_supported":len(rows),
        "official":counts["official"],
        "expanded_2025":counts["expanded_2025"],
        "point_margin_sign_changed_between_models":sum(
            np.sign(r["official"]["go_minus_fg_pp"])
            !=np.sign(r["expanded_2025"]["go_minus_fg_pp"])
            for r in rows
        ),
        "strategy_leader_changed_between_models":sum(
            r["official"]["top_strategy"]
            !=r["expanded_2025"]["top_strategy"]
            for r in rows
        ),
    }


def run()->None:
    print("=== STEP 78: FROZEN Q4 GO-vs-FG LOCAL STABILITY GRID ===",flush=True)
    print("Prespecified 14 nearby states; 2014-22 production WP unchanged; "
          "2025 538-donor preview is descriptive, not validated.",flush=True)
    scenarios=[]
    for label,state in build_states():
        production=isolated_engine_for_preview()["recommend"](
            state,n_simulations=SIMULATIONS,seed=SEED,
        )
        new=preview_observed_2025_resolved_recommendation(
            state,simulations=SIMULATIONS,action_seed=SEED,
        )
        official=snapshot(production)
        expanded=snapshot(new)
        if official["supported"] and expanded["supported"]:
            sens=recommendation_sensitivity(production,new)
            entry={
                "name":label, "state":state,
                "official":official,"expanded_2025":expanded,
                "model_sensitive_strategy_only":bool(sens["model_sensitive"]),
            }
            print(
                f"{label:23} official GO-FG="
                f"{official['go_minus_fg_pp']:+6.2f}pp "
                f"expanded GO-FG={expanded['go_minus_fg_pp']:+6.2f}pp "
                f"leaders={official['top_strategy']}/"
                f"{expanded['top_strategy']} "
                f"expanded_GO_FG_MC_tie="
                f"{expanded['go_vs_fg_near_tie_mc_only']}",
                flush=True,
            )
        else:
            entry={
                "name":label,"state":state,
                "official":official,"expanded_2025":expanded,
                "model_sensitive_strategy_only":None,
            }
            print(f"{label:23} INELIGIBLE for paired GO/FG study",flush=True)
        scenarios.append(entry)
    summary=summarize(scenarios)
    print("RESULTS:",json.dumps(summary,sort_keys=True),flush=True)
    report={
        "study":"Q4 local perturbation, not historical policy calibration",
        "frozen_production_engine":True,
        "experimental_only":True,
        "2025_already_explored":True,
        "2026_outcomes_used":False,
        "not_causal_policy_validation":True,
        "no_onside_kick_assumptions":True,
        "selected_kickoff_population":"538 observed 2025 Q4 ordinary+verified penalty",
        "simulations_per_action":SIMULATIONS,
        "seed":SEED,
        "summary":summary,"scenarios":scenarios,
        "limitations":(
            "Fixed one-at-a-time local states, no historical decision ground truth; "
            "MC SE omits sampling selection, football-model bias and kick-team effects."
        ),
    }
    OUTPUT.parent.mkdir(exist_ok=True,parents=True)
    OUTPUT.write_text(json.dumps(report,indent=2)+"\n")
    print("Frozen development report:",OUTPUT.relative_to(ROOT),flush=True)
    print("Production engine, Streamlit UI and 2026 prospective gate unchanged.")


if __name__=="__main__":
    run()
