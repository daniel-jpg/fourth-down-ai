"""Step 80: fixed-grid research evaluation of Q4 native kickoff features.

Keep the same frozen fourth-down/WP models, simulate the prespecified
Step 78 states, and change ONLY the previously missing normal Q4 kickoff
field-position/yards-to-go features in a private engine. Compare with
unchanged production and 538 observed 2025 continuation.

This is an input-representation candidate, NOT a validated 2026+
forecast or a causal FG-vs-GO test. No policy promotion or 2026 data access.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from src.kickoff_q4_native_features_candidate import (
    use_q4_native_kickoff_candidate,
)
from src.streamlit_kickoff_preview import (
    isolated_engine_for_preview,
    preview_observed_2025_resolved_recommendation,
)

STEP78=ROOT/"src/78_audit_q4_strategy_stability.py"
OUTPUT=ROOT/"artifacts/80_q4_native_kickoff_candidate_grid.json"
SIMULATIONS=1200
SEED=42


def step78():
    spec=importlib.util.spec_from_file_location("predeclared_step78_states",STEP78)
    if spec is None or spec.loader is None:
        raise RuntimeError("Frozen local football scenario study unavailable.")
    obj=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


def model_native_only(state:dict)->dict:
    private=isolated_engine_for_preview()
    orig=private["evaluate_post_play_states"]
    with use_q4_native_kickoff_candidate(private):
        result=private["recommend"](
            state,n_simulations=SIMULATIONS,seed=SEED,
        )
    if orig.__globals__["evaluate_post_play_states"] is not orig:
        raise RuntimeError("Private temporary evaluator was not restored.")
    return result


def tally(rows:list[dict])->dict:
    if not rows:
        raise ValueError("No paired situations.")
    models=("official","native_only_candidate","expanded_2025")
    result={}
    for model in models:
        selected=[r[model] for r in rows if r[model]["supported"]]
        result[model]={
            "supported":len(selected),
            "GO_top_count":sum(x["top_strategy"]=="GO FOR IT" for x in selected),
            "FG_top_count":sum(x["top_strategy"]=="FIELD GOAL" for x in selected),
            "PUNT_top_count":sum(x["top_strategy"]=="PUNT" for x in selected),
            "GO_vs_FG_MC_near_ties":sum(
                x["go_vs_fg_near_tie_mc_only"] for x in selected
            ),
        }
    result["native_vs_official_strategy_flip_count"]=sum(
        r["native_only_candidate"]["top_strategy"]
        !=r["official"]["top_strategy"]
        for r in rows if
        r["native_only_candidate"]["supported"] and r["official"]["supported"]
    )
    result["native_vs_expanded_strategy_disagreement_count"]=sum(
        r["native_only_candidate"]["top_strategy"]
        !=r["expanded_2025"]["top_strategy"]
        for r in rows if
        r["native_only_candidate"]["supported"] and r["expanded_2025"]["supported"]
    )
    return result


def run():
    print("=== STEP 80: NATIVE NORMAL-KICKOFF WP INPUT-CANDIDATE GRID ===",
          flush=True)
    frozen=step78()
    scenarios=frozen.build_states()
    if len(scenarios)!=14 or scenarios[0][0]!="reference":
        raise RuntimeError("Do not adjust the preregistered local grid.")
    out=[]
    for label,state in scenarios:
        official=isolated_engine_for_preview()["recommend"](
            state,n_simulations=SIMULATIONS,seed=SEED,
        )
        native=model_native_only(state)
        expanded=preview_observed_2025_resolved_recommendation(
            state,simulations=SIMULATIONS,action_seed=SEED,
        )
        scores={
            "official":frozen.snapshot(official),
            "native_only_candidate":frozen.snapshot(native),
            "expanded_2025":frozen.snapshot(expanded),
        }
        if not all(v["supported"] for v in scores.values()):
            raise RuntimeError(f"Missing GO/FG support in fixed case {label}.")
        rec={"scenario":label,"state":state,**scores}
        out.append(rec)
        print(
            f"{label:24} official GO-FG="
            f"{scores['official']['go_minus_fg_pp']:+7.2f} "
            f"native={scores['native_only_candidate']['go_minus_fg_pp']:+7.2f} "
            f"expanded={scores['expanded_2025']['go_minus_fg_pp']:+7.2f} "
            f"leaders="
            f"{scores['official']['top_strategy']}/"
            f"{scores['native_only_candidate']['top_strategy']}/"
            f"{scores['expanded_2025']['top_strategy']}",flush=True,
        )
    summary=tally(out)
    print("GRID SUMMARY:",json.dumps(summary,sort_keys=True),flush=True)
    report={
        "design":"Fixed local Step 78 Q4 scenario grid; no fit or policy validation",
        "frozen_production_weights":True,
        "private_native_kickoff_candidate":True,
        "experimental_only":True,
        "2026_outcomes_read":False,
        "2025_history_already_inspected":True,
        "no_causal_fourth_down_claim":True,
        "native_fields":{"down":"missing","ydstogo":0,
                         "yardline_100":35},
        "simulations_per_action":SIMULATIONS,
        "random_seed":SEED,
        "summary":summary,
        "scenarios":out,
        "limitations":(
            "Approximate MC SE omits model uncertainty. Native input "
            "encoding is plausible and structurally closer to historical "
            "PBP but not proven to improve causal FG-vs-GO decisions. "
            "The 2025 donors are not an independent untouched test."
        ),
        "promotion":"NOT AUTHORIZED",
    }
    OUTPUT.parent.mkdir(exist_ok=True,parents=True)
    OUTPUT.write_text(json.dumps(report,indent=2)+"\n")
    print("Saved candidate report:",OUTPUT.relative_to(ROOT),flush=True)
    print("Live Streamlit app and frozen production models unchanged.",flush=True)


if __name__=="__main__":
    run()
