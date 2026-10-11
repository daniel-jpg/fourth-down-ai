"""Step 75: frozen Rams Q4 hypothetical onside-policy sensitivity.

Only an experimental simulation under EXPLICITLY ASSUMED decisions.
The field position/elapsed distributions are observed 2025 outcomes
(488 ordinary, 37 verified onside, just four successful kickers).
No claim of a 2026 team-specific onside attempt or recovery probability,
and no inference about which real FG/GO choice was optimal.

The production win-probability model and Streamlit recommendation stay frozen.
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
    preview_observed_2025_recommendation,
    preview_observed_2025_with_onside,
    preview_rank_with_uncertainty,
    observed_2025_onside_summary,
)

R = ROOT/"artifacts/75_rams_onside_what_if.json"
RAMS={
    "qtr":4,
    "game_seconds_remaining":299.,
    "yardline_100":6.,"ydstogo":4.,
    "score_differential":-4.,
    "posteam_timeouts_remaining":3.,
    "defteam_timeouts_remaining":3.,
    "site":"AWAY","roof":"outdoors",
}
SIMULATIONS=2400
SEED=42
# This is a fixed, illustrative sensitivity grid, NOT data-derived policy.
GRID=[
    ("ordinary_only",0.,.10),
    ("10pct_attempt_10pct_recover",.10,.10),
    ("25pct_attempt_05pct_recover",.25,.05),
    ("25pct_attempt_10pct_recover",.25,.10),
    ("25pct_attempt_20pct_recover",.25,.20),
    ("50pct_attempt_10pct_recover",.50,.10),
    ("always_onside_never_recover",1.,0.),
    ("always_onside_always_recover",1.,1.),
]


def run():
    print("=== STEP 75: ONSIDE FG-SCENARIO SENSITIVITY (NOT POLICY ADVICE) ===")
    print("Rams-style away offense trailing by four, Q4 4:59 at opponent six.")
    print("2025 selected onside recovery evidence:",observed_2025_onside_summary())
    production_baseline=preview_observed_2025_recommendation(
        RAMS,simulations=SIMULATIONS,action_seed=SEED,
    )
    ordinary=production_baseline["results"].set_index("action")
    rows=[]
    for label,attempt,recovery in GRID:
        experiment=preview_observed_2025_with_onside(
            RAMS,attempt_probability=attempt,recovery_probability=recovery,
            simulations=SIMULATIONS,action_seed=SEED,
        )
        ranked=preview_rank_with_uncertainty(experiment)
        values=experiment["results"].set_index("action")
        fg=float(values.loc["FIELD_GOAL","expected_win_probability"])
        go=float(values.loc["NORMAL_GO_PASS","expected_win_probability"])
        fg_mc=float(values.loc["FIELD_GOAL","mc_se"])
        go_mc=float(values.loc["NORMAL_GO_PASS","mc_se"])
        if not all(np.isfinite(x) for x in (fg,go,fg_mc,go_mc)):
            raise ValueError("Onside simulation generated invalid probabilities")
        if attempt==0:
            if not np.array_equal(
                values["expected_win_probability"].to_numpy(),
                ordinary["expected_win_probability"].to_numpy(),
                equal_nan=True,
            ):
                raise RuntimeError("Zero-attempt onside scenario MUST match ordinary baseline")
        else:
            if go != float(ordinary.loc["NORMAL_GO_PASS","expected_win_probability"]):
                raise RuntimeError("Hypothetical FG onside branch changed GO outcome")
        rec={
            "label":label,"p_attempt_assumed":attempt,"p_recovery_assumed":recovery,
            "fg_wp":fg,"go_pass_wp":go,
            "fg_mc_se":fg_mc,"go_mc_se":go_mc,
            "go_minus_fg_pp":100*(go-fg),
            "near_tie_mc_only":bool(ranked["near_tie"]),
            "scenario_point_leader":ranked["leader"],
        }
        rows.append(rec)
        print(
            f"{label:34} FG={fg:.2%} GO={go:.2%} "
            f"GO-FG={100*(go-fg):+.2f}pp "
            f"MC-tie={rec['near_tie_mc_only']}",
            flush=True,
        )
    if not rows[-1]["fg_wp"] > rows[-2]["fg_wp"]:
        raise RuntimeError("Forced recovery should improve FG continuation versus failure")
    report={
        "data_season":2025,
        "2026_outcomes_read":False,
        "only_historical_onside_recovery_transitions":True,
        "on_side_recovery_records":4,
        "ordinary_records":488,
        "onside_matched_records":37,
        "simulations_per_action":SIMULATIONS,
        "model_status":"experiment only; no production promotion",
        "uncertainty_note":"Monte Carlo SE excludes coaching, data, model and policy uncertainty",
        "assumptions_note":"Attempt/recovery percentages are manually chosen scenario values, not 2026 predictions",
        "rows":rows,
    }
    R.parent.mkdir(exist_ok=True,parents=True)
    R.write_text(json.dumps(report,indent=2)+"\n")
    print("Saved experimental sensitivity:",R.relative_to(ROOT))
    print("Never use this single historical game example as a causal FG/GO recommendation.")


if __name__=="__main__":
    run()
