"""Step 77: 2025 ordinary-only vs penalty-resolved Rams model sensitivity.

No learned updates, no policy claims and no 2026 game data.
This script uses the exact same frozen continuation WP model under
different experimental 2025 donor distributions. Explicit what-if
onside assumptions remain separate. One historical Q4 return TD
cannot calibrate future scoring-event risk.
"""
from __future__ import annotations
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from src.streamlit_kickoff_preview import (
    preview_observed_2025_recommendation,
    preview_observed_2025_resolved_recommendation,
    preview_observed_2025_with_onside,
    preview_rank_with_uncertainty,
    observed_2025_resolved_summary,
    isolated_engine_for_preview,
)

OUT = ROOT / "artifacts/77_rams_penalty_resolved_sensitivity.json"
SEED = 42
SIMULATIONS = 2400
RAMS = {
    "qtr":4,"game_seconds_remaining":299.,
    "yardline_100":6.,"ydstogo":4.,
    "score_differential":-4.,
    "posteam_timeouts_remaining":3.,
    "defteam_timeouts_remaining":3.,
    "site":"AWAY","roof":"outdoors",
}
# Stress probabilities are hypothetical upper bounds, NOT empirical 2026 rates.
RETURN_TD_STRESS = (0., .001, .0025, .005, .01)


def paired_action_snapshot(label: str, result: dict) -> dict:
    rows=result["results"].set_index("action")
    elig=rows.loc[rows["eligible"]]
    assert {"FIELD_GOAL","NORMAL_GO_PASS"}.issubset(set(elig.index))
    fg=float(elig.loc["FIELD_GOAL","expected_win_probability"])
    go=float(elig.loc["NORMAL_GO_PASS","expected_win_probability"])
    se_fg=float(elig.loc["FIELD_GOAL","mc_se"])
    se_go=float(elig.loc["NORMAL_GO_PASS","mc_se"])
    if not all(np.isfinite(x) for x in (fg,go,se_fg,se_go)):
        raise ValueError("Invalid scenario win probabilities or simulation errors.")
    rank=preview_rank_with_uncertainty(result)
    return {
        "scenario":label,
        "fg_win_probability":fg,"go_pass_win_probability":go,
        "fg_mc_se":se_fg,"go_mc_se":se_go,
        "go_minus_fg_pp":100*(go-fg),
        "near_tie_mc_only":bool(rank["near_tie"]),
        "leader_point_estimate":rank["leader"],
    }


def max_single_action_win_probability_shift(p_event: float) -> float:
    """General upper bound, NOT a simulation of scoring-return touchdowns.

    If a previously omitted event has conditional probability p for a
    chosen action, and its unknown conditional winning probability lies
    in [0,1], adding it changes that action's unconditional WP by at most
    p in absolute probability units. A made-FG-only subbranch has
    strictly no larger effect. Does not bound both arms jointly.
    """
    if not np.isfinite(p_event) or not 0<=p_event<=1:
        raise ValueError("Return TD scenario probability must be in [0,1].")
    return float(p_event)


def main():
    print("=== STEP 77: 488 vs 538 HISTORICAL Q4 DONORS (NOT PRODUCTION) ===")
    print("Rams-style Q4 4:59, away, down 4, 4th-and-4 at opponent 6.")
    official=isolated_engine_for_preview()["recommend"](
        RAMS, n_simulations=SIMULATIONS, seed=SEED
    )
    old=preview_observed_2025_recommendation(
        RAMS,simulations=SIMULATIONS,action_seed=SEED,
    )
    new=preview_observed_2025_resolved_recommendation(
        RAMS,simulations=SIMULATIONS,action_seed=SEED,
    )
    onside=preview_observed_2025_with_onside(
        RAMS,simulations=SIMULATIONS,action_seed=SEED,
        attempt_probability=.25,recovery_probability=.10,
        include_verified_penalties=True,
    )
    rows=[
        paired_action_snapshot("official_frozen_engine",official),
        paired_action_snapshot("previous_488_ordinary_only",old),
        paired_action_snapshot("expanded_538_verified_nonscoring",new),
        paired_action_snapshot("expanded_plus_25pct_onside_10pct_recovery",onside),
    ]
    for item in rows:
        print(f"{item['scenario']:<44} "
              f"FG={item['fg_win_probability']:.2%} "
              f"GO={item['go_pass_win_probability']:.2%} "
              f"GO-FG={item['go_minus_fg_pp']:+.2f}pp "
              f"MC_near_tie={item['near_tie_mc_only']}", flush=True)
    original_go=rows[1]["go_pass_win_probability"]
    if abs(rows[2]["go_pass_win_probability"]-original_go)<1e-9:
        print("Coincidental GO-prediction equality; independent donor rolls remain valid.")
    # Stress-test a rare, score-changing return touchdown by probability
    # bounds, NOT by pretending one 2025 event calibrates a 2026 branch.
    td_limits=[
        {
            "assumed_td_event_probability":p,
            "single_action_max_absolute_wp_shift_pp":
                100*max_single_action_win_probability_shift(p),
        }
        for p in RETURN_TD_STRESS
    ]
    summary=observed_2025_resolved_summary()
    assert summary["kickoffs"]==538
    assert summary["penalty_first_snaps"]==50
    report={
        "2025_already_inspected":True,
        "2026_outcomes_used":False,
        "models_frozen":True,
        "experimental_only":True,
        "validated_causal_fourth_down_policy":False,
        "simulation_per_action":SIMULATIONS,
        "2025_nonscoring_kickoff_summary":summary,
        "comparisons":rows,
        "return_td_sample":1,
        "return_td_empirical_denominator":584,
        "return_td_probability_estimated":False,
        "return_td_upper_bound_stress_not_forecast":td_limits,
        "return_td_bound_scope":(
            "Absolute win-probability shift for a single action under "
            "an assumed omitted-event rate; NOT a two-arm policy guarantee."
        ),
    }
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(report,indent=2)+"\n")
    print("Saved scenario comparison:",OUT.relative_to(ROOT),flush=True)
    print("No live model, 2026 outcomes or training changes.",flush=True)


if __name__=="__main__":
    main()
