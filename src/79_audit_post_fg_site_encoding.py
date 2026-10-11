"""Step 79: attribute anomalous HOME/AWAY sensitivity to post-FG kickoff WP.

The frozen Step 78 grid identified a strategy flip when only the offensive
game site changes near 4th-and-3 at opponent 30, Q4 7:38, down three.

Hold the event fixed: the 48-yard field goal is MADE and consumes five
seconds, tying the game. Compare three representations of the resulting
possession separately at HOME, AWAY and NEUTRAL:
  (1) existing synthetic kickoff-pending WP feature vector
  (2) native-like kickoff feature vector (normal 35, zero ydstogo)
  (3) mean over ALL 538 frozen score-neutral 2025 first-snap transitions.

This is deterministic *conditional made-FG* WP, not an action Q value.
No 2026 data, no new WP fit, no production changes, no causal policy claim.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from src.streamlit_kickoff_preview import isolated_engine_for_preview
from src.kickoff_2025_resolved import (
    load_resolved_2025_pool, resolve_post_score_q4_resolved,
)

OUTPUT=ROOT/"artifacts/79_kickoff_site_encoding_wp_attribution.json"
BASE={
    "qtr":4,"game_seconds_remaining":458.,
    "yardline_100":30.,"ydstogo":3.,
    "score_differential":-3.,
    "posteam_timeouts_remaining":3.,
    "defteam_timeouts_remaining":3.,
    "site":"AWAY","roof":"outdoors",
}
SITES=("AWAY","HOME","NEUTRAL")
FG_EXECUTION_SECONDS=5.


class CompleteDonorCycle:
    """Deliberately integrate every frozen donor exactly once, no MC noise."""
    def __init__(self,size:int):
        self.size=size
    def integers(self,low,high,size=None):
        if low!=0 or high!=self.size or size!=self.size:
            raise ValueError("The deterministic integration must include every donor.")
        return np.arange(self.size)


def one_site(site:str,engine:dict,donors)->dict:
    if site not in SITES:
        raise ValueError("Unsupported site axis.")
    base=engine["normalize_state"]({**BASE,"site":site})
    made=engine["make_wp_state"](
        base,
        possession_original=False,
        yardline_100=np.nan,down=np.nan,ydstogo=np.nan,
        elapsed_seconds=FG_EXECUTION_SECONDS,
        score_change_original=3.,
        is_kickoff=True,
    )
    if int(made["qtr"])!=4 or made["is_kickoff"]!=1.:
        raise RuntimeError("Successful FG must remain kickoff pending.")
    if float(made["_score_change_original"])!=3.:
        raise RuntimeError("Lost the made-FG score transition.")
    native=dict(made)
    native.update({
        "yardline_100":35.0,
        "down":np.nan,
        "ydstogo":0.0,
        "goal_to_go":0.0,
    })
    # Score and possession unchanged from the synthetic made-FG state.
    # These are diagnostic coordinate choices, not a new kickoff forecast.
    eval_one=engine["evaluate_post_play_states"]
    synthetic=float(eval_one(base,[made])[0])
    native_wp=float(eval_one(base,[native])[0])

    matched=resolve_post_score_q4_resolved(
        [made for _ in range(len(donors))],
        donors,rng=CompleteDonorCycle(len(donors)),
    )
    probs=np.asarray(eval_one(base,matched),dtype=float)
    if len(probs)!=len(donors) or not np.isfinite(probs).all():
        raise RuntimeError("Invalid exact frozen-donor WP integral.")
    if not np.isfinite([synthetic,native_wp]).all():
        raise RuntimeError("Invalid kickoff-pending WP output.")
    if not ((0<=probs)&(probs<=1)).all():
        raise ValueError("First-snap WP not a probability.")

    is_penalty=donors["transition_type"].eq("penalty_first_snap").to_numpy()
    if int(is_penalty.sum())!=50 or len(donors)!=538:
        raise RuntimeError("2025 donor universe unexpectedly changed.")
    if any(state["is_kickoff"]!=0 for state in matched):
        raise RuntimeError("Kickoff remained unresolved in the donor integration.")
    if any(state["is_home_posteam"]!=made["is_home_posteam"] for state in matched):
        raise RuntimeError("Ordinary/penalty first snap must retain receiving team.")
    end=float(probs.mean())
    return {
        "site":site,
        "post_fg_original_team_score_margin":0.0,
        "post_fg_clock_seconds":float(made["game_seconds_remaining"]),
        "model_coordinate_original_home_axis":int(base["_wp_original_on_home_axis"]),
        "receiving_team_home_axis":int(made["is_home_posteam"]),
        "is_kickoff_feature_at_pending":int(made["is_kickoff"]),
        "synthetic_pending_win_probability":synthetic,
        "native_like_pending_win_probability":native_wp,
        "all_538_first_snap_mean_win_probability":end,
        "488_ordinary_first_snap_mean_win_probability":float(
            probs[~is_penalty].mean()
        ),
        "50_penalty_first_snap_mean_win_probability":float(
            probs[is_penalty].mean()
        ),
        "synthetic_to_native_difference_pp":100*(native_wp-synthetic),
        "native_to_first_snap_difference_pp":100*(end-native_wp),
        "total_first_snap_minus_synthetic_pp":100*(end-synthetic),
        "donors":int(len(donors)),
        "simulations_or_random_sampling":False,
    }


def combine(sites:list[dict])->dict:
    lookup={s["site"]:s for s in sites}
    if len(lookup)!=3 or set(lookup)!=set(SITES):
        raise ValueError("Need exactly HOME, AWAY and NEUTRAL evidence.")
    kinds=(
        "synthetic_pending_win_probability",
        "native_like_pending_win_probability",
        "all_538_first_snap_mean_win_probability",
    )
    result={}
    for key in kinds:
        away=float(lookup["AWAY"][key])
        home=float(lookup["HOME"][key])
        neutral=float(lookup["NEUTRAL"][key])
        for val in (away,home,neutral):
            if not 0<=val<=1:
                raise ValueError("WP estimate outside [0,1].")
        mean=(home+away)/2.
        result[key]={
            "home_wp":home,
            "away_wp":away,
            "neutral_wp":neutral,
            "home_minus_away_wp_pp":100*(home-away),
            "neutral_minus_home_away_average_pp":100*(neutral-mean),
        }
    return result


def run():
    print("=== STEP 79: 4TH-AND-3 SITE SENSITIVITY INSIDE MADE-FG STATE ===",
          flush=True)
    print("FG ties game; 5s fixed FG clock; 538 descriptive first-snap donors.",
          flush=True)
    engine=isolated_engine_for_preview()
    donors=load_resolved_2025_pool()
    rows=[one_site(site,engine,donors) for site in SITES]
    report={
        "subject":"conditional success of 48-yard FG, tied-game kickoff",
        "exact_input":BASE,
        "score_change_original":3.,
        "field_goal_elapsed_seconds":FG_EXECUTION_SECONDS,
        "2025_historical_donors":538,
        "2026_outcomes_used":False,
        "frozen_production_win_probability_model":True,
        "not_action_level_fourth_down_value":True,
        "no_causal_or_future_accuracy_claim":True,
        "by_site":rows,
        "site_spread_diagnostic":combine(rows),
    }
    for row in rows:
        print(
            f"{row['site']:8} kickoff-pending="
            f"{row['synthetic_pending_win_probability']:.2%} "
            f"native-like={row['native_like_pending_win_probability']:.2%} "
            f"538-donor-first-snap={row['all_538_first_snap_mean_win_probability']:.2%} "
            f"total_delta="
            f"{row['total_first_snap_minus_synthetic_pp']:+.2f}pp",
            flush=True,
        )
    for name,val in report["site_spread_diagnostic"].items():
        print(f"{name}: HOME-AWAY={val['home_minus_away_wp_pp']:+.2f}pp "
              f"NEUTRAL-MIDPOINT="
              f"{val['neutral_minus_home_away_average_pp']:+.2f}pp",flush=True)
    OUTPUT.parent.mkdir(exist_ok=True,parents=True)
    OUTPUT.write_text(json.dumps(report,indent=2)+"\n")
    print("Saved diagnostic:",OUTPUT.relative_to(ROOT),flush=True)
    print("No live model change; this is not counterfactual decision validation.")


if __name__=="__main__":
    run()
