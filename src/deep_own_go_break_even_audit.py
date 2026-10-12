"""Issue #9: read-only simulated GO-vs-PUNT break-even conversion audit.

This is a CONDITIONAL MODEL SENSITIVITY ANALYSIS, not causal football
evidence: for a fixed pre-snap state, separately sample frozen simulator
conditional GO-success and GO-failure branches by temporarily replacing
the conversion-probability callback in a PRIVATE cloned function namespace.
The original loaded engine, fitted estimator, public app and all model
assets remain unchanged. A fixed-branch interpolation reports which GO
conversion probability would make the simulator indifferent to punting.

Only original 2014-2024 training/previously-inspected validation PBP is
used for empirical support counts. 2025 remains excluded from this audit
despite being present in the pinned archive; 2026/2027 outcome sources
are neither downloaded nor opened. Zero matched GO cases is explicitly
reported as UNSUPPORTED, never replaced by smoothing or a fake estimate.

Monte Carlo standard errors describe simulator draws, NOT model error.
Altering p while holding success/failure donor distributions fixed is a
hypothetical sensitivity experiment, NOT an identified optimal action.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.streamlit_kickoff_preview import (
    isolated_engine_for_preview, isolated_engine_from_loaded_engine,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL_SPLIT = ROOT / "data/fourth_down_modeling_split.parquet"
EXPECTED_GIT_BLOBS = {
    "src/42_build_decision_engine.py": "fa69ee309221d8e25af86026956c082d1862d89a",
    "data/fourth_down_modeling_split.parquet": "fa1d64f187810c08edc9a113051c8b633e2e9168",
    "models/fake_conversion_model.joblib": "d4f2dd5037209d263c20cf28bca50d70138245d4",
    "models/win_probability_model.joblib": "4cdafd1687594ff624560ad140725d9c32d72a17",
    "data/go_shared_transition_pool.parquet": "018535dd7ccf98bed0e411c6a4bb08e1a1924b5a",
    "models/punt_field_position_model.joblib": "cc69d8ff315bef8ca6a4270dcb7262b236ca0ea2",
}
GO = ("NORMAL_GO_PASS", "NORMAL_GO_RUN")
REFERENCE_SEEDS = (43, 101)
REFERENCE_OWN_YARDS = (5, 10, 20, 35)
REPORT_PATH = ROOT / "artifacts/issue9_deep_own_go_break_even.json"
SUPPORT_FIELDS = [
    "game_id", "season", "split", "action",
    "ydstogo", "yardline_100", "converted",
]


def _check_number(value: Any, label: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{label} must be numeric")
    f = float(value)
    if not math.isfinite(f) or not 0 <= f <= 1:
        raise ValueError(f"{label} must be finite probability")
    return f


def _git_blob(path: Path) -> str:
    raw = path.read_bytes()
    return hashlib.sha1(
        f"blob {len(raw)}\0".encode("ascii") + raw
    ).hexdigest()


def verify_frozen_research_files() -> dict[str, str]:
    """Hash the already checked-in files before opening any data or model."""
    for name, sha in EXPECTED_GIT_BLOBS.items():
        path = ROOT / name
        if path.is_symlink() or not path.is_file() or _git_blob(path) != sha:
            raise ValueError(
                f"Frozen {name} source hash mismatch; refuse unreviewed data/model"
            )
    return dict(EXPECTED_GIT_BLOBS)


def break_even_sensitivity(
    *, p_model: float, wp_failure: float, wp_success: float,
    wp_punt: float,
) -> dict:
    """Linear *conditional donor WP* interpolation and conversion threshold.

    The conditional success/failure continuation distributions remain
    unchanged; no counterfactual NFL outcome or confidence interval for
    model error is inferred. Values near thresholds require larger MC runs.
    """
    p = _check_number(p_model, "predicted GO conversion")
    q0 = _check_number(wp_failure, "GO failure continuation WP")
    q1 = _check_number(wp_success, "GO success continuation WP")
    punt = _check_number(wp_punt, "PUNT continuation WP")
    slope = q1 - q0
    predicted = q0 + p * slope
    delta = predicted - punt
    if slope <= 0:
        # Successful GO cannot be worth more under these sampled branches:
        # no conventional conversion threshold has a meaningful direction.
        return {
            "status": "NONPOSITIVE_SIMULATED_SUCCESS_PREMIUM",
            "p_model": p, "wp_if_all_go_fail": q0,
            "wp_if_all_go_convert": q1, "wp_punt": punt,
            "wp_go_mixture_at_model_p": predicted,
            "simulated_go_minus_punt_pp": 100*delta,
            "model_implied_break_even_conversion_p": None,
            "model_probability_over_break_even_pp": None,
            "hypothetical_go_minus_punt_pp_if_conversion_overestimated_by": {},
        }
    pstar = (punt - q0)/slope
    if pstar < 0:
        status = "MODEL_GO_AHEAD_AT_ALL_CONVERSION_RATES"
    elif pstar > 1:
        status = "MODEL_PUNT_AHEAD_AT_ALL_CONVERSION_RATES"
    else:
        status = "INTERIOR_SIMULATED_BREAK_EVEN"
    sensitivity = {
        str(n): round(100*(q0+max(0.,p-n/100.)*slope-punt),5)
        for n in (5, 10, 15, 20, 25, 30)
    }
    return {
        "status": status,
        "p_model": p, "wp_if_all_go_fail": q0,
        "wp_if_all_go_convert": q1, "wp_punt": punt,
        "wp_go_mixture_at_model_p": predicted,
        "simulated_go_minus_punt_pp": round(100*delta,5),
        "model_implied_break_even_conversion_p": pstar,
        "model_probability_over_break_even_pp": round(100*(p-pstar),5),
        "hypothetical_go_minus_punt_pp_if_conversion_overestimated_by": sensitivity,
    }


def _check_state(state: dict) -> None:
    if not isinstance(state, dict):
        raise ValueError("Explicit fourth-down state is required")
    for key in (
        "qtr", "game_seconds_remaining", "yardline_100", "ydstogo",
        "score_differential", "posteam_timeouts_remaining",
        "defteam_timeouts_remaining", "site", "roof",
    ):
        if key not in state:
            raise ValueError(f"Missing pre-snap {key}")
    if (state["qtr"] != 1 or
            not 65 <= float(state["yardline_100"]) <= 99.5 or
            not 1.5 <= float(state["ydstogo"]) <= 3.5):
        raise ValueError("Only prereviewed Q1 deep-own 4th-and-2/3 states allowed")


def sample_private_conversion_arms(
    engine: dict, state: dict, action: str,
    *, seed: int, n_simulations: int,
) -> dict:
    """Sample two fixed GO-conversion branches and PUNT on isolated engine."""
    _check_state(state)
    if action not in GO:
        raise ValueError("Only normal GO RUN/PASS branches can be analyzed")
    if type(seed) is not int or seed < 0 or seed > 2**32-1:
        raise ValueError("Explicit nonnegative PRNG seed required")
    if type(n_simulations) is not int or not 32 <= n_simulations <= 5000:
        raise ValueError("Audited simulation budget must be 32..5000")
    private = isolated_engine_from_loaded_engine(engine)
    base = private["normalize_state"](state)
    for act in (action, "PUNT"):
        ok, reason = private["action_eligible"](base, act)
        if not ok:
            raise ValueError(f"Cannot compare ineligible {act}: {reason}")
    original = private["go_conversion_probability"]
    p = float(original(base, action))
    _check_number(p, "frozen GO conversion probability")
    samples={}
    try:
        # Pure counterfactual MODEL switch, private namespace only.
        for fixed in (0., 1.):
            private["go_conversion_probability"] = (
                lambda st, a, forced=fixed: forced
            )
            value = private["simulate_go"](
                base,action,n_simulations,np.random.default_rng(seed)
            )
            samples["failure" if fixed==0 else "success"] = value
    finally:
        private["go_conversion_probability"] = original
    if (engine["go_conversion_probability"] is original or
            not math.isclose(float(original(base, action)),p,abs_tol=1e-12)):
        # The cloned function is expected to be a different function object
        # but must reproduce unchanged frozen prediction.
        raise RuntimeError("Original GO conversion callback changed")
    punt=private["simulate_punt"](
        base,n_simulations,np.random.default_rng(seed)
    )
    numbers=[samples["failure"]["q_value"],
             samples["success"]["q_value"],punt["q_value"]]
    result=break_even_sensitivity(
        p_model=p,
        wp_failure=float(numbers[0]),
        wp_success=float(numbers[1]),
        wp_punt=float(numbers[2]),
    )
    return {
        **result,
        "action":action,
        "seed":seed,
        "simulations_per_arm":n_simulations,
        "branch_mc_standard_errors": {
            "failure":float(samples["failure"]["mc_se"]),
            "success":float(samples["success"]["mc_se"]),
            "punt":float(punt["mc_se"]),
        },
        "assumptions": {
            "conditional_outcome_donors_unchanged":True,
            "conversion_probability_varied_only_in_private_simulator":True,
            "opponent_short_field_punishment_in_simulator":True,
            "source_is_historical_development_not_future_games":True,
            "causal_policy_benefit_established":False,
        },
    }


def historical_local_support() -> list[dict]:
    """Development GO/PUNT decision counts; don't call them true propensities."""
    verify_frozen_research_files()
    df=pd.read_parquet(MODEL_SPLIT,columns=SUPPORT_FIELDS)
    if (df.empty or set(df["season"].dropna().astype(int)) !=
            set(range(2014,2026))
            or not set(df["split"].dropna()).issubset(
                {"train","validation","test"}
            )):
        raise ValueError("Historical source year or split contract changed")
    if not (
        df.loc[df["season"].between(2014,2022),"split"].eq("train").all()
        and df.loc[df["season"].between(2023,2024),"split"].eq("validation").all()
        and df.loc[df["season"].eq(2025),"split"].eq("test").all()
    ):
        raise ValueError("Historical development chronology changed")
    df=df.loc[df["season"].between(2014,2024)]
    results=[]
    for period, years in (
        ("train_2014_2022",range(2014,2023)),
        ("already_explored_validation_2023_2024",range(2023,2025)),
    ):
        sample=df.loc[df["season"].isin(years)]
        for own in REFERENCE_OWN_YARDS:
            region=sample.loc[
                sample["yardline_100"].ge(100-own)
                & sample["ydstogo"].between(1.5,3.5)
                & sample["action"].isin((*GO,"PUNT"))
            ]
            go=region.loc[region["action"].isin(GO)]
            punt=region.loc[region["action"].eq("PUNT")]
            converted=pd.to_numeric(go["converted"],errors="coerce")
            if (not converted.empty and
                    (converted.isna().any() or
                     not converted.isin((0,1)).all())):
                raise ValueError("GO conversion labels missing or invalid")
            results.append({
                "period":period,
                "territory":f"OWN_1_TO_{own}",
                "go_chosen_count":len(go),
                "normal_go_pass":int(go["action"].eq("NORMAL_GO_PASS").sum()),
                "normal_go_run":int(go["action"].eq("NORMAL_GO_RUN").sum()),
                "go_converted":int(converted.sum()) if len(converted) else 0,
                "go_games":int(go["game_id"].nunique()),
                "punt_chosen_count":len(punt),
                "punt_games":int(punt["game_id"].nunique()),
                "go_share_of_observed_go_or_punt":(
                    round(len(go)/len(region),6) if len(region) else None
                ),
                "go_conversion_rate_observed": (
                    round(float(converted.mean()),6) if len(go) else None
                ),
                "overlap_or_causal_propensity_identified":False,
            })
    return results


def planned_state(own_yardline: int) -> dict:
    if type(own_yardline) is not int or own_yardline not in REFERENCE_OWN_YARDS:
        raise ValueError("Only prespecified own-5/10/20/35 states")
    return {
        "qtr":1, "game_seconds_remaining":3030.0,
        "yardline_100":float(100-own_yardline), "ydstogo":3.0,
        "score_differential":0.0,
        "posteam_timeouts_remaining":3.0,
        "defteam_timeouts_remaining":3.0,
        "site":"NEUTRAL", "roof":"outdoors",
    }


def run_research_audit(*, seeds: tuple[int,...]=REFERENCE_SEEDS,
                       simulations: int=1200) -> dict:
    if seeds != REFERENCE_SEEDS:
        raise ValueError("Preregistered illustrative seeds must remain unchanged")
    # Check frozen source hashes before loading data or fitted engine.
    blobs=verify_frozen_research_files()
    support=historical_local_support()
    engine=isolated_engine_for_preview()
    observed_originals={
        "go_conversion_probability":engine["go_conversion_probability"],
        "simulate_go":engine["simulate_go"],
        "simulate_punt":engine["simulate_punt"],
    }
    outcomes=[]
    for own in REFERENCE_OWN_YARDS:
        state=planned_state(own)
        for seed in seeds:
            for action in GO:
                out=sample_private_conversion_arms(
                    engine,state,action,seed=seed,
                    n_simulations=simulations,
                )
                outcomes.append({"own_yardline":own,**out})
    for name,old in observed_originals.items():
        if engine[name] is not old:
            raise RuntimeError("Research audit mutated public engine namespace")
    return {
        "status":"ISSUE9_HISTORICAL_ONLY_MODEL_BREAK_EVEN_SENSITIVITY",
        "frozen_artifact_git_blobs":blobs,
        "study_population":"already examined 2014-2024 historical development only",
        "scenario_definition":"Q1 tied 05:30 left; 4th-and-3; NEUTRAL; 3 TO each",
        "scenarios":list(REFERENCE_OWN_YARDS),
        "seeds":list(seeds),
        "simulation_draws_per_conditional_arm":simulations,
        "empirical_support":support,
        "model_conditional_sensitivity":outcomes,
        "method":"GO(p)=p*WP(convert)+(1-p)*WP(fail), with existing conditional donor distributions held fixed; compare independent PUNT draws",
        "does_not_measure_true_go_success_probability":True,
        "does_not_estimate_causal_choice_of_go":True,
        "does_not_provide_model_specification_uncertainty_ci":True,
        "frozen_public_policy_unchanged":True,
        "contains_2026_2027_outcomes":False,
        "2025_not_used_for_development_metrics":True,
        "requires_independent_future_validation":True,
    }


def main() -> None:
    import argparse
    parser=argparse.ArgumentParser(
        description="No model change: historical support and forced-branch GO break-even"
    )
    parser.add_argument("--out",type=Path,default=REPORT_PATH)
    parser.add_argument("--simulations",type=int,default=1200)
    args=parser.parse_args()
    if (not args.out.resolve().is_relative_to(ROOT/"artifacts")
            or args.out.exists()):
        raise ValueError("Research may create NEW artifacts/ output only")
    report=run_research_audit(simulations=args.simulations)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2,sort_keys=True,
                                  allow_nan=False)+"\n")
    print(json.dumps({
        "status":report["status"],
        "empirical_support":report["empirical_support"],
        "summary":[{
            "own":x["own_yardline"],"action":x["action"],"seed":x["seed"],
            "p_model":round(x["p_model"],5),
            "break_even_p":(
                round(x["model_implied_break_even_conversion_p"],5)
                if x["model_implied_break_even_conversion_p"] is not None
                else None
            ),
            "model_go_minus_punt_pp":x["simulated_go_minus_punt_pp"],
            "required_model_p_overstatement_to_flip_pp":
                x["model_probability_over_break_even_pp"],
        } for x in report["model_conditional_sensitivity"]],
        "output":str(args.out),
    },sort_keys=True),flush=True)


if __name__=="__main__":
    main()
