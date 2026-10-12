"""Measure *fabricated* 2027 frozen policy forecast compute latency only.

Uses exact registered Q1-Q3 adaptive 300/1200 trials and native Q4 1200,
historical 2014-2025 nuisance coefficients, pinned 2027 model assets.
NO NFL event retrieval, future outcomes, live feed, coaching deployment
or prospective policy approval.

These measurements are from the machine running the command; they do
NOT include a paid data feed's latency, input validation/transport,
trusted timestamp witness or live server deployment overhead.
"""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from statistics import median
import time
from typing import Any

from src.fourth_down_policy_study import POLICY_COMMIT
from src.frozen_2027_policy_bridge import (
    APP_HIGH_SIMULATIONS, build_synthetic_policy_forecast,
    check_pinned_engine, load_verified_engine,
)
from src.historical_nuisance_portable import (
    load_historical_parameters, assert_historical_source_integrity,
)

ROOT=Path(__file__).resolve().parents[1]
MANIFEST={
    "mode":"synthetic",
    "source_type":"fabricated_only",
    "policy_commit":POLICY_COMMIT,
}


def fabricated_state(**patch: Any) -> dict:
    """Fixed fabricated 2027 regulation fourth-down state."""
    original={
        "game_id":"synthetic_latency_2027_01_ABC_DEF",
        "play_id":"12345","season":2027,"week":1,"quarter":1,"down":4,
        "game_seconds_remaining":3100.0,
        "yardline_100":6.0,"ydstogo":4.0,"goal_to_go":0,
        "offense_team":"ABC","home_team":"ABC","away_team":"DEF",
        "site":"HOME","site_advantage":1,
        "score_differential":-4.0,
        "posteam_timeouts_remaining":3,
        "defteam_timeouts_remaining":2,
        "roof":"outdoors","second_half_receiver":None,
        "feature_asof":"pre_snap","source_type":"fabricated_only",
    }
    original.update(patch)
    return original


CASES=(
    {
        "name":"Q1_opponent_6_fourth_and_4",
        "overrides":{"quarter":1,"game_seconds_remaining":3100.0},
    },
    {
        "name":"Q2_half_receiver_known",
        "overrides":{
            "quarter":2,"game_seconds_remaining":2100.0,
            "yardline_100":30.0,"ydstogo":3.0,
            "second_half_receiver":"DEFENSE","score_differential":-3.0,
        },
    },
    {
        "name":"Q4_late_opponent_6_native_kickoff",
        "overrides":{
            "quarter":4,"game_seconds_remaining":299.0,
            "yardline_100":6.0,"ydstogo":4.0,
            "score_differential":-4.0,
        },
    },
    {
        "name":"Q4_deep_own_5_fourth_and_3",
        "overrides":{
            "quarter":4,"game_seconds_remaining":299.0,
            "yardline_100":95.0,"ydstogo":3.0,
            "score_differential":4.0,
        },
    },
)


def measure_fabricated_policy(*, repeats: int = 3) -> dict:
    if type(repeats) is not int or not 1 <= repeats <= 5:
        raise ValueError("Benchmark repeats must be integer 1..5")
    start=time.perf_counter()
    check_pinned_engine()
    engine=load_verified_engine()
    nuisance=load_historical_parameters()
    assert_historical_source_integrity(nuisance)
    startup_ms=round((time.perf_counter()-start)*1000,3)
    profiles=[]
    for case in CASES:
        state=fabricated_state(**case["overrides"])
        timings=[]
        last=None
        for _ in range(repeats):
            start=time.perf_counter()
            item=build_synthetic_policy_forecast(
                MANIFEST,state,engine=engine,
                nuisance_snapshot=nuisance,
                n_simulations=APP_HIGH_SIMULATIONS,
            )
            timings.append(round((time.perf_counter()-start)*1000,3))
            last=item
        audit=last["audit"]
        if (audit["full_precision"] is not True
                or not audit["observed_action_access"] is False
                or not audit["outcome_access"] is False
                or audit["prospectively_validated"] is not False
                or audit["frozen_policy_commit"] != POLICY_COMMIT
                or audit["registered_app_blob"] !=
                "b5588fb16bff7bb595155ab5141e946303f56a1f"):
            raise RuntimeError("Benchmark policy provenance is not preregistered")
        if case["name"].startswith("Q4"):
            if (audit["simulation_count"]!=1200
                    or audit["seed"]!=43
                    or audit["kickoff_input_mode"]!="Q4_NATIVE_STANDARD_35"):
                raise RuntimeError("Q4 native kickoff not evaluated at exact registered precision")
        elif (audit["pilot_simulations"]!=300
              or audit["simulation_count"] not in (300,1200)
              or audit["seed"] not in (42,43)):
            raise RuntimeError("Registered Q1-Q3 adaptive policy precision missing")
        profiles.append({
            "scenario":case["name"],
            "quarter":state["quarter"],
            "registered_policy_action":last["pre_snap"]["policy_action"],
            "normal_action_count":len(last["pre_snap"]["eligible_actions"]),
            "simulation_count":audit["simulation_count"],
            "seed":audit["seed"],
            "kickoff_input_mode":audit["kickoff_input_mode"],
            "adaptive_escalated":audit["adaptive_escalated"],
            "measurements_ms":timings,
            "median_ms":round(median(timings),3),
            "max_ms":max(timings),
            "synthetic_run":True,
        })
    return {
        "status":"FABRICATED_COMPUTE_LATENCY_ONLY_NO_REAL_SNAP_DEADLINE",
        "registered_policy_commit":POLICY_COMMIT,
        "source_type":"fabricated_only",
        "full_registered_precision":True,
        "startup_and_model_load_ms":startup_ms,
        "repeats":repeats,
        "python_version":platform.python_version(),
        "platform":platform.platform(),
        "cases":profiles,
        "includes_network_latency":False,
        "includes_vendor_created_to_collector_receipt_latency":False,
        "includes_external_witness_receipt_latency":False,
        "includes_trusted_real_snap_timestamps":False,
        "establishes_2027_prospective_ready":False,
        "interpretation":"Machine-specific descriptive runtime only, not p95 live end-to-end or NFL efficacy",
    }


def main() -> None:
    parser=argparse.ArgumentParser(description="Offline full-precision 2027-shaped fourth-down forecast timing")
    parser.add_argument("--repeats",type=int,default=3)
    parser.add_argument(
        "--out",type=Path,
        default=ROOT/"artifacts/2027_fabricated_frozen_policy_latency.json",
    )
    opts=parser.parse_args()
    result=measure_fabricated_policy(repeats=opts.repeats)
    opts.out.parent.mkdir(parents=True,exist_ok=True)
    opts.out.write_text(json.dumps(
        result,indent=2,sort_keys=True,allow_nan=False
    )+"\n")
    print(json.dumps({
        "status":result["status"],
        "startup_and_model_load_ms":result["startup_and_model_load_ms"],
        "repeats":result["repeats"],
        "cases":[{
            "scenario":p["scenario"],
            "median_ms":p["median_ms"],
            "max_ms":p["max_ms"],
            "simulation_count":p["simulation_count"],
        } for p in result["cases"]],
        "output":str(opts.out),
    },sort_keys=True),flush=True)


if __name__ == "__main__":
    main()
