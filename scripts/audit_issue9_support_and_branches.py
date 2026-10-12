"""Historical support and transition-branch audit for issue #9.

Only checked-in 2014-2025 development files and unchanged frozen models.
Never accesses 2026+/held-out future outcomes or modifies model artifacts.
This audit does NOT infer optimal causal fourth-down coaching policy.
"""
from __future__ import annotations

from collections import defaultdict
import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.kickoff_q4_native_features_candidate import native_q4_normal_kickoff
from src.streamlit_kickoff_preview import (
    isolated_engine_for_preview,
    isolated_engine_from_loaded_engine,
)

ROOT = Path(__file__).resolve().parents[1]


def historical_support() -> None:
    df = pd.read_parquet(ROOT / "data/fourth_down_modeling_split.parquet")
    required = {"season", "action", "yardline_100", "ydstogo",
                "split", "converted"}
    assert required.issubset(df.columns), sorted(required - set(df.columns))
    assert int(df["season"].max()) <= 2025, "Protected later data must not be present"
    go = df.loc[
        df["action"].isin(["NORMAL_GO_PASS", "NORMAL_GO_RUN"])
    ].copy()
    punts = df.loc[df["action"].eq("PUNT")].copy()
    print("== Historical development support (2014-25, descriptive) ==")
    print("Dataset: 4th-down modeling_split, GO n=", len(go),
          "PUNT n=", len(punts), "range", int(df.season.min()),
          "-", int(df.season.max()))
    for split, group in go.groupby("split"):
        print(f"GO split {split}: n={len(group)}, years="
              f"{int(group.season.min())}-{int(group.season.max())}")
    print("Action training features in frozen GO model omit continuous yardline_100.")
    go_train = go.loc[go["split"].eq("train")]
    for yl, label in [
        (95, "own 1-5"), (90, "own 1-10"), (80, "own 1-20"),
        (65, "own 1-35")
    ]:
        selection = (
            go["yardline_100"].ge(yl)
            & go["ydstogo"].gt(1.5)
            & go["ydstogo"].le(3.5)
        )
        selection_train = (
            go_train["yardline_100"].ge(yl)
            & go_train["ydstogo"].gt(1.5)
            & go_train["ydstogo"].le(3.5)
        )
        rows=go.loc[selection]
        tr=go_train.loc[selection_train]
        print(f"{label} fourth-and-2-or-3 GO: full={len(rows)}, "
              f"TRAIN(<=2022)={len(tr)} "
              f"converted_train={int(tr.converted.sum()) if len(tr) else 0}, "
              f"converted_all={int(rows.converted.sum()) if len(rows) else 0}")
        for act in ["NORMAL_GO_PASS", "NORMAL_GO_RUN"]:
            acts=tr.loc[tr.action.eq(act)]
            print(f"  {act}: train_n={len(acts)}, "
                  f"converted_n={int(acts.converted.sum()) if len(acts) else 0}")
        pts=punts.loc[
            punts["yardline_100"].ge(yl)
            & punts["ydstogo"].gt(1.5)
            & punts["ydstogo"].le(3.5)
            & punts["split"].eq("train")
        ]
        print(f"  PUNT same region train_n={len(pts)}")
    # Engine actually picks 250 nearest donors from each outcome/action,
    # even if all 250 are distant from a rare target field position.
    engine=isolated_engine_for_preview()
    gp=engine["go_pool"]
    for action in ["NORMAL_GO_PASS", "NORMAL_GO_RUN"]:
        is_pass=int(action=="NORMAL_GO_PASS")
        for converted in [0, 1]:
            matching=gp.loc[
                gp.is_pass.eq(is_pass)&gp.converted.eq(converted)
            ].copy()
            matching["similarity"] = (
                (matching.yardline_100-95).abs()/10.
                + (matching.ydstogo-3).abs()/2.
                + 5 * ((matching.yardline_100.le(matching.ydstogo)).astype(int))
            )
            nearest=matching.nsmallest(250,"similarity")
            local=nearest.loc[(nearest.yardline_100 >=90)
                              & nearest.ydstogo.between(1.5,3.5)]
            print(f"Transition donors {action} converted={converted}: "
                  f"population={len(matching)}, sampled_top250={len(nearest)}, "
                  f"near-own-10-&-2to3={len(local)}, "
                  f"median_donor_yardline={nearest.yardline_100.median():.1f}, "
                  f"min_yardline={nearest.yardline_100.min():.1f}, "
                  f"median_gap={(nearest.yardline_100-95).abs().median():.1f}")
    return engine


def analyze_branches(engine, case_name: str, s: dict, simulations: int=1200) -> None:
    base = engine["normalize_state"](s)
    private = isolated_engine_from_loaded_engine(engine)
    original_eval=private["evaluate_post_play_states"]
    original_recommend=engine["recommend"]
    original_wp=engine["predict_original_team_wp"]
    captures = []
    in_evaluator = [False]
    # Try resolution may recursively evaluate touchdown continuation.
    # Only count the top-level n simulated outcomes, not nested Trys.
    # Wrap private evaluator WITHOUT mutating original globals.
    def inspect(base, states):
        transformed = [
            native_q4_normal_kickoff(x) if int(base["qtr"]) == 4 else x
            for x in states
        ]
        outer = not in_evaluator[0]
        if outer:
            in_evaluator[0] = True
        try:
            wps = original_eval(base, transformed)
            assert len(wps) == len(states)
            if outer:
                for row, wp in zip(transformed,wps):
                    captures.append({
                        "possession_original":bool(row["_possession_original"]),
                        "score_change":float(row["_score_change_original"]),
                        "wp":float(wp),
                        "yardline":float(row["yardline_100"]),
                        "down":float(row["down"]),
                        "is_kickoff":bool(row["is_kickoff"]),
                    })
            return wps
        finally:
            if outer:
                in_evaluator[0] = False
    private["evaluate_post_play_states"]=inspect
    def sim(action: str):
        captures.clear()
        rng=np.random.default_rng(43)
        if action=="FIELD_GOAL":
            result=private["simulate_field_goal"](base,simulations,rng)
        elif action=="PUNT":
            result=private["simulate_punt"](base,simulations,rng)
        else:
            result=private["simulate_go"](base,action,simulations,rng)
        assert private["evaluate_post_play_states"] is inspect
        arr=pd.DataFrame(captures)
        assert len(arr)==simulations, (
            case_name,action,len(arr),simulations
        )
        branch=pd.Series(
            np.where(arr.score_change!=0, "score",
                     np.where(arr.possession_original,"retains_ball",
                              "opponent_ball"))
        )
        pieces=[]
        for key in ["score","retains_ball","opponent_ball"]:
            subset=arr.loc[branch.eq(key)]
            if len(subset):
                m=subset["yardline"].mean(skipna=True)
                pieces.append(
                    f"{key}:{len(subset)}"
                    f" ({len(subset)/len(arr):.1%}),"
                    f"WP={subset.wp.mean():.3f},"
                    f"field={m:.1f}"
                )
        print(f"{case_name} {action} WP={float(result['q_value']):.5f} "
              f"SE={100*float(result['mc_se']):.2f}pp "
              + "; ".join(pieces))
        return result
    print("== "+case_name+" ==")
    base_wp=float(engine["current_win_probability"](base))
    print(f"pre-decision WP={base_wp:.5f}, site={base['site']}, "
          f"WP home-axis={base['_wp_original_on_home_axis']}, "
          f"home_diff={base['home_score_differential']}, "
          f"original score_diff={base['score_differential']}")
    if int(base["qtr"]) == 4:
        fg_probs=engine["field_goal_probabilities"](base)
        print("FG outcome distribution:", fg_probs)
    for action in ["NORMAL_GO_PASS","NORMAL_GO_RUN"]:
        p=engine["go_conversion_probability"](base, action)
        print(f"{action} predicted conversion={p:.4f}")
    for act in ["NORMAL_GO_PASS","NORMAL_GO_RUN","FIELD_GOAL","PUNT"]:
        eligible,reason=engine["action_eligible"](base,act)
        if eligible:
            sim(act)
        else:
            print(f"{case_name} {act} ineligible: {reason}")
    assert engine["recommend"] is original_recommend
    assert engine["predict_original_team_wp"] is original_wp


def main():
    engine=historical_support()
    scenarios=[
        ("own5 Q1 NEUTRAL", dict(
            qtr=1,game_seconds_remaining=3030,
            yardline_100=95,ydstogo=3,score_differential=0,
            posteam_timeouts_remaining=3,defteam_timeouts_remaining=3,
            site="NEUTRAL",roof="outdoors"
        )),
    ]
    for site in ("AWAY","NEUTRAL","HOME"):
        scenarios.append((
            f"Q4 4:59 opponent6 {site}",
            dict(qtr=4,game_seconds_remaining=299,
                 yardline_100=6,ydstogo=4,score_differential=-4,
                 posteam_timeouts_remaining=3,defteam_timeouts_remaining=3,
                 site=site,roof="outdoors")
        ))
    for name,st in scenarios:
        analyze_branches(engine,name,st)
    print("PASS: historical support and action-branch diagnostics completed. "
          "These statistics are descriptive, not independent predictive or causal validation.")


if __name__=="__main__":
    main()
