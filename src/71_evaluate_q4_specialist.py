"""Step 71: a Q4 last-five-minute specialist WP model, tested out of time.

2024 is the first validation year, 2025 exploratory as already inspected.
No 2026 data are loaded, and no live model files or Streamlit are modified.
The current 2014-22 production model is the baseline. Candidate uses
the SAME 14 preplay inputs and football-monotone fit from Step 67,
but trains on Q4 <=300-second scrimmage states only.

Scoring metric is home-win Brier and game-cluster confidence interval.
Changing conditional WIN calibration is not fourth-down causal policy
validation. Separate touchdown/field-goal simulations must be validated
before any promotion.
"""
from __future__ import annotations
import importlib.util
import json
import os
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from src.wp_monotonic_candidate import (
    ConstrainedRegulationWP, fit_binary_wp, home_advantage_matrix,
)

SCRIPT=ROOT/"src/67_train_modern_wp_candidate.py"
REPORT=ROOT/"artifacts/q4_last_five_wp_candidate_report.json"
TRAIN_SPLITS=((2023,2024),(2024,2025))


def load_step67():
    spec=importlib.util.spec_from_file_location("frozen_step67_wp_inputs",SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("Step 67 missing")
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def last_five(data:pd.DataFrame)->pd.DataFrame:
    """Pre-registered Q4 <=300s; no labels used for selection."""
    needed={"qtr","game_seconds_remaining","game_id","season","final_home_margin"}
    if missing:=needed.difference(data.columns):
        raise ValueError(f"Missing historical inputs: {sorted(missing)}")
    if data.empty or not data["season"].between(2014,2025).all():
        raise ValueError("Only completed 2014-25 history is permitted")
    return data.loc[
        data["qtr"].eq(4)&data["game_seconds_remaining"].between(1,300)
    ].copy()


def test_split(data:pd.DataFrame,train_end:int,valid_year:int):
    if (train_end,valid_year) not in TRAIN_SPLITS:
        raise ValueError("Only fixed 2024/2025 validation splits allowed")
    tr=data.loc[data["season"].between(2014,train_end)].copy()
    val=data.loc[data["season"].eq(valid_year)].copy()
    if tr.empty or val.empty or tr["season"].max()>=val["season"].min():
        raise RuntimeError("Missing data or chronological leakage")
    if set(tr["game_id"])&set(val["game_id"]):
        raise RuntimeError("Game-ID leakage")
    return tr,val


def subgroup_metrics(name:str,frame:pd.DataFrame,base:np.ndarray,
                     pred:np.ndarray,cluster_ci)->dict:
    n=len(frame)
    if n==0:
        return {"group":name,"n":0}
    y=(frame["final_home_margin"].to_numpy(float)>0).astype(float)
    x=np.asarray(base,float)
    z=np.asarray(pred,float)
    if x.shape!=(n,) or z.shape!=(n,) or not np.isfinite(x).all() or not np.isfinite(z).all():
        raise ValueError("WP predictions are invalid")
    d=(y-x)**2-(y-z)**2
    lo,hi=cluster_ci(d,frame["game_id"].to_numpy())
    return {"group":name,"n":n,"games":int(frame["game_id"].nunique()),
            "baseline_brier":float(np.mean((y-x)**2)),
            "specialist_brier":float(np.mean((y-z)**2)),
            "improvement":float(d.mean()),
            "ci95":[lo,hi],
            "actual_home_win":float(y.mean()),
            "predicted_base":float(x.mean()),
            "predicted_specialist":float(z.mean())}


def evaluate(valid:pd.DataFrame,features:list[str],binary,frozen,cluster_ci):
    mat=valid[features].to_numpy(float)
    classes={int(c):i for i,c in enumerate(frozen.classes_)}
    if not {0,2}.issubset(classes):
        raise ValueError("Frozen multiclass WP labels missing")
    base=frozen.predict_proba(mat)[:,classes[2]]
    specialist=ConstrainedRegulationWP(binary,frozen,features)
    pred=specialist.predict_proba(mat)[:,2]
    if (not np.isfinite(pred).all() or ((pred<0)|(pred>1)).any()):
        raise ValueError("Candidate returned invalid WP")
    home=valid["is_home_posteam"].eq(1).to_numpy()
    margin=valid["home_score_differential"].to_numpy(float)
    masks=[
        ("ALL Q4 LAST FIVE",np.ones(len(valid),bool)),
        ("HOME POSSESSION",home),
        ("AWAY POSSESSION",~home),
        ("HOME TRAILING BY 1-3",(margin>=-3)&(margin<=-1)),
        ("AWAY TRAILING BY 1-3",(margin>=1)&(margin<=3)),
        ("ONE SCORE GAME",abs(margin)<=8),
        ("RECEIVING REGION OWN 20-40",
         (~home)&valid["yardline_100"].between(60,80).to_numpy()),
    ]
    return [subgroup_metrics(name,valid.loc[mask],base[mask],pred[mask],cluster_ci)
            for name,mask in masks]


def main():
    os.chdir(ROOT)
    print("=== STEP 71: FROZEN Q4 LAST-FIVE WP SPECIALIST ===",flush=True)
    print("2014-23 -> 2024, 2014-24 -> examined 2025. NEVER reads 2026.",flush=True)
    step=load_step67()
    step.assert_frozen_inputs()
    features=list(json.loads(step.BASE_SPEC.read_text())["features"])
    frozen=joblib.load(step.BASE_MODEL)
    # The existing helper loads only 2014-2025.
    raw=step.load_seasons()
    data=last_five(step.usable_regulation_states(raw,features))
    del raw
    report={"scope":"Q4 final five minutes, regulation scrimmage states only",
            "2025":"exploratory; already previously inspected",
            "2026":"unopened",
            "promotion":"NOT AUTHORIZED",
            "baseline":"frozen 2014-22 model",
            "folds":[]}
    for train_end,valid_year in TRAIN_SPLITS:
        tr,val=test_split(data,train_end,valid_year)
        print(f"Train <= {train_end}: {len(tr):,} states; "
              f"evaluate {valid_year}: {len(val):,} states.",flush=True)
        trained=fit_binary_wp(
            tr[features].to_numpy(float),
            tr["final_home_margin"].to_numpy(float),
            features,tr["game_id"].to_numpy())
        rows=evaluate(val,features,trained,frozen,step.cluster_brier_ci)
        report["folds"].append({"train_through":train_end,
                                "validation_year":valid_year,"metrics":rows})
        for row in rows:
            if not row["n"]:
                print(f"  {row['group']}: no data",flush=True)
            else:
                print(f"  {row['group']} n={row['n']} "
                      f"Brier base={row['baseline_brier']:.5f} "
                      f"candidate={row['specialist_brier']:.5f} "
                      f"delta={row['improvement']:+.5f} "
                      f"game-CI={row['ci95']}",flush=True)
    REPORT.parent.mkdir(parents=True,exist_ok=True)
    REPORT.write_text(json.dumps(report,indent=2))
    print("No production change and no 2026 data accessed.",flush=True)


if __name__=="__main__":
    main()
