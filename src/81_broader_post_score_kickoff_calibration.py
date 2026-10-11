"""Step 81: broader real-outcome Q4 post-score kickoff calibration.

FIXED SCOPE, before seeing results:
- REG 2023/2024/2025, Q4, clock >40 seconds and <=900 seconds.
- Real standard-own-35 kickoffs with a clearly linked PRECEDING
  made field goal, extra point, or 2-point attempt by the kicking team.
- Exclude announced onside attempts and kickoffs after another intervening
  kickoff (re-kicks). DO NOT select based on touchback, penalties, return
  TDs, next scrimmage snap, or final game outcome.
- Compare baseline engine's synthesized Q4 post-score kickoff features
  (down, ydstogo, yardline_100 all missing, goal_to_go=0) against the
  fixed native candidate (down missing, ydstogo=0, yardline_100=35,
  goal_to_go=0). Both use SAME frozen 2014-22-trained WP classifier.
- Score P(home win) against actual final home win. Final result is
  LABEL ONLY, never an input feature. Ties are home-win=0.
- Paired Brier difference, game-cluster 95% bootstrap CI for overall
  and annual primary results. All subgroups descriptive, never tuned.

Outcomes for 2023-25 have been inspected in other project studies,
especially 2025. This is an exploratory retrospective RELIABILITY check,
not a fresh holdout, causal FG-vs-GO validation, or promotion gate.
2026 game data must remain unopened.
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys

import nflreadpy as nfl
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.streamlit_kickoff_preview import isolated_engine_for_preview
from src.kickoff_q4_native_features_candidate import native_q4_normal_kickoff

YEARS = (2023, 2024, 2025)
OUTPUT = ROOT / "artifacts/81_broad_q4_native_kickoff_calibration.json"
SNAPSHOT = ROOT / "artifacts/81_broad_q4_kickoff_scored_rows.csv"
EARLIER = ROOT / "src/61_audit_observed_kickoff_wp.py"
CLOCK_FLOOR_EXCLUSIVE = 40.0
KICKOFF_OWN_35 = 35.0
BOOTSTRAP_REPEATS = 1500
# Most scoring drive/try actions are identifiable as field_goal/extra_point
# or as a two-point play with the official pre-snap two_point_attempt flag.
PREVIOUS_ACTIONS = {
    "run", "pass", "qb_kneel", "qb_spike", "punt", "field_goal",
    "extra_point", "two_point_conversion",
}
REQUIRED = {
    "season", "season_type", "game_id", "play_id", "play_type",
    "qtr", "posteam", "home_team", "away_team", "result",
    "game_seconds_remaining", "half_seconds_remaining",
    "down", "ydstogo", "yardline_100", "goal_to_go",
    "posteam_score", "defteam_score",
    "posteam_timeouts_remaining", "defteam_timeouts_remaining",
    "field_goal_result", "desc",
}


def import_2024_feature_helpers():
    spec = importlib.util.spec_from_file_location(
        "historical_native_kickoff_input_61", EARLIER
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Missing previously verified historic WP features.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def historical_post_score_kickoffs(raw: pd.DataFrame) -> pd.DataFrame:
    """Select ONLY pre-kickoff identifiable states, NOT realized outcomes."""
    if raw.empty:
        raise ValueError("Empty NFL play-by-play.")
    if missing := REQUIRED.difference(raw.columns):
        raise ValueError(f"Missing historical PBP fields: {sorted(missing)}")
    if not raw["season"].isin(YEARS).all():
        raise ValueError("Only 2023-25 completed seasons are allowed.")
    df = raw.sort_values(["game_id", "play_id"]).reset_index(drop=True).copy()
    if df.duplicated(["game_id", "play_id"]).any():
        raise ValueError("Duplicate game/play IDs risk wrong scoring attribution.")
    game = df["game_id"]
    substantive = df["play_type"].isin(PREVIOUS_ACTIONS)

    def prior_action(col: str) -> pd.Series:
        # Shift INSIDE games, so kickoff's own row is never a prior action.
        return df[col].where(substantive).groupby(game).transform(
            lambda group: group.ffill().shift(1)
        )

    last_kind = prior_action("play_type")
    last_team = prior_action("posteam")
    last_fg = prior_action("field_goal_result")
    last_id = pd.to_numeric(prior_action("play_id"), errors="coerce")

    last_two = (
        pd.to_numeric(prior_action("two_point_attempt"), errors="coerce")
        if "two_point_attempt" in df.columns
        else pd.Series(0., index=df.index)
    )
    last_kick_id = pd.to_numeric(
        df["play_id"].where(df["play_type"].eq("kickoff"))
        .groupby(game).transform(lambda group: group.ffill().shift(1)),
        errors="coerce",
    )
    # A second kickoff before any scrimmage snap is a re-kick: never
    # attribute that second kickoff to the earlier scoring action.
    is_first_kick_after_score = (
        last_kick_id.isna() | last_id.gt(last_kick_id)
    )
    fg_made = (
        last_kind.eq("field_goal")
        & last_fg.fillna("").astype(str).str.lower().eq("made")
    )
    following_xp = last_kind.eq("extra_point")
    following_two = (
        last_kind.eq("two_point_conversion")
        | (last_kind.isin(["run", "pass"]) & last_two.eq(1))
    )
    prev_score_type = np.select(
        [fg_made, following_xp, following_two],
        ["made_fg", "post_extra_point", "post_two_point"],
        default="unknown",
    )
    is_onside = (
        df["desc"].fillna("").astype(str)
        .str.contains("onside", case=False, regex=False)
    )
    clock = pd.to_numeric(df["game_seconds_remaining"], errors="coerce")
    ownspot = pd.to_numeric(df["yardline_100"], errors="coerce")
    eligible = (
        df["season_type"].eq("REG")
        & df["play_type"].eq("kickoff")
        & df["qtr"].eq(4)
        & clock.gt(CLOCK_FLOOR_EXCLUSIVE)
        & clock.le(900)
        & ownspot.eq(KICKOFF_OWN_35)
        & ~is_onside
        & is_first_kick_after_score
        & df["posteam"].notna()
        & last_team.notna()
        & df["posteam"].ne(last_team)
    )
    # Require the receiving team to be one of the current game's teams.
    receivers = (
        df["posteam"].eq(df["home_team"])
        | df["posteam"].eq(df["away_team"])
    )
    # Keep every scoring-kick outcome (penalty, return TD, future snap absent).
    # The prior known score type, not realized later events, defines the cohort.
    eligible = eligible & receivers & (prev_score_type != "unknown")
    result = df.loc[eligible].copy()
    result["post_score_type"] = prev_score_type[eligible]
    result["prior_scoring_action_id"] = last_id[eligible].to_numpy(float)
    result["original_kicking_team"] = last_team[eligible].to_numpy()
    result["kicker_site"] = np.where(
        result["original_kicking_team"].eq(result["home_team"]),
        "HOME", "AWAY"
    )
    if not result["original_kicking_team"].eq(
        np.where(
            result["kicker_site"].eq("HOME"),
            result["home_team"],
            result["away_team"],
        )
    ).all():
        raise ValueError("Kicking team cannot be assigned a valid game site.")
    if result.duplicated(["game_id", "play_id"]).any():
        raise ValueError("Kickoff scoring sample has duplicate event IDs.")
    return result.reset_index(drop=True)


def kickoff_model_features(
    selected: pd.DataFrame, feature_helper,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Construct baseline, fixed candidate and genuine PBP feature frames.

    Original and candidate differ in exactly TWO scalar inputs. Use
    the existing historical feature encoder so scores, timeouts and
    home/away orientation match the original model's contract.
    """
    if selected.empty:
        raise ValueError("No pre-score declared normal kickoff samples.")
    if not selected["yardline_100"].eq(KICKOFF_OWN_35).all():
        raise ValueError("All candidate donor kickoff spots must be own 35.")
    fields = feature_helper.FIELDS
    if missing := set(fields).difference(selected.columns):
        raise ValueError(f"Selected kickoff features missing {sorted(missing)}")
    adapted = selected.rename(columns={
        col: f"{col}_kick" for col in fields
    }).copy()
    native = feature_helper.feature_frame(adapted, "kick")
    baseline = native.copy()
    for key in ("down", "ydstogo", "yardline_100"):
        baseline[key] = np.nan
    baseline["goal_to_go"] = 0.0
    candidate = baseline.copy()
    candidate["ydstogo"] = 0.0
    candidate["yardline_100"] = KICKOFF_OWN_35
    # Exact representation match to the separately implemented Step 80
    # private-engine candidate for a normal Q4 post-score kickoff.
    probe = native_q4_normal_kickoff({
        "qtr":4, "is_kickoff":1.0,
        "_score_change_original":3.0,
        "game_seconds_remaining":float(native["game_seconds_remaining"].iloc[0]),
        "down":np.nan,"ydstogo":np.nan,"yardline_100":np.nan,
        "goal_to_go":0.0,
    })
    if (
        probe["ydstogo"] != candidate["ydstogo"].iloc[0]
        or probe["yardline_100"] != candidate["yardline_100"].iloc[0]
    ):
        raise AssertionError("Historical candidate does not match Step 80.")
    for key in ("qtr", "game_seconds_remaining", "half_seconds_remaining",
                "home_score_differential", "home_timeouts_remaining",
                "away_timeouts_remaining", "is_home_posteam", "is_kickoff"):
        if not np.allclose(
            baseline[key].to_numpy(dtype=float),
            candidate[key].to_numpy(dtype=float),
            equal_nan=True,
        ):
            raise AssertionError(f"Candidate changed protected kickoff {key}")
    return baseline, candidate, native


def score_candidate(kicks: pd.DataFrame, engine: dict,
                    feature_helper) -> pd.DataFrame:
    """Final game result is label ONLY, never an inference-time feature."""
    baseline, candidate, native = kickoff_model_features(kicks, feature_helper)
    predict = feature_helper.model_probabilities
    model, features = engine["wp_model"], list(engine["WP_FEATURES"])
    a = predict(model, features, baseline)
    b = predict(model, features, candidate)
    real = predict(model, features, native)
    final_margin = pd.to_numeric(kicks["result"], errors="coerce").to_numpy(float)
    if not np.isfinite(final_margin).all():
        raise ValueError("Final outcome labels missing; do not impute.")
    game_result = (final_margin > 0).astype(float)
    receiver_home = native["is_home_posteam"].to_numpy(dtype=bool)
    home_margin = native["home_score_differential"].to_numpy(float)
    recv_margin = np.where(receiver_home, home_margin, -home_margin)
    elapsed = native["game_seconds_remaining"].to_numpy(float)
    out = pd.DataFrame({
        "season": kicks["season"].to_numpy(int),
        "game_id": kicks["game_id"].astype(str).to_numpy(),
        "kick_play_id": kicks["play_id"].to_numpy(float),
        "post_score_type": kicks["post_score_type"].astype(str).to_numpy(),
        "kicker_site": kicks["kicker_site"].astype(str).to_numpy(),
        "seconds": elapsed,
        "receiver_margin": recv_margin,
        "receiver_score_context": np.select(
            [recv_margin < -3, recv_margin < 0, recv_margin == 0,
             recv_margin <= 3, recv_margin > 3],
            ["trails_4plus", "trails_1to3", "tied",
             "leads_1to3", "leads_4plus"],
            default="missing",
        ),
        "clock_bucket": np.select(
            [elapsed <= 300, elapsed <= 600, elapsed <= 900],
            ["Q4_last5_gt40", "Q4_5to10", "Q4_10to15"],
            default="missing",
        ),
        "actual_home_win": game_result,
        "p_synthetic": a,
        "p_native_candidate": b,
        "p_native_pbp": real,
        "native_goal_to_go_missing": native["goal_to_go"].isna().to_numpy(bool),
    })
    if (
        len(out) != len(kicks)
        or out["receiver_score_context"].eq("missing").any()
        or out["clock_bucket"].eq("missing").any()
        or out[["p_synthetic","p_native_candidate","p_native_pbp"]].isna().any().any()
        or not out[["p_synthetic","p_native_candidate","p_native_pbp"]].apply(
            lambda x:x.between(0,1).all()
        ).all()
    ):
        raise RuntimeError("Kickoff WP calibration result invalid.")
    return out


def game_cluster_ci(delta: np.ndarray, games: np.ndarray, *,
                    repeats: int = BOOTSTRAP_REPEATS,
                    seed: int = 81) -> tuple[float,float]:
    """Resample GAMES, not multiple same-game kickoffs independently."""
    delta = np.asarray(delta, dtype=float)
    games = np.asarray(games)
    if not len(delta) or delta.ndim != 1 or delta.shape != games.shape:
        raise ValueError("Nonempty paired scoring residual and game IDs required.")
    if not np.isfinite(delta).all() or repeats < 100:
        raise ValueError("Finite scores and >=100 game bootstraps needed.")
    gid,inv=np.unique(games,return_inverse=True)
    totals=np.bincount(inv,weights=delta,minlength=len(gid))
    counts=np.bincount(inv,minlength=len(gid))
    rng=np.random.default_rng(seed)
    choices=rng.integers(0,len(gid),size=(repeats,len(gid)))
    weighted=totals[choices].sum(axis=1)/counts[choices].sum(axis=1)
    lo,hi=np.quantile(weighted,[.025,.975])
    return float(lo),float(hi)


def metrics(name:str, rows:pd.DataFrame, *, ci:bool=False) -> dict:
    """Report paired Brier improvements, not ex post policy superiority."""
    n=len(rows)
    out={"label":name,"n":n,"games":int(rows["game_id"].nunique())}
    if n == 0:
        out["support"]="NO DATA"
        return out
    y=rows["actual_home_win"].to_numpy(float)
    p0=rows["p_synthetic"].to_numpy(float)
    p1=rows["p_native_candidate"].to_numpy(float)
    p_pb=rows["p_native_pbp"].to_numpy(float)
    for x in (y,p0,p1,p_pb):
        if not np.isfinite(x).all():
            raise ValueError("Nonfinite calibration inputs.")
    improve=(y-p0)**2-(y-p1)**2
    eps=1e-12
    def logloss(p):
        return float(np.mean(
            -y*np.log(np.clip(p,eps,1-eps))
            -(1-y)*np.log1p(-np.clip(p,eps,1-eps))
        ))
    out.update({
        "support":"LOW SUPPORT" if n<30 or out["games"]<20 else "DESCRIPTIVE",
        "actual_home_win_share":float(y.mean()),
        "legacy_mean_home_wp":float(p0.mean()),
        "candidate_mean_home_wp":float(p1.mean()),
        "native_pbp_mean_home_wp":float(p_pb.mean()),
        "legacy_brier":float(np.mean((y-p0)**2)),
        "candidate_brier":float(np.mean((y-p1)**2)),
        "native_pbp_brier":float(np.mean((y-p_pb)**2)),
        "candidate_brier_gain_positive_is_better":float(improve.mean()),
        "legacy_logloss":logloss(p0),
        "candidate_logloss":logloss(p1),
        "candidate_logloss_gain_positive_is_better":
            float(logloss(p0)-logloss(p1)),
        "legacy_mean_calibration_residual":float(np.mean(y-p0)),
        "candidate_mean_calibration_residual":float(np.mean(y-p1)),
        "mean_absolute_candidate_change_pp":float(
            np.mean(np.abs(p1-p0))*100
        ),
    })
    if ci:
        lo,hi=game_cluster_ci(improve,rows["game_id"].to_numpy())
        out["candidate_brier_gain_game_cluster_ci95"]=[lo,hi]
        out["confidence_interval_includes_zero"]=bool(lo<=0<=hi)
    return out


def run()->None:
    print("=== STEP 81: EXPANDED OBSERVED 2023-25 Q4 POST-SCORE KICKOFF WP ===",
          flush=True)
    print("Regular season, standard-35 Q4 >40s, verified prior scoring play; "
          "ALL return TD/penalty outcomes retained.",flush=True)
    engine=isolated_engine_for_preview()
    helpers=import_2024_feature_helpers()
    all_selected=[]
    for yr in YEARS:
        print(f"Loading completed {yr} REG play-by-play...",flush=True)
        raw=nfl.load_pbp([yr]).to_pandas()
        if raw.empty or not raw["season"].eq(yr).all():
            raise RuntimeError(f"Unexpected historical data in {yr}.")
        sample=historical_post_score_kickoffs(raw)
        if sample.empty:
            raise RuntimeError(f"No identified post-score kickoffs in {yr}.")
        print(
            f"  {yr}: {len(sample)} post-score standard-35 non-onside "
            f"Q4 kicks across {sample['game_id'].nunique()} games; "
            f"score sources={sample['post_score_type'].value_counts().to_dict()}",
            flush=True,
        )
        all_selected.append(score_candidate(sample,engine,helpers))
    score=pd.concat(all_selected,ignore_index=True)
    score["game_id"]=score["season"].astype(str)+"_"+score["game_id"]
    groups=[
        ("all_three_seasons",score,True),
        *[(f"{yr}",score.loc[score["season"].eq(yr)],True)
          for yr in YEARS],
        *[(v,score.loc[score["post_score_type"].eq(v)],False)
          for v in ("made_fg","post_extra_point","post_two_point")],
        *[(v,score.loc[score["kicker_site"].eq(v)],False)
          for v in ("HOME","AWAY")],
        *[(v,score.loc[score["clock_bucket"].eq(v)],False)
          for v in ("Q4_last5_gt40","Q4_5to10","Q4_10to15")],
        *[(v,score.loc[score["receiver_score_context"].eq(v)],False)
          for v in ("trails_4plus","trails_1to3","tied",
                    "leads_1to3","leads_4plus")],
    ]
    reports=[metrics(label,rows,ci=with_ci)
             for label,rows,with_ci in groups]
    main=reports[0]
    if main["n"]<100:
        raise RuntimeError("Insufficient historical post-score kickoff evidence.")
    print("=== PAIRED BRIER RESULTS (positive gain means candidate better) ===",
          flush=True)
    for item in reports:
        if not item["n"]:
            print(f"{item['label']}: no data",flush=True)
            continue
        print(
            f"{item['label']:22} n={item['n']:4} "
            f"games={item['games']:4} "
            f"legacy={item['legacy_brier']:.5f} "
            f"candidate={item['candidate_brier']:.5f} "
            f"gain={item['candidate_brier_gain_positive_is_better']:+.5f} "
            f"mean_delta={item['mean_absolute_candidate_change_pp']:.2f}pp "
            f"CI={item.get('candidate_brier_gain_game_cluster_ci95','descriptive')}",
            flush=True,
        )
    report={
        "design":"Predeclared broad REG Q4 normal post-score kickoff reliability analysis",
        "historical_years":list(YEARS),
        "2026_game_data_used":False,
        "not_pristine_holdout":True,
        "frozen_2014_22_wp_model":True,
        "not_fourth_down_causal_policy_validation":True,
        "no_future_event_selection":True,
        "return_tds_and_penalty_outcomes_retained":True,
        "onside_predeclared_actions_excluded":True,
        "special_rekicks_excluded":True,
        "clock_floor_exclusive":CLOCK_FLOOR_EXCLUSIVE,
        "standard_kickoff_yardline_100":KICKOFF_OWN_35,
        "bootstrap":{"unit":"game","repeats":BOOTSTRAP_REPEATS,
                     "seed":81,"method":"paired cluster bootstrap"},
        "outcome":"final home win, from result label only (ties=0)",
        "cohort_count":int(len(score)),
        "unique_games":int(score["game_id"].nunique()),
        "groups":reports,
        "limitations":(
            "Post-score decisions and onside declarations are observational. "
            "The 2023-25 seasons have already been inspected; 2025 is "
            "especially reused. No candidate tuning, prospective validation "
            "or randomized FG-vs-GO outcomes. Small subgroups are descriptive "
            "and not multiple-testing corrected. Selected scoring kickoffs "
            "cannot establish counterfactual fourth-down policy quality."
        ),
        "production_promotion":"NOT AUTHORIZED",
    }
    OUTPUT.parent.mkdir(exist_ok=True,parents=True)
    OUTPUT.write_text(json.dumps(report,indent=2)+"\n")
    score.to_csv(SNAPSHOT,index=False,float_format="%.7f")
    print("Wrote",OUTPUT.relative_to(ROOT),"and",SNAPSHOT.relative_to(ROOT),
          flush=True)
    print("Production recommendation and WP artifact unchanged.",flush=True)


if __name__=="__main__":
    os.chdir(ROOT)
    run()
