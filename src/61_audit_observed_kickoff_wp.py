"""Read-only 2024 observational kickoff-versus-first-snap WP validation.

The frozen WP model was trained through 2022. This uses 2024 games only,
without fitting a model or touching app.py. Evaluation is for historical
calibration/consistency, NOT for choosing a 2025+ kickoff policy.

Run: ./.venv/bin/python src/61_audit_observed_kickoff_wp.py
"""

from pathlib import Path
import json
import os
import runpy

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIMMAGE = {"run", "pass", "no_play", "qb_kneel", "qb_spike"}
FIELDS = [
    "game_seconds_remaining", "half_seconds_remaining", "qtr", "down",
    "ydstogo", "yardline_100", "goal_to_go", "posteam_score",
    "defteam_score", "posteam_timeouts_remaining",
    "defteam_timeouts_remaining", "posteam", "home_team", "result",
]


def feature_frame(rows: pd.DataFrame, suffix: str) -> pd.DataFrame:
    """Replicate original production feature encoding from actual 2024 PBP.

    Do NOT infer scores from final result, which is label only.
    Preserve native missing values on kickoff downs/goal_to_go.
    """
    def col(key: str):
        return rows[f"{key}_{suffix}"]

    home_pos = col("posteam").eq(col("home_team")).to_numpy(dtype=bool)
    def num(key: str) -> np.ndarray:
        return pd.to_numeric(col(key), errors="coerce").to_numpy(dtype=float)

    scored = num("posteam_score") - num("defteam_score")
    own_timeout = num("posteam_timeouts_remaining")
    opp_timeout = num("defteam_timeouts_remaining")
    q = num("qtr")
    remaining = num("game_seconds_remaining")
    return pd.DataFrame({
        "game_seconds_remaining": remaining,
        "half_seconds_remaining": num("half_seconds_remaining"),
        "qtr": q,
        "down": num("down"),
        "ydstogo": num("ydstogo"),
        "yardline_100": num("yardline_100"),
        "goal_to_go": num("goal_to_go"),
        "home_score_differential": np.where(home_pos, scored, -scored),
        "home_timeouts_remaining": np.where(home_pos, own_timeout, opp_timeout),
        "away_timeouts_remaining": np.where(home_pos, opp_timeout, own_timeout),
        "is_home_posteam": home_pos.astype(float),
        "is_kickoff": np.ones(len(rows)) if suffix == "kick" else np.zeros(len(rows)),
        "is_overtime": (q >= 5).astype(float),
        "final_five_minutes": ((q == 4) & (remaining <= 300)).astype(float),
    }, index=rows.index)


def pair_2024_q4(pbp: pd.DataFrame) -> pd.DataFrame:
    """Causally match ordinary 2024 Q4 kickoffs to first eligible offensive snap.

    An intervening kickoff starts a new sequence; this prevents crossing into
    an unrelated possession. Exceptional kickoffs are excluded explicitly.
    """
    need = set(FIELDS) | {"game_id", "play_id", "play_type", "season", "desc", "touchback"}
    missing = need.difference(pbp.columns)
    if missing:
        raise ValueError(f"Missing PBP columns: {sorted(missing)}")
    if pbp.empty or not pbp["season"].eq(2024).all():
        raise ValueError("Only 2024 validation-season PBP is permitted.")
    df = pbp.sort_values(["game_id", "play_id"]).reset_index(drop=True).copy()
    df["kick_seq"] = df["play_type"].eq("kickoff").groupby(df["game_id"]).cumsum()
    k = df.loc[df["play_type"].eq("kickoff")].copy()
    s = df.loc[
        df["play_type"].isin(SCRIMMAGE)
        & df["down"].notna() & df["yardline_100"].notna()
        & df["posteam"].notna()
    ].drop_duplicates(["game_id", "kick_seq"]).copy()
    k = k[["game_id", "kick_seq", "play_id", "desc", "touchback", *FIELDS[:-1], "result"]]
    s = s[["game_id", "kick_seq", "play_id", *FIELDS]]
    pairs = k.merge(s, on=["game_id", "kick_seq"], how="inner",
                    suffixes=("_kick", "_snap"), validate="one_to_one")
    # KICKOFF postseason can still be Q4. They are included as held-out 2024 games.
    elapsed = (pd.to_numeric(pairs["game_seconds_remaining_kick"], errors="coerce")
               - pd.to_numeric(pairs["game_seconds_remaining_snap"], errors="coerce"))
    own = 100 - pd.to_numeric(pairs["yardline_100_snap"], errors="coerce")
    is_tb = pd.to_numeric(pairs["touchback"], errors="coerce").fillna(0).eq(1)
    onside = pairs["desc"].fillna("").str.contains("onside", case=False, regex=False)
    ordinary = (is_tb & own.isin([20, 30])) | (~is_tb & elapsed.between(1, 35))
    eligible = (
        pairs["qtr_kick"].eq(4) & pairs["qtr_snap"].eq(4)
        & pairs["play_id_snap"].gt(pairs["play_id_kick"])
        & elapsed.between(0, 35) & own.between(1, 99)
        & pairs["posteam_kick"].notna()
        & pairs["posteam_kick"].eq(pairs["posteam_snap"])
        & ~onside & ordinary
    )
    out = pairs.loc[eligible].copy().reset_index(drop=True)
    if out.empty:
        raise ValueError("No valid 2024 ordinary Q4 kickoff pairs.")
    kfeat = feature_frame(out, "kick")
    sfeat = feature_frame(out, "snap")
    same_scores = np.isclose(
        kfeat["home_score_differential"], sfeat["home_score_differential"],
        atol=0.01, rtol=0,
    )
    out = out.loc[same_scores].reset_index(drop=True)
    if out.empty:
        raise ValueError("No matched kickoffs preserve scores through the first snap.")
    return out


def model_probabilities(model, features: list[str], rows: pd.DataFrame) -> np.ndarray:
    if missing := set(features).difference(rows.columns):
        raise ValueError(f"Missing model features: {sorted(missing)}")
    ix = {int(c): i for i, c in enumerate(model.classes_)}
    if 2 not in ix:
        raise ValueError("Model lacks home-win class 2.")
    p = model.predict_proba(rows[features].to_numpy())[:, ix[2]]
    if not np.isfinite(p).all() or not ((0 <= p) & (p <= 1)).all():
        raise ValueError("Invalid learned win probabilities.")
    return np.asarray(p, dtype=float)


def compare_actual_pair_predictions(pairs, model, features) -> pd.DataFrame:
    """Predict P(home win) at each real kickoff and actual next scrimmage snap."""
    kick = feature_frame(pairs, "kick")
    snap = feature_frame(pairs, "snap")
    kickoff_native = model_probabilities(model, features, kick)
    # Emulate how production make_wp_state encodes a kickoff (goal_to_go=0).
    synthetic_kick = kick.copy()
    for f in ("down", "ydstogo", "yardline_100"):
        synthetic_kick[f] = np.nan
    synthetic_kick["goal_to_go"] = 0.0
    kickoff_synthetic = model_probabilities(model, features, synthetic_kick)
    snap_wp = model_probabilities(model, features, snap)
    game_result = pd.to_numeric(pairs["result_kick"], errors="coerce").to_numpy(float)
    if not np.isfinite(game_result).all():
        raise ValueError("Missing final game outcome labels.")
    margin = kick["home_score_differential"].to_numpy(float)
    home_receiving = kick["is_home_posteam"].to_numpy(dtype=bool)
    receiver_margin = np.where(home_receiving, margin, -margin)
    return pd.DataFrame({
        "game_id": pairs["game_id"].to_numpy(),
        "seconds": kick["game_seconds_remaining"].to_numpy(),
        "home_receiving": home_receiving,
        "receiver_margin": receiver_margin,
        "actual_home_win": (game_result > 0).astype(float),
        "p_kickoff_native": kickoff_native,
        "p_kickoff_synthetic": kickoff_synthetic,
        "p_first_snap": snap_wp,
        "kickoff_goal_to_go_missing": kick["goal_to_go"].isna().to_numpy(),
    })


def summary(scores: pd.DataFrame) -> dict:
    if scores.empty:
        raise ValueError("No scores to summarize")
    target = scores["actual_home_win"].to_numpy(float)
    native = scores["p_kickoff_native"].to_numpy(float)
    synthetic = scores["p_kickoff_synthetic"].to_numpy(float)
    after = scores["p_first_snap"].to_numpy(float)
    return {
        "n": len(scores),
        "games": int(scores["game_id"].nunique()),
        "observed_home_win_pct": float(target.mean()*100),
        "native_kick_mean_pct": float(native.mean()*100),
        "synthetic_kick_mean_pct": float(synthetic.mean()*100),
        "after_mean_pct": float(after.mean()*100),
        "native_brier": float(np.mean((native-target)**2)),
        "synthetic_brier": float(np.mean((synthetic-target)**2)),
        "after_brier": float(np.mean((after-target)**2)),
        "synthetic_minus_native_absmean_pp": float(np.mean(np.abs(synthetic-native))*100),
        "pending_minus_after_mean_pp": float(np.mean((synthetic-after))*100),
        "kick_goal_to_go_missing_pct": float(scores["kickoff_goal_to_go_missing"].mean()*100),
    }


def print_summary(label, scores):
    if scores.empty:
        print(label+": no pairs")
        return
    x = summary(scores)
    caution = " [SMALL SAMPLE]" if x["n"] < 30 else ""
    print(f"{label}: n={x['n']} games={x['games']} "+caution)
    print(f"  actual home wins {x['observed_home_win_pct']:.2f}%; "
          f"mean home WP native kickoff={x['native_kick_mean_pct']:.2f}%, "
          f"synthetic kickoff={x['synthetic_kick_mean_pct']:.2f}%, "
          f"first snap={x['after_mean_pct']:.2f}%")
    print(f"  Brier native kickoff={x['native_brier']:.5f}, "
          f"synthetic kickoff={x['synthetic_brier']:.5f}, "
          f"first snap={x['after_brier']:.5f}")
    print(f"  synthetic kickoff - first snap mean={x['pending_minus_after_mean_pp']:+.2f} pp; "
          f"mean abs synthetic-native kickoff difference={x['synthetic_minus_native_absmean_pp']:.2f} pp")


def main():
    import nflreadpy as nfl
    os.chdir(ROOT)
    print("=== REAL 2024 KICKOFF WP VALIDATION (READ ONLY) ===")
    print("Loading 2024 PBP; production WP artifact is frozen, trained through 2022.")
    pbp = nfl.load_pbp([2024]).sort(["game_id", "play_id"]).to_pandas()
    pairs = pair_2024_q4(pbp)
    print(f"Normal ordinary Q4 kickoff/first-snap pairs: {len(pairs)}")
    engine = runpy.run_path(str(ROOT / "src/42_build_decision_engine.py"))
    with open(ROOT / "data/win_probability_spec.json") as handle:
        features = json.load(handle)["features"]
    scored = compare_actual_pair_predictions(pairs, engine["wp_model"], features)
    print_summary("ALL 2024 Q4", scored)
    print(f"Native kickoff goal_to_go missing: "
          f"{scored['kickoff_goal_to_go_missing'].mean():.1%}")
    print("\n=== KICKING TEAM HOME VS AWAY ===")
    for side in ("HOME", "AWAY"):
        # 'posteam' on a kickoff is the RECEIVING team.
        sub = scored.loc[scored["home_receiving"].eq(side == "AWAY")]
        print_summary("Kicker "+side, sub)
    print("\n=== Q4 LAST FIVE MINUTES BY KICKING TEAM ===")
    q5 = scored.loc[scored["seconds"].le(300)]
    for side in ("HOME", "AWAY"):
        print_summary("Kicker "+side, q5.loc[q5["home_receiving"].eq(side == "AWAY")])
    print("\n=== Q4 LAST FIVE MINUTES, RECEIVER LEADING 1-3 ===")
    lead = q5.loc[q5["receiver_margin"].between(1, 3)]
    for side in ("HOME", "AWAY"):
        print_summary("Kicker "+side, lead.loc[lead["home_receiving"].eq(side == "AWAY")])
    print("\nINTERPRETATION: first-snap WP has more information than kickoff WP;")
    print("lower Brier alone does not establish a causal correction.")
    print("This evaluates only observed 2024 ordinary kickoffs, not 2025+ rules.")
    print("No models, production recommendations, or datasets were changed.")


if __name__ == "__main__":
    main()
