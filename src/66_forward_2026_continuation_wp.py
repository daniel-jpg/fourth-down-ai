"""Step 66: frozen prospective kickoff-continuation validation on 2026 W6-18.

PRE-REGISTERED BEFORE EXAMINING 2026 W6+ OUTCOMES.
Only ordinary, regulation Q4, standard-own-35 kickoffs with >40 seconds
remaining count toward this evaluation. Their *actual receiving field position*
is NEVER supplied to any forecast. For each kickoff, the candidate integrates
the frozen WP model over the empirical, REG Q4 standard-35 kickoff outcomes
observed in 2025 only. No 2026 data enters the donor distribution or model.

Baseline: existing engine's synthetic kickoff representation (missing
down, ydstogo, yardline_100; goal_to_go=0).
Comparator: training-aligned/native kickoff representation.
Candidate: mean post-return/touchback first-down WP, with field position and
return clock drawn ONLY from the 2025 donor distribution.
First-snap WP is a HINDSIGHT diagnostic, not a pre-kickoff forecast.

No tuning or production changes allowed after inspecting 2026 W6-18 data.
Even a positive result supports ordinary-kickoff WP only: it does NOT prove
fourth-down counterfactual correctness, and omits onside/special outcomes.
WP model artifact and feature spec are pinned by Git blob hash.

Until at least 100 unique prospectively evaluable games exist, this script
prints availability ONLY, not Brier scores, to avoid serial peeking.
Once the gate is reached, treat the result as ONE frozen evaluation and do not
change the candidate in response to it.

Run from repository root after enough 2026 games have finished:
  ./.venv/bin/python src/66_forward_2026_continuation_wp.py
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import runpy
import sys

import nflreadpy as nfl
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

STEP61 = ROOT / "src/61_audit_observed_kickoff_wp.py"
STEP64 = ROOT / "src/64_validate_2025_kickoff_encoding.py"
STEP65 = ROOT / "src/65_crossfit_2025_continuation_wp.py"

# Fixed before 2026 Week 6+ outcomes. Use the same model as Step 65.
PINNED_GIT_BLOBS = {
    "models/win_probability_model.joblib":
        "4cdafd1687594ff624560ad140725d9c32d72a17",
    "data/win_probability_spec.json":
        "857ef77f7ca0a041502ed28811fdcbf48a5d6e95",
}
MIN_PROSPECTIVE_GAMES = 100
FIRST_WEEK = 6
LAST_WEEK = 18
MIN_CLOCK = 40.0
SCRIMMAGE = {"run", "pass", "no_play", "qb_kneel", "qb_spike"}


def git_blob_sha(path: Path) -> str:
    """Compute Git SHA-1 object ID, not SHA-1 of raw file bytes."""
    data = path.read_bytes()
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def enforce_frozen_artifacts(root: Path = ROOT) -> None:
    for relative, pinned in PINNED_GIT_BLOBS.items():
        path = root / relative
        if not path.exists() or git_blob_sha(path) != pinned:
            raise RuntimeError(
                f"Frozen WP artifact changed: {relative}. "
                "Do not re-evaluate on an altered WP model/spec."
            )


def frozen_2025_donors(pbp: pd.DataFrame, helper64, helper65) -> pd.DataFrame:
    """Construct 2025-only donors, discarding final game outcome information."""
    if pbp.empty or not pbp["season"].eq(2025).all():
        raise ValueError("Donors must come solely from 2025.")
    pairs = helper64["ordinary_2025_q4_pairs"](pbp)
    train = helper65["with_kickoff_weeks"](pairs, pbp)
    if train.empty or not train["week"].between(1, 18).all():
        raise ValueError("Need 2025 REG ordinary Q4 donor games.")
    own = 100 - pd.to_numeric(train["yardline_100_snap"], errors="coerce")
    clock = (
        pd.to_numeric(train["game_seconds_remaining_kick"], errors="coerce")
        - pd.to_numeric(train["game_seconds_remaining_snap"], errors="coerce")
    )
    tb = pd.to_numeric(train["touchback"], errors="coerce").eq(1)
    if (not own.between(1, 99).all()
            or not clock.between(0, 35).all()
            or not own.loc[tb].isin([20, 35]).all()
            or not clock.loc[~tb].between(1, 35).all()
            or int(tb.sum()) < 15 or int((~tb).sum()) < 15):
        raise ValueError("2025 donor outcomes have invalid ordinary kickoff states.")
    return pd.DataFrame({
        "own": own.to_numpy(float),
        "elapsed": clock.to_numpy(float),
        "touchback": tb.to_numpy(bool),
    })


def prospective_2026_pairs(pbp: pd.DataFrame, fields: list[str]) -> pd.DataFrame:
    """Match 2026 W6+ ordinary kickoffs to first snap; no W1-5 rows.

    Empty result means not yet published, not an evaluation failure.
    Never use 2026 data as donors; only season and week filters.
    """
    required = set(fields) | {
        "game_id", "season", "week", "season_type", "play_id", "play_type",
        "touchback", "desc",
    }
    if missing := required.difference(pbp.columns):
        raise ValueError(f"Missing 2026 PBP columns: {sorted(missing)}")
    if pbp.empty or not pbp["season"].eq(2026).all():
        raise ValueError("Validation PBP must be 2026 only.")

    weeks = pd.to_numeric(pbp["week"], errors="coerce")
    later = pbp.loc[
        weeks.between(FIRST_WEEK, LAST_WEEK)
        & pbp["season_type"].eq("REG")
    ].copy()
    if later.empty:
        return pd.DataFrame()
    later = later.sort_values(["game_id", "play_id"]).reset_index(drop=True)
    later["kick_seq"] = (
        later["play_type"].eq("kickoff")
        .groupby(later["game_id"]).cumsum()
    )
    join = ["game_id", "kick_seq"]
    kick = later.loc[later["play_type"].eq("kickoff"), [
        *join, "week", "play_id", "desc", "touchback", *fields,
    ]].copy()
    snap = later.loc[
        later["play_type"].isin(SCRIMMAGE)
        & later["posteam"].notna() & later["down"].notna()
        & later["yardline_100"].notna(),
        [*join, "play_id", *fields],
    ].drop_duplicates(join).copy()
    if kick.empty or snap.empty:
        return pd.DataFrame()

    pairs = kick.merge(
        snap, on=join, how="inner", suffixes=("_kick", "_snap"),
        validate="one_to_one",
    )
    if pairs.empty:
        return pairs
    clock = (
        pd.to_numeric(pairs["game_seconds_remaining_kick"], errors="coerce")
        - pd.to_numeric(pairs["game_seconds_remaining_snap"], errors="coerce")
    )
    own = 100 - pd.to_numeric(pairs["yardline_100_snap"], errors="coerce")
    is_tb = pd.to_numeric(pairs["touchback"], errors="coerce").eq(1)
    special = pairs["desc"].fillna("").str.contains(
        "onside|penalty", case=False, regex=True
    )
    # Require complete games with binary final-result labels. No partial
    # live-game rows, tied finals, or score-changing intervening events.
    result = pd.to_numeric(pairs["result_kick"], errors="coerce")
    eligible = (
        pairs["qtr_kick"].eq(4) & pairs["qtr_snap"].eq(4)
        & pairs["game_seconds_remaining_kick"].gt(MIN_CLOCK)
        & pairs["play_id_snap"].gt(pairs["play_id_kick"])
        & pairs["posteam_kick"].notna()
        & pairs["posteam_kick"].eq(pairs["posteam_snap"])
        & pd.to_numeric(pairs["yardline_100_kick"], errors="coerce").eq(35)
        & clock.between(0, 35) & own.between(1, 99)
        & ((is_tb & own.isin([20, 35])) | (~is_tb & clock.between(1, 35)))
        & result.notna() & result.ne(0)
        & ~special
    )
    result_pairs = pairs.loc[eligible].copy().reset_index(drop=True)
    if result_pairs.empty:
        return result_pairs
    # A kickoff must not change the score; no intervening score or
    # possession flip is allowed before the receiving team's first snap.
    def home_margin(frame, suffix):
        raw_margin = (
            pd.to_numeric(frame[f"posteam_score_{suffix}"], errors="coerce")
            - pd.to_numeric(frame[f"defteam_score_{suffix}"], errors="coerce")
        )
        home_has_ball = frame[f"posteam_{suffix}"].eq(frame[f"home_team_{suffix}"])
        return np.where(home_has_ball, raw_margin, -raw_margin)

    equal_score = np.isclose(
        home_margin(result_pairs, "kick"),
        home_margin(result_pairs, "snap"), rtol=0, atol=0.01,
    )
    same_final = np.isclose(
        pd.to_numeric(result_pairs["result_kick"], errors="coerce"),
        pd.to_numeric(result_pairs["result_snap"], errors="coerce"),
        rtol=0, atol=0,
    )
    result_pairs = result_pairs.loc[equal_score & same_final].reset_index(drop=True)
    if result_pairs.empty:
        return result_pairs
    wk = pd.to_numeric(result_pairs["week"], errors="coerce")
    if not wk.between(FIRST_WEEK, LAST_WEEK).all():
        raise AssertionError("2026 W1-5 leakage into prospective validation.")
    result_pairs["week"] = wk.astype(int)
    if result_pairs.groupby("game_id")["week"].nunique().gt(1).any():
        raise ValueError("A game appears in multiple weeks.")
    return result_pairs


def prospective_predictions(pairs: pd.DataFrame, donors: pd.DataFrame,
                            wp_model, features: list[str], helper61,
                            helper65) -> pd.DataFrame:
    """All three forecasts use only at-kickoff information and frozen donors."""
    if pairs.empty or not pairs["week"].between(FIRST_WEEK, LAST_WEEK).all():
        raise ValueError("No prospective 2026 W6-18 ordinary kickoffs.")
    native = helper61["feature_frame"](pairs, "kick")
    snap = helper61["feature_frame"](pairs, "snap")
    if (not native["yardline_100"].eq(35).all()
            or not native["ydstogo"].eq(0).all()
            or native["down"].notna().any()
            or not native["goal_to_go"].eq(0).all()):
        raise ValueError("2026 kickoff feature conventions changed; stop evaluation.")

    original = native.copy()
    for field in ("down", "ydstogo", "yardline_100"):
        original[field] = np.nan
    original["goal_to_go"] = 0.0
    predict = helper61["model_probabilities"]
    y = pd.to_numeric(pairs["result_kick"], errors="coerce").to_numpy(float)
    if not np.isfinite(y).all() or (y == 0).any():
        raise ValueError("Only completed non-tied games may be evaluated.")
    home_receiver = native["is_home_posteam"].to_numpy(bool)
    home_margin = native["home_score_differential"].to_numpy(float)
    forecast = pd.DataFrame({
        "game_id": pairs["game_id"].to_numpy(),
        "week": pairs["week"].to_numpy(int),
        "seconds": native["game_seconds_remaining"].to_numpy(float),
        "home_receives": home_receiver,
        "receiver_margin": np.where(home_receiver, home_margin, -home_margin),
        "actual_home_win": (y > 0).astype(float),
        "engine_pending": predict(wp_model, features, original),
        "native_pending": predict(wp_model, features, native),
        "continuation": helper65["expected_first_snap_home_wp"](
            wp_model, features, native, donors,
        ),
        "oracle_first_snap": predict(wp_model, features, snap),
    })
    return forecast


def frozen_gate(scores: pd.DataFrame, helper65) -> str:
    """A conservative RESEARCH gate; never authorizes production deployment."""
    if scores["game_id"].nunique() < MIN_PROSPECTIVE_GAMES:
        return "NOT READY: fewer than 100 distinct completed games."
    y = scores["actual_home_win"].to_numpy(float)
    games = scores["game_id"].to_numpy()
    candidate = scores["continuation"].to_numpy(float)
    engine = scores["engine_pending"].to_numpy(float)
    native = scores["native_pending"].to_numpy(float)
    gain_engine = (y-engine)**2 - (y-candidate)**2
    gain_native = (y-native)**2 - (y-candidate)**2
    lo_engine, _ = helper65["game_cluster_ci"](gain_engine, games)
    lo_native, _ = helper65["game_cluster_ci"](gain_native, games)
    subgroups_ok = all(
        not scores.loc[mask].empty and float(np.mean(
            (y[mask] - engine[mask])**2 - (y[mask] - candidate[mask])**2
        )) >= 0
        for mask in (
            ~scores["home_receives"].to_numpy(bool),
            scores["home_receives"].to_numpy(bool),
        )
    )
    if lo_engine > 0 and lo_native > 0 and subgroups_ok:
        return ("RESEARCH GATE PASS: all prespecified Brier conditions met. "
                "Still NOT sufficient to enable FG/GO production logic.")
    return ("RESEARCH GATE FAIL/INCONCLUSIVE: insufficient consistent evidence "
            "to replace kickoff WP forecasts.")


def main() -> None:
    os.chdir(ROOT)
    print("\n=== STEP 66: FROZEN 2026 W6-18 FORWARD VALIDATION ===")
    print("Only 2025 donors; no adaptation to 2026 W1-5 or later outcomes.")
    print("No production/model changes; no data written.")
    enforce_frozen_artifacts()
    print("Frozen WP model and feature spec: verified exact Git blob hashes.")
    helper61 = runpy.run_path(str(STEP61))
    helper64 = runpy.run_path(str(STEP64))
    helper65 = runpy.run_path(str(STEP65))

    pbp26 = nfl.load_pbp([2026]).sort(["game_id", "play_id"]).to_pandas()
    pairs26 = prospective_2026_pairs(pbp26, helper61["FIELDS"])
    n_games = 0 if pairs26.empty else pairs26["game_id"].nunique()
    print(f"2026 W6-18 completed, eligible Q4 kicks: {len(pairs26)} "
          f"from {n_games} games.")
    if n_games < MIN_PROSPECTIVE_GAMES:
        print(f"NOT READY: need >= {MIN_PROSPECTIVE_GAMES} unique games. "
              "No forecasts or Brier metrics computed or displayed.")
        print("Run ONCE after the game-count threshold is met; do not tune.")
        return

    pbp25 = nfl.load_pbp([2025]).sort(["game_id", "play_id"]).to_pandas()
    donors = frozen_2025_donors(pbp25, helper64, helper65)
    print(f"Frozen 2025 donor sample: {len(donors)} ordinary Q4 kicks; "
          f"touchbacks={donors['touchback'].mean():.2%}.")
    engine = runpy.run_path(str(ROOT / "src/42_build_decision_engine.py"))
    forecasts = prospective_predictions(
        pairs26, donors, engine["wp_model"], list(engine["WP_FEATURES"]),
        helper61, helper65,
    )
    helper65["summarize"]("FORWARD 2026 ALL", forecasts)
    helper65["summarize"](
        "FORWARD 2026 KICKER HOME",
        forecasts.loc[~forecasts["home_receives"]],
    )
    helper65["summarize"](
        "FORWARD 2026 KICKER AWAY",
        forecasts.loc[forecasts["home_receives"]],
    )
    print("\nPREDECLARED RESEARCH DECISION: ", frozen_gate(forecasts, helper65))
    print("DO NOT deploy. This only evaluates ordinary kickoffs, not special "
          "teams events, FG/GO counterfactuals or decision utility.")


if __name__ == "__main__":
    main()
