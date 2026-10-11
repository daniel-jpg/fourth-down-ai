"""READ-ONLY 2025 real-kickoff test of the frozen 2024-derived encoding fix.

This tests a PREDECLARED candidate: ordinary Q4 kickoff metadata
  down=NaN, ydstogo=0, yardline_100=35, goal_to_go=0.
This choice came from 2024 feature attribution. 2025 has never been used to
fit the frozen production WP model (trained 2014-2022), but 2025 has been
examined for different kickoff-model experiments; it is NOT globally pristine.
Do not tune based on this comparison and call it fresh independent validation.

Only 2025 regular-season ordinary Q4 kickoffs from the standard own-35 spot
are used. Each is matched with its actual next receiving-team snap, with
the game's eventual result used strictly as its label (not an input).
No 2026 data is read; no production changes, model training or file writes.

Run:
    ./.venv/bin/python src/64_validate_2025_kickoff_encoding.py
"""

from __future__ import annotations

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

OBSERVED_2024 = ROOT / "src/61_audit_observed_kickoff_wp.py"
KICKOFF_INPUT_FIELDS = ("down", "ydstogo", "yardline_100", "goal_to_go")
SCORABLE = {"run", "pass", "no_play", "qb_kneel", "qb_spike"}


def ordinary_2025_q4_pairs(pbp: pd.DataFrame) -> pd.DataFrame:
    """Causally pair normal 2025 REG kickoffs with next possession.

    Reject 2024/2026 rows; do not cross into other kickoffs or change-score
    possession boundaries. Exclude onside plays, penalty-marked kicks and
    unusual kicking spots. Use the first eligible scrimmage row after each
    kickoff; matches are not necessarily consecutive play IDs.
    """
    helpers = runpy.run_path(str(OBSERVED_2024))
    fields = helpers["FIELDS"]
    required = set(fields) | {
        "season", "season_type", "game_id", "play_id", "play_type",
        "touchback", "desc",
    }
    if missing := required.difference(pbp.columns):
        raise ValueError(f"Missing required PBP columns: {sorted(missing)}")
    if pbp.empty or not pbp["season"].eq(2025).all():
        raise ValueError("Only 2025 PBP is eligible (no 2024 or 2026).")

    df = pbp.sort_values(["game_id", "play_id"]).reset_index(drop=True).copy()
    df["kick_seq"] = df["play_type"].eq("kickoff").groupby(df["game_id"]).cumsum()
    shared = ["game_id", "kick_seq"]
    kick = df.loc[df["play_type"].eq("kickoff"), [
        *shared, "play_id", "desc", "touchback", "season_type", *fields,
    ]].copy()
    snap = df.loc[
        df["play_type"].isin(SCORABLE)
        & df["posteam"].notna()
        & df["down"].notna()
        & df["yardline_100"].notna(),
        [*shared, "play_id", "season_type", *fields],
    ].drop_duplicates(shared).copy()

    pairs = kick.merge(
        snap, on=shared, how="inner", validate="one_to_one",
        suffixes=("_kick", "_snap"),
    )
    own = 100 - pd.to_numeric(pairs["yardline_100_snap"], errors="coerce")
    elapsed = (
        pd.to_numeric(pairs["game_seconds_remaining_kick"], errors="coerce")
        - pd.to_numeric(pairs["game_seconds_remaining_snap"], errors="coerce")
    )
    touchback = pd.to_numeric(
        pairs["touchback"], errors="coerce"
    ).fillna(0).eq(1)
    ordinary = (
        (touchback & own.isin([20, 35]))
        | (~touchback & elapsed.between(1, 35))
    )
    special = pairs["desc"].fillna("").str.contains(
        "onside|penalty", case=False, regex=True
    )
    eligible = (
        pairs["season_type_kick"].eq("REG")
        & pairs["season_type_snap"].eq("REG")
        & pairs["qtr_kick"].eq(4)
        & pairs["qtr_snap"].eq(4)
        & pairs["play_id_snap"].gt(pairs["play_id_kick"])
        & pd.to_numeric(pairs["yardline_100_kick"], errors="coerce").eq(35)
        & pairs["posteam_kick"].notna()
        & pairs["posteam_kick"].eq(pairs["posteam_snap"])
        & elapsed.between(0, 35)
        & own.between(1, 99)
        & ~special
        & ordinary
    )
    out = pairs.loc[eligible].copy().reset_index(drop=True)
    if out.empty:
        raise ValueError("No 2025 REG ordinary Q4 standard-35 kickoff pairs found.")

    encode = helpers["feature_frame"]
    kick_features = encode(out, "kick")
    snap_features = encode(out, "snap")
    unchanged_score = np.isclose(
        kick_features["home_score_differential"].to_numpy(dtype=float),
        snap_features["home_score_differential"].to_numpy(dtype=float),
        rtol=0, atol=0.01,
    )
    out = out.loc[unchanged_score].copy().reset_index(drop=True)
    if out.empty:
        raise ValueError("No score-consistent 2025 Q4 kickoff/first-snap pairs.")
    return out


def candidate_features(native: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Baseline synthetic inputs vs frozen candidate from 2024 attribution."""
    if missing := set(KICKOFF_INPUT_FIELDS).difference(native.columns):
        raise ValueError(f"Missing kickoff feature(s): {sorted(missing)}")
    if native.empty:
        raise ValueError("Need 2025 standard-35 kickoff samples.")
    original_engine = native.copy()
    for key in ("down", "ydstogo", "yardline_100"):
        original_engine[key] = np.nan
    original_engine["goal_to_go"] = 0.0

    candidate = original_engine.copy()
    candidate["ydstogo"] = 0.0
    candidate["yardline_100"] = 35.0
    return original_engine, candidate


def score_actual_pairs(pairs: pd.DataFrame, model, features: list[str]) -> pd.DataFrame:
    """Evaluate identical outcome labels with native, engine, and candidate WP."""
    helper = runpy.run_path(str(OBSERVED_2024))
    feature_frame = helper["feature_frame"]
    predict = helper["model_probabilities"]

    native = feature_frame(pairs, "kick")
    snap = feature_frame(pairs, "snap")
    synthetic, aligned = candidate_features(native)
    p_native = predict(model, features, native)
    p_engine = predict(model, features, synthetic)
    p_candidate = predict(model, features, aligned)
    p_snap = predict(model, features, snap)

    actual = pd.to_numeric(pairs["result_kick"], errors="coerce")
    if not np.isfinite(actual.to_numpy(dtype=float)).all():
        raise ValueError("Missing or nonfinite final home win labels.")
    home_receives = native["is_home_posteam"].to_numpy(dtype=bool)
    margin_home = native["home_score_differential"].to_numpy(dtype=float)
    return pd.DataFrame({
        "game_id": pairs["game_id"].to_numpy(),
        "seconds": native["game_seconds_remaining"].to_numpy(dtype=float),
        "home_receives": home_receives,
        "receiver_margin": np.where(home_receives, margin_home, -margin_home),
        "observed_home_win": actual.gt(0).to_numpy(dtype=float),
        "p_native": p_native,
        "p_engine": p_engine,
        "p_aligned": p_candidate,
        "p_first_snap": p_snap,
        "native_down": native["down"].to_numpy(dtype=float),
        "native_ydstogo": native["ydstogo"].to_numpy(dtype=float),
        "native_yardline": native["yardline_100"].to_numpy(dtype=float),
        "native_goal_to_go": native["goal_to_go"].to_numpy(dtype=float),
    })


def game_cluster_ci(
    differences: np.ndarray, game_ids: np.ndarray,
    *, repeats: int = 1200, seed: int = 64,
) -> tuple[float, float]:
    """Paired Brier improvement bootstrap, resampling whole games."""
    delta = np.asarray(differences, dtype=float)
    ids = np.asarray(game_ids)
    if delta.ndim != 1 or ids.shape != delta.shape or not len(delta):
        raise ValueError("Need matching nonempty Brier differences and game IDs.")
    if not np.isfinite(delta).all():
        raise ValueError("Brier differences must be finite.")
    if repeats < 100:
        raise ValueError("At least 100 bootstrap resamples required.")
    games = pd.unique(ids)
    idx_by_game = [np.flatnonzero(ids == game) for game in games]
    rng = np.random.default_rng(seed)
    estimates = np.empty(repeats, dtype=float)
    for j in range(repeats):
        selected = rng.integers(0, len(games), len(games))
        sample_idx = np.concatenate([idx_by_game[i] for i in selected])
        estimates[j] = delta[sample_idx].mean()
    lo, hi = np.quantile(estimates, [0.025, 0.975])
    return float(lo), float(hi)


def grouped_report(name: str, scores: pd.DataFrame, *, repeats: int = 1200) -> None:
    if scores.empty:
        print(f"{name}: no qualifying games")
        return

    truth = scores["observed_home_win"].to_numpy(dtype=float)
    current = scores["p_engine"].to_numpy(dtype=float)
    aligned = scores["p_aligned"].to_numpy(dtype=float)
    native = scores["p_native"].to_numpy(dtype=float)
    snapped = scores["p_first_snap"].to_numpy(dtype=float)
    brier = lambda p: float(np.mean((truth-p)**2))
    diff = (truth-current)**2 - (truth-aligned)**2
    lo, hi = game_cluster_ci(
        diff, scores["game_id"].to_numpy(), repeats=repeats
    )
    n = len(scores)
    print(f"\n{name}: {n} ordinary Q4 kickoffs, {scores['game_id'].nunique()} games"
          + (" [SMALL SAMPLE]" if n < 30 else ""))
    print(f"  Actual home wins: {100*truth.mean():.2f}%")
    print(f"  Avg predicted home WP: engine={100*current.mean():.2f}% "
          f"train-aligned={100*aligned.mean():.2f}% "
          f"2025-native={100*native.mean():.2f}% "
          f"first-snap={100*snapped.mean():.2f}%")
    print(f"  Brier: engine={brier(current):.5f}, "
          f"train-aligned={brier(aligned):.5f}, "
          f"native={brier(native):.5f}, first-snap={brier(snapped):.5f}")
    print(f"  Brier improvement (positive better)={diff.mean():+.5f}, "
          f"game-bootstrap 95% CI=[{lo:+.5f}, {hi:+.5f}]")


def main() -> None:
    os.chdir(ROOT)
    print("\n=== STEP 64: 2025 REAL KICKOFF WP FEATURE-ENCODING CHECK ===")
    print("No model training, no parameter selection from 2025, no changes to app.")
    print("Frozen candidate selected from 2024 diagnostics: NaN/0/35/0.")
    print("Loading 2025 regular-season play-by-play...")
    pbp = nfl.load_pbp([2025]).sort(["game_id", "play_id"]).to_pandas()
    pairs = ordinary_2025_q4_pairs(pbp)
    print(f"Paired 2025 REG Q4 ordinary kickoff -> first snap: {len(pairs)} "
          f"from {pairs['game_id'].nunique()} games.")
    print("Only kickoffs from the standard 35; exclude onside, penalties, "
          "unmatched/special outcomes.")

    engine = runpy.run_path(str(ROOT / "src/42_build_decision_engine.py"))
    features = list(engine["WP_FEATURES"])
    scores = score_actual_pairs(pairs, engine["wp_model"], features)

    print("\n=== CHECK INPUT PARITY AGAINST 2025 REAL KICKOFF ROWS ===")
    for field in ("native_down", "native_ydstogo",
                  "native_yardline", "native_goal_to_go"):
        counts = scores[field].value_counts(dropna=False).head(5)
        print(f"{field}: {counts.to_dict()}")
    mismatch = np.max(np.abs(scores["p_aligned"] - scores["p_native"]))
    print(f"Max(abs(train-aligned WP - real kickoff WP)) = {mismatch:.9f}")
    if mismatch > 1e-9:
        print("WARNING: Candidate does NOT reproduce observed 2025 metadata; "
              "investigate before considering promotion.")

    grouped_report("2025 REG Q4 ALL", scores)
    grouped_report("KICKING TEAM HOME", scores.loc[~scores["home_receives"]])
    grouped_report("KICKING TEAM AWAY", scores.loc[scores["home_receives"]])
    late = scores.loc[scores["seconds"].le(300)]
    grouped_report("LAST FIVE MINUTES, KICKER HOME",
                   late.loc[~late["home_receives"]])
    grouped_report("LAST FIVE MINUTES, KICKER AWAY",
                   late.loc[late["home_receives"]])
    for low, high, label in (
        (1, 3, "RECEIVER LEADING 1-3"),
        (-3, -1, "RECEIVER TRAILING 1-3"),
    ):
        sample = late.loc[late["receiver_margin"].between(low, high)]
        grouped_report(f"LAST FIVE MINUTES, {label}, KICKER HOME",
                       sample.loc[~sample["home_receives"]])
        grouped_report(f"LAST FIVE MINUTES, {label}, KICKER AWAY",
                       sample.loc[sample["home_receives"]])

    print("\nINTERPRETATION: Brier scores measure prediction performance on "
          "observed kickoffs; first-snap estimates incorporate later information.")
    print("A positive Brier improvement does NOT establish the right FG/GO "
          "decision or validate the hypothetical 2025+ kickoff continuation.")
    print("2025 was previously inspected for kickoff probabilities; do not "
          "claim it is untouched for all model selection.")
    print("READ ONLY. No 2026 games loaded and no files or models written.")


if __name__ == "__main__":
    main()
