"""Step 65: chronological, read-only validation of pre-kickoff WP continuation.

We already confirmed that original synthetic kickoff features misrepresent the
training distribution, while a native-like encoding does not improve 2025
predictive accuracy uniformly. This test asks a more useful question:
if we replace an unresolved ordinary kickoff with the EXPECTED first-down WP,
using a donor distribution learned only from PRIOR 2025 weeks, does predictive
accuracy improve on LATER 2025 real games?

No held-out kickoff's actual outcome is used for its continuation prediction.
The actual first snap is used only as a hindsight reference for calibration,
not in any prediction. 2025 was examined by earlier diagnostics, so this is a
chronological DEVELOPMENT comparison, not pristine independent validation.

Conditional ordinary-kickoff study only. No onside returns/recoveries, unusual
penalties, return TDs, or end-of-regulation transitions. No 2026 data, no
training of the frozen WP model, no production/app changes and no saved files.

Run at repository root:
  ./.venv/bin/python src/65_crossfit_2025_continuation_wp.py
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

STEP_64 = ROOT / "src/64_validate_2025_kickoff_encoding.py"
STEP_61 = ROOT / "src/61_audit_observed_kickoff_wp.py"

# Predeclared expansion windows; strictly earlier donor weeks than validation.
FOLDS = ((8, 9, 12), (12, 13, 15), (15, 16, 18))
MIN_CLOCK = 40.0  # avoid unmodeled end-of-Q4/terminal boundaries


def with_kickoff_weeks(pairs: pd.DataFrame, pbp: pd.DataFrame) -> pd.DataFrame:
    """Attach 2025 REG week by the actual kickoff ID, one-to-one."""
    fields = {"season", "game_id", "play_id", "play_type", "week", "season_type"}
    if missing := fields.difference(pbp.columns):
        raise ValueError(f"Missing PBP columns: {sorted(missing)}")
    if (pbp.empty or not pbp["season"].eq(2025).all()
            or pairs.empty or
            not {"game_id", "play_id_kick", "game_seconds_remaining_kick"}.issubset(pairs.columns)):
        raise ValueError("Only nonempty, 2025 paired kickoff data is eligible.")
    metadata = pbp.loc[pbp["play_type"].eq("kickoff"), [
        "game_id", "play_id", "week", "season_type",
    ]].rename(columns={"play_id": "play_id_kick"})
    if metadata.duplicated(["game_id", "play_id_kick"]).any():
        raise ValueError("Duplicate kickoff IDs in metadata.")
    joined = pairs.merge(
        metadata, on=["game_id", "play_id_kick"], how="left",
        validate="one_to_one",
    )
    week = pd.to_numeric(joined["week"], errors="coerce")
    clock = pd.to_numeric(joined["game_seconds_remaining_kick"], errors="coerce")
    result = joined.loc[
        joined["season_type"].eq("REG")
        & week.between(1, 18) & (clock > MIN_CLOCK)
    ].copy()
    if result.empty:
        raise ValueError("No eligible 2025 REG Q4 kickoffs with >40 seconds.")
    result["week"] = pd.to_numeric(result["week"]).astype(int)
    if result.groupby("game_id")["week"].nunique().gt(1).any():
        raise ValueError("A game cannot occupy multiple weeks.")
    return result.reset_index(drop=True)


def earlier_donors(train: pd.DataFrame, valid: pd.DataFrame) -> pd.DataFrame:
    """Use only actual historical transitions in strictly EARLIER weeks."""
    needed = {
        "week", "game_id", "touchback", "yardline_100_snap",
        "game_seconds_remaining_kick", "game_seconds_remaining_snap",
    }
    if missing := needed.difference(train.columns):
        raise ValueError(f"Missing training donor columns: {sorted(missing)}")
    if missing := {"week", "game_id"}.difference(valid.columns):
        raise ValueError(f"Missing validation columns: {sorted(missing)}")
    if train.empty or valid.empty or train["week"].max() >= valid["week"].min():
        raise ValueError("Chronological leakage: donor week is not earlier.")
    if set(train["game_id"]) & set(valid["game_id"]):
        raise ValueError("Game leakage between donors and validation.")

    own = 100 - pd.to_numeric(train["yardline_100_snap"], errors="coerce")
    elapsed = (
        pd.to_numeric(train["game_seconds_remaining_kick"], errors="coerce")
        - pd.to_numeric(train["game_seconds_remaining_snap"], errors="coerce")
    )
    touchback = pd.to_numeric(train["touchback"], errors="coerce").fillna(0).eq(1)
    if (not own.between(1, 99).all()
            or not elapsed.between(0, 35).all()
            or not (own[ touchback].isin([20, 35])).all()
            or not (elapsed[~touchback].between(1, 35)).all()):
        raise ValueError("Invalid ordinary 2025 donor returns or touchbacks.")
    if touchback.sum() < 15 or (~touchback).sum() < 15:
        raise ValueError("Too few prior ordinary touchbacks or returns.")
    return pd.DataFrame({
        "own": own.to_numpy(float),
        "elapsed": elapsed.to_numpy(float),
        "touchback": touchback.to_numpy(bool),
    })


def expected_first_snap_home_wp(
    wp_model, features: list[str], kickoff_features: pd.DataFrame,
    donors: pd.DataFrame, *, max_kicks_per_batch: int = 18,
) -> np.ndarray:
    """EXACT empirical donor mean, never the current kickoff's actual return.

    For each kickoff, combine *all previous-week* standard-35 Q4 ordinary
    kickoff outcomes with the frozen WP model. Scores, timeouts and home/away
    possession are held at pre-kickoff values; modeled return clock is removed.
    This intentionally conditions on ordinary kickoffs, not onside/special.
    """
    required_features = {
        "game_seconds_remaining", "half_seconds_remaining", "qtr", "down",
        "ydstogo", "yardline_100", "goal_to_go", "is_kickoff",
        "final_five_minutes",
    }
    if missing := required_features.difference(features):
        raise ValueError(f"Missing required WP features: {sorted(missing)}")
    if missing := set(features).difference(kickoff_features.columns):
        raise ValueError(f"Missing kickoff features: {sorted(missing)}")
    if donors.empty or not {"own", "elapsed", "touchback"}.issubset(donors.columns):
        raise ValueError("Missing prior kickoff donors.")
    if max_kicks_per_batch < 1:
        raise ValueError("max_kicks_per_batch must be positive.")
    own = donors["own"].to_numpy(float)
    elapsed = donors["elapsed"].to_numpy(float)
    if (not np.isfinite(own).all() or not np.isfinite(elapsed).all()
            or not ((own > 0) & (own < 100)).all()
            or not ((elapsed >= 0) & (elapsed <= 35)).all()):
        raise ValueError("Invalid first-possession donor states.")

    classes = np.asarray(wp_model.classes_).astype(int)
    class_index = np.flatnonzero(classes == 2)
    if len(class_index) != 1:
        raise ValueError("WP model lacks exactly one home-win class 2.")
    pos = {name: features.index(name) for name in required_features}
    n_donors = len(donors)
    outcomes = np.empty(len(kickoff_features), dtype=float)

    for first in range(0, len(kickoff_features), max_kicks_per_batch):
        end = min(first+max_kicks_per_batch, len(kickoff_features))
        kick = kickoff_features.iloc[first:end]
        rows = np.repeat(kick[features].to_numpy(dtype=float), n_donors, axis=0)
        repeated_clock = np.repeat(kick["game_seconds_remaining"].to_numpy(float),
                                   n_donors)
        repeated_half = np.repeat(kick["half_seconds_remaining"].to_numpy(float),
                                  n_donors)
        tiled_elapsed = np.tile(elapsed, len(kick))
        yardline100 = 100.0 - np.tile(own, len(kick))
        next_clock = np.maximum(0.0, repeated_clock - tiled_elapsed)
        next_half = np.maximum(0.0, repeated_half - tiled_elapsed)

        rows[:, pos["game_seconds_remaining"]] = next_clock
        rows[:, pos["half_seconds_remaining"]] = next_half
        rows[:, pos["down"]] = 1.0
        rows[:, pos["ydstogo"]] = np.minimum(10.0, yardline100)
        rows[:, pos["yardline_100"]] = yardline100
        rows[:, pos["goal_to_go"]] = (yardline100 <= 10).astype(float)
        rows[:, pos["is_kickoff"]] = 0.0
        rows[:, pos["final_five_minutes"]] = (next_clock <= 300).astype(float)

        p = wp_model.predict_proba(rows)[:, class_index[0]]
        if not np.isfinite(p).all() or not ((p >= 0) & (p <= 1)).all():
            raise ValueError("WP model returned invalid post-kickoff probabilities.")
        outcomes[first:end] = p.reshape(len(kick), n_donors).mean(axis=1)
    return outcomes


def crossfit_predictions(
    pairs: pd.DataFrame, wp_model, features: list[str], feature_frame,
    model_probabilities,
) -> pd.DataFrame:
    """Score pre-kickoff forecasts only on chronologically later weeks."""
    if pairs.empty or not pairs["week"].between(1, 18).all():
        raise ValueError("Use only 2025 REG weeks 1-18.")
    if pairs.groupby("game_id")["week"].nunique().gt(1).any():
        raise ValueError("A game spans multiple weeks.")
    frames = []
    for last_train, first_valid, last_valid in FOLDS:
        train = pairs.loc[pairs["week"].le(last_train)].copy()
        valid = pairs.loc[pairs["week"].between(first_valid, last_valid)].copy()
        if valid.empty:
            raise ValueError(f"No validation kicks in weeks {first_valid}-{last_valid}.")
        donors = earlier_donors(train, valid)
        native = feature_frame(valid, "kick")
        first_snap = feature_frame(valid, "snap")
        original = native.copy()
        for field in ("down", "ydstogo", "yardline_100"):
            original[field] = np.nan
        original["goal_to_go"] = 0.0

        actual = pd.to_numeric(valid["result_kick"], errors="coerce").to_numpy(float)
        if not np.isfinite(actual).all():
            raise ValueError("Missing final-game outcome labels.")
        home_receives = native["is_home_posteam"].to_numpy(bool)
        receiver_margin = native["home_score_differential"].to_numpy(float)
        df = pd.DataFrame({
            "game_id": valid["game_id"].to_numpy(),
            "week": valid["week"].to_numpy(int),
            "seconds": native["game_seconds_remaining"].to_numpy(float),
            "home_receives": home_receives,
            "receiver_margin": np.where(home_receives, receiver_margin,
                                        -receiver_margin),
            "actual_home_win": (actual > 0).astype(float),
            "engine_pending": model_probabilities(wp_model, features, original),
            "native_pending": model_probabilities(wp_model, features, native),
            "continuation": expected_first_snap_home_wp(
                wp_model, features, native, donors,
            ),
            "oracle_first_snap": model_probabilities(wp_model, features, first_snap),
            "train_end": last_train,
            "donors": len(donors),
            "train_touchback_rate": float(donors["touchback"].mean()),
        })
        if not df["week"].gt(last_train).all():
            raise AssertionError("Training and validation overlap.")
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def game_cluster_ci(delta: np.ndarray, games: np.ndarray,
                    *, repeats: int = 1200, seed: int = 65):
    """Paired bootstrap over games, not individual kickoff rows."""
    values = np.asarray(delta, dtype=float)
    ids = np.asarray(games)
    if values.ndim != 1 or values.size == 0 or ids.shape != values.shape:
        raise ValueError("Brier differences and game IDs must align.")
    if not np.isfinite(values).all() or repeats < 100:
        raise ValueError("Finite Brier differences and >=100 replicates required.")
    unique = pd.unique(ids)
    indices = [np.flatnonzero(ids == game) for game in unique]
    rng = np.random.default_rng(seed)
    boot = np.empty(repeats)
    for k in range(repeats):
        drawn = rng.integers(0, len(unique), len(unique))
        sample = np.concatenate([indices[j] for j in drawn])
        boot[k] = values[sample].mean()
    return tuple(map(float, np.quantile(boot, [0.025, 0.975])))


def summarize(label: str, scores: pd.DataFrame) -> None:
    if scores.empty:
        print(f"{label}: no qualifying situations")
        return
    y = scores["actual_home_win"].to_numpy(float)
    n = len(scores)
    groups = scores["game_id"].nunique()
    brier = lambda p: float(np.mean((y-p)**2))
    base = scores["engine_pending"].to_numpy(float)
    native = scores["native_pending"].to_numpy(float)
    continuation = scores["continuation"].to_numpy(float)
    after = scores["oracle_first_snap"].to_numpy(float)
    gain_prod = (y-base)**2 - (y-continuation)**2
    gain_native = (y-native)**2 - (y-continuation)**2
    ci_prod = game_cluster_ci(gain_prod, scores["game_id"].to_numpy())
    ci_native = game_cluster_ci(gain_native, scores["game_id"].to_numpy())
    print(f"\n{label}: n={n}, games={groups}"
          + (" [SMALL SAMPLE]" if n < 30 else ""))
    print(f"  Actual home wins: {y.mean():.2%}")
    print(f"  Predicted home WP: engine={base.mean():.2%}, "
          f"native={native.mean():.2%}, continuation={continuation.mean():.2%}, "
          f"oracle first snap={after.mean():.2%}")
    print(f"  Brier: engine={brier(base):.5f}, native={brier(native):.5f}, "
          f"continuation={brier(continuation):.5f}, "
          f"oracle snap={brier(after):.5f}")
    print(f"  Continuation improvement vs engine: {gain_prod.mean():+.5f}, "
          f"95% game-bootstrap [{ci_prod[0]:+.5f}, {ci_prod[1]:+.5f}]")
    print(f"  Continuation improvement vs native: {gain_native.mean():+.5f}, "
          f"95% game-bootstrap [{ci_native[0]:+.5f}, {ci_native[1]:+.5f}]")


def main():
    os.chdir(ROOT)
    print("\n=== STEP 65: CHRONOLOGICAL EXPECTED KICKOFF CONTINUATION WP ===")
    print("2025 REG Q4 standard-35 ordinary kicks only, clock >40s.")
    print("Returns and touchbacks are from PRIOR 2025 weeks only; exact average.")
    print("No realized return used to forecast that same kickoff.")
    print("No 2026 data; no app/model changes or output files.")
    pbp = nfl.load_pbp([2025]).sort(["game_id", "play_id"]).to_pandas()
    helper = runpy.run_path(str(STEP_64))
    observed = runpy.run_path(str(STEP_61))
    pairs = with_kickoff_weeks(helper["ordinary_2025_q4_pairs"](pbp), pbp)
    engine = runpy.run_path(str(ROOT / "src/42_build_decision_engine.py"))
    result = crossfit_predictions(
        pairs, engine["wp_model"], list(engine["WP_FEATURES"]),
        observed["feature_frame"], observed["model_probabilities"],
    )
    print(f"\nEligible paired kickoffs total: {len(pairs)}; "
          f"chronologically evaluated: {len(result)} "
          f"from {result['game_id'].nunique()} games.")
    for last_train, first_valid, last_valid in FOLDS:
        train = pairs.loc[pairs["week"].le(last_train)]
        val = result.loc[result["week"].between(first_valid, last_valid)]
        print(f"Prior W1-{last_train}: {len(train)} donors "
              f"TB={train['touchback'].eq(1).mean():.1%}; "
              f"validate W{first_valid}-{last_valid}: {len(val)} kicks.")

    summarize("ALL 2025 CHRONOLOGICAL VALIDATION", result)
    summarize("KICKER HOME", result.loc[~result["home_receives"]])
    summarize("KICKER AWAY", result.loc[result["home_receives"]])
    late = result.loc[result["seconds"].le(300)]
    summarize("LAST FIVE, KICKER HOME", late.loc[~late["home_receives"]])
    summarize("LAST FIVE, KICKER AWAY", late.loc[late["home_receives"]])
    for lo, hi, label in ((1, 3, "RECEIVER LEADING 1-3"),
                           (-3, -1, "RECEIVER TRAILING 1-3")):
        sample = late.loc[late["receiver_margin"].between(lo, hi)]
        summarize(f"LAST FIVE, {label}, KICKER HOME",
                  sample.loc[~sample["home_receives"]])
        summarize(f"LAST FIVE, {label}, KICKER AWAY",
                  sample.loc[sample["home_receives"]])
    print("\nREAD ONLY. Development comparison; 2025 has been examined previously.")
    print("Conditioning on ordinary kickoffs omits rare and onside outcomes; "
          "it does not validate FG/GO counterfactuals.")
    print("Do not change production based solely on these diagnostic results.")


if __name__ == "__main__":
    main()
