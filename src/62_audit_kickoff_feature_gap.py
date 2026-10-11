"""Attribute native-vs-engine kickoff WP differences to individual features.

Read only. Uses only 2024 observed ordinary Q4 kickoff rows. This check
does NOT recommend copying historical PBP values into future kickoff states;
first identify which model inputs create the distribution shift.

Run from the project root:
  ./.venv/bin/python src/62_audit_kickoff_feature_gap.py
"""
from __future__ import annotations

from pathlib import Path
import os
import runpy

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
KICKOFF_FIELDS = ("down", "ydstogo", "yardline_100", "goal_to_go")


def kickoff_variants(native: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Isolate the exact four columns changed by the engine kickoff encoding."""
    missing = set(KICKOFF_FIELDS).difference(native.columns)
    if missing:
        raise ValueError(f"Missing native feature(s): {sorted(missing)}")
    if native.empty:
        raise ValueError("Need at least one historical kickoff row.")

    engine = native.copy()
    for col in KICKOFF_FIELDS[:3]:
        engine[col] = np.nan
    engine["goal_to_go"] = 0.0

    variants = {"native": native.copy(), "engine": engine}
    for col in KICKOFF_FIELDS:
        single = native.copy()
        single[col] = engine[col]
        variants[f"only_{col}"] = single

        restored = engine.copy()
        restored[col] = native[col]
        variants[f"restore_{col}"] = restored

    step = native.copy()
    for col in KICKOFF_FIELDS:
        step = step.copy()
        step[col] = engine[col]
        variants[f"step_{col}"] = step
    return variants


def predicted_kickoffs(model, features: list[str],
                       frames: dict[str, pd.DataFrame]) -> dict[str, np.ndarray]:
    """Predict home-win probabilities without fitting, calibrating or modifying the model."""
    class_index = {int(c): i for i, c in enumerate(model.classes_)}
    if 2 not in class_index:
        raise ValueError("WP model must have home-win class 2.")
    outputs = {}
    for name, frame in frames.items():
        if missing := set(features).difference(frame.columns):
            raise ValueError(f"Missing WP columns: {sorted(missing)}")
        p = model.predict_proba(frame[features].to_numpy())[:, class_index[2]]
        if not np.isfinite(p).all():
            raise ValueError(f"Nonfinite probabilities for {name}")
        outputs[name] = np.asarray(p, dtype=float)
    return outputs


def feature_contributions(p: dict[str, np.ndarray],
                          selector: np.ndarray) -> pd.DataFrame:
    """Report one-field changes and order-dependent sequential contributions.

    Individual changes need not add: tree-based models contain interactions.
    Sequential changes add exactly for this specified field order.
    """
    selector = np.asarray(selector, dtype=bool)
    n = len(p["native"])
    if selector.shape != (n,) or not selector.any():
        raise ValueError("Need a nonempty mask matching prediction length.")
    previous = p["native"]
    rows = []
    for col in KICKOFF_FIELDS:
        next_step = p[f"step_{col}"]
        rows.append({
            "feature": col,
            "single_change_pp": 100 * float(np.mean(
                (p[f"only_{col}"] - p["native"])[selector])),
            "restore_change_pp": 100 * float(np.mean(
                (p[f"restore_{col}"] - p["engine"])[selector])),
            "sequential_change_pp": 100 * float(np.mean(
                (next_step - previous)[selector])),
        })
        previous = next_step
    return pd.DataFrame(rows)


def _top_values(series: pd.Series) -> str:
    values = series.value_counts(dropna=False).head(6)
    return "; ".join(f"{str(k)}:{v}" for k, v in values.items())


def main() -> None:
    import nflreadpy as nfl

    os.chdir(ROOT)
    print("\n=== 2024 KICKOFF WP FEATURE ATTRIBUTION (READ ONLY) ===")
    print("Only the four kickoff input fields are changed in this audit.")
    print("Loading 2024 PBP and frozen production win-probability model...")
    helper = runpy.run_path(str(ROOT / "src/61_audit_observed_kickoff_wp.py"))
    pbp = nfl.load_pbp([2024]).sort(["game_id", "play_id"]).to_pandas()
    pairs = helper["pair_2024_q4"](pbp)
    native = helper["feature_frame"](pairs, "kick")
    engine = runpy.run_path(str(ROOT / "src/42_build_decision_engine.py"))
    frames = kickoff_variants(native)
    p = predicted_kickoffs(engine["wp_model"], list(engine["WP_FEATURES"]), frames)

    print(f"\nEligible ordinary 2024 Q4 kickoffs: {len(pairs)}")
    print("Historical kickoff feature values before engine's missing-value replacement:")
    for col in KICKOFF_FIELDS:
        series = native[col]
        print(f"  {col:13s} nonmissing={series.notna().mean():6.1%}"
              f" values={_top_values(series)}")

    # nflverse kickoff posteam is RECEIVING side, not kicking side.
    receiver_home = native["is_home_posteam"].to_numpy(dtype=bool)
    masks = (
        ("ALL", np.ones(len(native), dtype=bool)),
        ("KICKER HOME", ~receiver_home),
        ("KICKER AWAY", receiver_home),
        ("Q4 LAST FIVE, KICKER HOME",
         ~receiver_home & native["game_seconds_remaining"].le(300).to_numpy()),
        ("Q4 LAST FIVE, KICKER AWAY",
         receiver_home & native["game_seconds_remaining"].le(300).to_numpy()),
    )

    for label, mask in masks:
        if not mask.any():
            continue
        print(f"\n=== {label}: {int(mask.sum())} kickoffs ===")
        mean_native = float(np.mean(p["native"][mask]))
        mean_engine = float(np.mean(p["engine"][mask]))
        print(f"Native home WP: {100*mean_native:.2f}%")
        print(f"Synthetic engine home WP: {100*mean_engine:.2f}%")
        print(f"Change from synthetic encoding: {100*(mean_engine-mean_native):+.2f} pp")
        print("Single-swap and restore effects may interact; sequential changes add exactly.")
        detail = feature_contributions(p, mask)
        for _, r in detail.iterrows():
            print(
                f"  {r['feature']:13s} only={r['single_change_pp']:+7.2f} pp "
                f"restore={r['restore_change_pp']:+7.2f} pp "
                f"sequential={r['sequential_change_pp']:+7.2f} pp"
            )

    print("\n=== LARGEST NATIVE-TO-SYNTHETIC DIFFERENCES ===")
    worst = np.argsort(-np.abs(p["engine"] - p["native"]))[:8]
    for i in worst:
        k = pairs.iloc[int(i)]
        name = "HOME" if not receiver_home[i] else "AWAY"
        values = ", ".join(f"{f}={native.iloc[i][f]!r}" for f in KICKOFF_FIELDS)
        print(f"  {k['game_id']} play={k['play_id_kick']} "
              f"kicker={name} Q4 clock={native.iloc[i]['game_seconds_remaining']:.0f}s "
              f"native={100*p['native'][i]:.2f}% "
              f"synthetic={100*p['engine'][i]:.2f}% "
              f"delta={100*(p['engine'][i]-p['native'][i]):+.2f} pp "
              f"| {values}")

    print("\nNo files or models were changed. 2025/2026 data was not accessed.")
    print("Do not treat field attribution as evidence that historical kickoff fields")
    print("should be copied unchanged into synthetic 2025+ kickoffs.")


if __name__ == "__main__":
    main()
