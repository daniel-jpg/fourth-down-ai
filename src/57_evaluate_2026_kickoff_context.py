"""Train a frozen 2025 Q4 ordinary-kickoff touchback model; audit on 2026 W1-5.

Run from repository root:
  ./.venv/bin/python src/57_evaluate_2026_kickoff_context.py

READ ONLY. Training uses 2025. The 2026 feed is consulted ONLY for reporting
out-of-time metrics and must not be used to select hyperparameters, thresholds,
or production defaults. Onside kicks / rare outcomes are explicitly excluded.
"""

from pathlib import Path
import runpy
import sys

import nflreadpy as nfl
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.kickoff_touchback_model import (  # noqa: E402
    fit_2025_model, ordinary_q4, summarize_holdout,
)


def load_enriched(year, pairer, enricher):
    pbp = nfl.load_pbp([year]).sort(["game_id", "play_id"]).to_pandas()
    print(f"{year}: {len(pbp):,} PBP rows, {pbp['game_id'].nunique()} games")
    pairs = pairer(pbp, year)
    enriched = enricher(pairs, pbp)
    ordinary = ordinary_q4(enriched)
    q4 = enriched.loc[enriched["kick_qtr"].eq(4)]
    print(
        f"      Q4 kickoffs={len(q4)} ordinary={len(ordinary)} "
        f"onside={int(q4['status'].eq('onside').sum())} "
        f"other/unmatched={int(q4['status'].eq('other/unmatched').sum())} "
        f"excluded/missing-margin={len(q4)-len(ordinary)}"
    )
    return enriched


def report_groups(model, pairs):
    for label, view in (
        ("2026 Q4", ordinary_q4(pairs)),
        ("2026 Q4 LAST 5 MIN", ordinary_q4(pairs).loc[
            ordinary_q4(pairs)["kick_seconds"].le(300)
        ]),
    ):
        print(f"\n=== {label}: CONTEXT GROUPS (DESCRIPTIVE) ===")
        if view.empty:
            print("No eligible kickoffs")
            continue
        edges = [
            ("receiver leading 1-3", lambda m: m.between(1, 3)),
            ("receiver trailing 1-3", lambda m: m.between(-3, -1)),
            ("receiver trailing 4+", lambda m: m.le(-4)),
            ("receiver leading 4+", lambda m: m.ge(4)),
            ("tied", lambda m: m.eq(0)),
        ]
        for group, pred in edges:
            block = view.loc[pred(view["receiver_margin"])]
            if block.empty:
                continue
            probs = np.array([
                model.probability(r["receiver_margin"], r["kick_seconds"])
                for _, r in block.iterrows()
            ])
            rate = block["status"].eq("touchback").mean()
            marker = "SMALL SAMPLE" if len(block) < 30 else ""
            print(f"{group:23s} n={len(block):3d} observed={rate:.1%} "
                  f"predicted={probs.mean():.1%} {marker}")


def main():
    pairer = runpy.run_path(str(ROOT / "src/55_audit_2025_kickoffs.py"))["audit_kickoff_pairs"]
    enricher = runpy.run_path(str(ROOT / "src/56_audit_2025_kickoff_context.py"))["add_receiver_margin"]

    print("\n=== TRAIN: 2025 ONLY ===")
    development = load_enriched(2025, pairer, enricher)
    model = fit_2025_model(development)
    print(f"2025 Q4 training ordinary kickoffs: {model.training_count}")
    print(f"2025 Q4 constant touchback baseline: {model.training_touchback_rate:.2%}")
    for receiver_margin, label in [(1, "FG: receiver leading by 1"),
                                   (-3, "TD+XP: receiver trailing by 3"),
                                   (-4, "TD+2: receiver trailing by 4")]:
        print(f"Predicted TB at 4:59, {label}: "
              f"{model.probability(receiver_margin, 299):.2%}")

    print("\n=== TEST: 2026 WEEKS AVAILABLE (OUT OF TIME) ===")
    holdout = load_enriched(2026, pairer, enricher)
    summary = summarize_holdout(model, holdout)
    print(f"Evaluable Q4 sample: {summary['n']} kickoffs in {summary['games']} games")
    print(f"Observed touchback rate:   {summary['observed_touchback_rate']:.2%}")
    print(f"2025 constant forecast:    {summary['constant_prediction']:.2%}")
    print(f"2025 context forecast avg: {summary['contextual_prediction_mean']:.2%}")
    print(f"Constant baseline Brier:   {summary['constant_brier']:.5f}")
    print(f"Context model Brier:       {summary['contextual_brier']:.5f}")
    print(f"Brier improvement (>0 better): {summary['brier_improvement']:+.5f}")
    ci_lo, ci_hi = summary['improvement_ci']
    print(f"Game-cluster bootstrap 95% interval for improvement: [{ci_lo:+.5f}, {ci_hi:+.5f}]")
    print(f"Constant log loss: {summary['constant_log_loss']:.5f}")
    print(f"Context log loss:  {summary['contextual_log_loss']:.5f}")
    report_groups(model, holdout)
    print("\nREAD ONLY. The 2026 evaluation does not fit or change the model.")
    print("This model excludes onside recoveries, return touchdowns and penalties.")
    print("Do NOT enable in Streamlit solely because of this small early-season holdout.")


if __name__ == "__main__":
    main()
