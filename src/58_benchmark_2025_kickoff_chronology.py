"""Chronological 2025-only Q4 kickoff touchback benchmark; no data/model writes.

Run at repo root:
    ./.venv/bin/python src/58_benchmark_2025_kickoff_chronology.py

Models compete against a fold-specific constant. The logistic competitors use
smooth (no category cliffs) score and/or clock features. 2026 data is not loaded
by this script and is not a model-selection source. No Streamlit changes.
"""
from pathlib import Path
import runpy
import sys

import nflreadpy as nfl
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.kickoff_chrono_benchmark import (  # noqa: E402
    FOLDS, CANDIDATES, regular_season_kickoffs,
    benchmark, summarize_predictions,
)


def main() -> None:
    print("\n=== CHRONOLOGICAL 2025 BENCHMARK ===")
    print("2026 is not loaded; no production model is fitted or changed.")
    print("Loading 2025 PBP...")
    pbp = nfl.load_pbp([2025]).sort(["game_id", "play_id"]).to_pandas()
    print(f"PBP rows: {len(pbp):,}, games: {pbp['game_id'].nunique()}")
    pairer = runpy.run_path(str(ROOT / "src/55_audit_2025_kickoffs.py"))["audit_kickoff_pairs"]
    enricher = runpy.run_path(str(ROOT / "src/56_audit_2025_kickoff_context.py"))["add_receiver_margin"]
    enriched = enricher(pairer(pbp, 2025), pbp)
    sample = regular_season_kickoffs(enriched, pbp)
    print(f"2025 REG ordinary Q4 kickoffs: {len(sample)}; "
          f"games: {sample['game_id'].nunique()}; "
          f"observed touchbacks: {sample['status'].eq('touchback').mean():.1%}")
    print("Regulation Q4 ordinary kickoffs only; onside/unmatched/postseason excluded.")

    predictions = benchmark(sample)
    print("\n=== EXPANDING-WINDOW FOLDS (TRAIN STRICTLY EARLIER) ===")
    for train_end, val_first, val_last in FOLDS:
        train = sample.loc[sample["week"].le(train_end)]
        held = predictions.loc[predictions["val_block"].eq(f"{val_first}-{val_last}")]
        print(f"Train W1-{train_end}: {len(train)} kickoffs, "
              f"TB={train['status'].eq('touchback').mean():.1%}  |  "
              f"Validate W{val_first}-{val_last}: {len(held)} kickoffs, "
              f"{held['game_id'].nunique()} games, "
              f"TB={held['status'].eq('touchback').mean():.1%}")

    results = summarize_predictions(predictions, bootstrap_repeats=1200, seed=2025)
    print("\n=== 2025 CHRONOLOGICAL OUT-OF-SAMPLE RESULTS ===")
    print(f"Pooled validation: {len(predictions)} kickoffs, "
          f"{predictions['game_id'].nunique()} games; "
          f"observed TB={predictions['status'].eq('touchback').mean():.1%}")
    print("Positive Brier improvement is better than the fold-specific constant.")
    for rec in results.to_dict("records"):
        print(f"{rec['candidate']:13s} Brier={rec['brier']:.5f} "
              f"log-loss={rec['log_loss']:.5f} "
              f"Brier improvement={rec['improvement_vs_constant']:+.5f} "
              f"95% game-bootstrap=[{rec['ci_low']:+.5f}, {rec['ci_high']:+.5f}]")

    consistent = results.loc[(results["candidate"] != "constant")
                             & (results["improvement_vs_constant"] > 0)
                             & (results["ci_low"] > 0)]
    if consistent.empty:
        print("\nGATE: NO contextual variant convincingly beats the constant on these folds.")
    else:
        names = ", ".join(consistent["candidate"].tolist())
        print(f"\nGATE: Contextual candidate(s) pass this limited 2025 benchmark: {names}.")
        print("Still not production-validated; 2026 early data already examined separately.")
    print("READ ONLY. No parameter tuning on 2026 and no files written.")
    print("This benchmark predicts ordinary kickoff touchbacks only, not special outcomes.")


if __name__ == "__main__":
    main()
