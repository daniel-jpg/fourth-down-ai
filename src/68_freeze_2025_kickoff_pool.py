"""Freeze actual 2025 REG Q4 ordinary kickoff transitions as a source artifact.

Runs on GitHub Actions, not the user's Mac. The output is a small static CSV
containing public nflverse PBP-derived game/play IDs, post-kickoff receiving
field position and clock elapsed. This enables reproducible Streamlit previews
without downloading NFL play-by-play on each visitor request.

The causal event pairing and exclusions are the SAME Step 64/65 validators.
Do not include 2026 data, onside kicks, scoring returns, or unusual spots.
2025 has been inspected in prior development; this is NOT a clean holdout.

Run:
  python src/68_freeze_2025_kickoff_pool.py
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import runpy
import sys

import nflreadpy as nfl

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.kickoff_2025_empirical import (  # noqa: E402
    FROZEN_PATH, pool_from_matched_2025, validate_frozen_pool,
)


def main() -> None:
    print("=== STEP 68: FREEZE OBSERVED 2025 ORDINARY Q4 KICKOFFS ===", flush=True)
    pbp = nfl.load_pbp([2025]).sort(["game_id", "play_id"]).to_pandas()
    if pbp.empty or not pbp["season"].eq(2025).all():
        raise RuntimeError("Expected 2025-only nflverse play-by-play.")
    helper64 = runpy.run_path(str(ROOT / "src/64_validate_2025_kickoff_encoding.py"))
    helper65 = runpy.run_path(str(ROOT / "src/65_crossfit_2025_continuation_wp.py"))
    paired = helper64["ordinary_2025_q4_pairs"](pbp)
    paired = helper65["with_kickoff_weeks"](paired, pbp)
    pool = validate_frozen_pool(pool_from_matched_2025(paired))
    tb = pool["transition_type"].eq("touchback")
    returns = pool.loc[~tb]

    # Static, deterministic string representation; no runtime network calls.
    FROZEN_PATH.parent.mkdir(exist_ok=True, parents=True)
    pool.to_csv(FROZEN_PATH, index=False, float_format="%.3f")
    raw = FROZEN_PATH.read_bytes()
    print(
        f"Ordinary 2025 REG Q4 standard-35 first-possession pairs: {len(pool)} "
        f"in {pool['game_id'].nunique()} games.",
        flush=True,
    )
    print(
        f"Observed touchback rate {tb.mean():.2%}; "
        f"TB to own 20/35: {int(pool.loc[tb,'own_yardline'].eq(20).sum())}/"
        f"{int(pool.loc[tb,'own_yardline'].eq(35).sum())}.",
        flush=True,
    )
    print(
        f"Returns {len(returns)}; own-yardline mean "
        f"{returns['own_yardline'].mean():.2f}; "
        f"runoff mean {returns['elapsed_seconds'].mean():.2f}s.",
        flush=True,
    )
    print(f"Frozen CSV: {FROZEN_PATH.relative_to(ROOT)}")
    print(f"SHA256: {hashlib.sha256(raw).hexdigest()}", flush=True)
    print(
        "Frozen 2025 sample describes ordinary kickoffs only; it is not a "
        "validated forecast of 2026+ fourth-down decisions.",
        flush=True,
    )


if __name__ == "__main__":
    main()
