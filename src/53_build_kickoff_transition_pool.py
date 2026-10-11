"""Freeze ordinary 2024 dynamic-kickoff development transitions only.

Run from the repository root:
  ./.venv/bin/python src/53_build_kickoff_transition_pool.py

Never load 2025 data here: it is reserved for held-out evaluation.
"""

from pathlib import Path
import nflreadpy as nfl
from kickoff_transition import build_2024_pool


def main():
    print("Loading development-only 2024 kickoff PBP...")
    pbp = nfl.load_pbp([2024]).to_pandas()
    pool = build_2024_pool(pbp)

    q4 = pool[pool["quarter"] == 4]
    tb = q4[q4["transition_type"] == "touchback"]
    returns = q4[q4["transition_type"] == "return"]

    print(f"Total qualifying kickoff transitions: {len(pool):,}")
    print(f"Q4 transitions: {len(q4):,}")
    print(f"Q4 touchbacks: {len(tb):,}")
    print(f"Q4 returns: {len(returns):,}")
    print("Q4 touchback yards:", tb["own_yardline"].value_counts().to_dict())
    if not returns.empty:
        print("Q4 mean return starting yard:", round(returns["own_yardline"].mean(), 2))
        print("Q4 median return clock:", round(returns["elapsed_seconds"].median(), 2))
    if len(q4) < 250 or len(tb) < 20 or len(returns) < 50:
        raise RuntimeError("2024 donor support unexpectedly small; refusing to write.")

    path = Path("data/kickoff_2024_dev_transition_pool.parquet")
    path.parent.mkdir(parents=True, exist_ok=True)
    pool.to_parquet(path, index=False)
    print("Wrote:", path)


if __name__ == "__main__":
    main()
