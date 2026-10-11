"""2025-only ordinary Q4 kickoff donors and isolated 2025+ continuation.

This is an experimental model, not a validated production recommendation.
The frozen, auditable CSV is built from 2025 REG standard-35 kickoffs, paired
to the receiving team's next scrimmage snap with no intervening score.
Only normal touchbacks/returns with >40 seconds at the kickoff are included.

The entire observed 2025 distribution is sampled together: touchback vs return,
touchback placement (own 20/35), receiving spot and elapsed seconds. Unlike the
2024 sensitivity sampler there is no guessed 2025+ touchback rate.
It does not model onside plays, return TDs, exceptional penalties, or changing
2026 coaching behavior. This is a latest-completed-season descriptive scenario.

Never patch the globally cached Streamlit engine. Use ONLY a private module.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd


FROZEN_PATH = Path(__file__).resolve().parents[1] / (
    "data/kickoff_2025_reg_q4_ordinary.csv"
)
COLUMNS = [
    "season", "game_id", "week", "kick_play_id", "quarter",
    "transition_type", "own_yardline", "elapsed_seconds",
]


def pool_from_matched_2025(pairs: pd.DataFrame) -> pd.DataFrame:
    """Convert strictly verified Step 65 historical kickoff pairs to donors."""
    required = {
        "game_id", "week", "play_id_kick", "qtr_kick", "touchback",
        "yardline_100_kick", "yardline_100_snap",
        "game_seconds_remaining_kick", "game_seconds_remaining_snap",
    }
    if missing := required.difference(pairs.columns):
        raise ValueError(f"Missing historical matched fields: {sorted(missing)}")
    if pairs.empty or not pairs["week"].between(1, 18).all():
        raise ValueError("Need 2025 REG weeks 1-18 matched kickoffs.")
    if not pd.to_numeric(pairs["qtr_kick"], errors="coerce").eq(4).all():
        raise ValueError("Only regulation Q4 kickoff donors supported.")
    if not pd.to_numeric(pairs["yardline_100_kick"], errors="coerce").eq(35).all():
        raise ValueError("Only standard own-35 kickoffs are eligible.")
    if not pd.to_numeric(
        pairs["game_seconds_remaining_kick"], errors="coerce"
    ).gt(40).all():
        raise ValueError("Reject terminal-clock donors.")
    own = 100.0 - pd.to_numeric(
        pairs["yardline_100_snap"], errors="coerce"
    )
    elapsed = (
        pd.to_numeric(pairs["game_seconds_remaining_kick"], errors="coerce")
        - pd.to_numeric(pairs["game_seconds_remaining_snap"], errors="coerce")
    )
    tb = pd.to_numeric(pairs["touchback"], errors="coerce").eq(1)
    if (not own.between(1, 99).all()
            or not elapsed.between(0, 35).all()
            or not own.loc[tb].isin([20, 35]).all()
            or not elapsed.loc[~tb].between(1, 35).all()):
        raise ValueError("Invalid 2025 touchback or ordinary return.")
    pool = pd.DataFrame({
        "season": 2025,
        "game_id": pairs["game_id"].astype(str).to_numpy(),
        "week": pairs["week"].to_numpy(int),
        "kick_play_id": pd.to_numeric(
            pairs["play_id_kick"], errors="raise"
        ).to_numpy(float),
        "quarter": 4,
        "transition_type": np.where(tb, "touchback", "return"),
        "own_yardline": own.to_numpy(float),
        "elapsed_seconds": elapsed.to_numpy(float),
    }, columns=COLUMNS)
    return pool.sort_values(
        ["week", "game_id", "kick_play_id"]
    ).reset_index(drop=True)


def validate_frozen_pool(
    pool: pd.DataFrame, *, min_count: int = 300, min_games: int = 150
) -> pd.DataFrame:
    """Fail closed if data mixes seasons or nonordinary transitions."""
    if missing := set(COLUMNS).difference(pool.columns):
        raise ValueError(f"Frozen 2025 pool missing columns: {sorted(missing)}")
    if pool.empty or len(pool) < min_count or pool["game_id"].nunique() < min_games:
        raise ValueError("Insufficient 2025 ordinary kickoff donors.")
    if (not pool["season"].eq(2025).all()
            or not pool["quarter"].eq(4).all()
            or not pool["week"].between(1, 18).all()):
        raise ValueError("Frozen donors must be 2025 REG Q4 only.")
    if pool.duplicated(["game_id", "kick_play_id"]).any():
        raise ValueError("Duplicate kickoff donors.")
    if not pool["transition_type"].isin(["touchback", "return"]).all():
        raise ValueError("Pool includes nonordinary kickoff outcomes.")
    if not pool["own_yardline"].between(1, 99).all():
        raise ValueError("Invalid receiving first-down field position.")
    if not pool["elapsed_seconds"].between(0, 35).all():
        raise ValueError("Invalid kickoff elapsed time.")
    tb = pool["transition_type"].eq("touchback")
    if not pool.loc[tb, "own_yardline"].isin([20, 35]).all():
        raise ValueError("2025 touchbacks must start at own 20/35.")
    # Clock delta is between the kickoff row and the FIRST receiving snap,
    # not necessarily the kick's flight time. Historical PBP may record
    # small nonzero intervals even for a labeled touchback. Preserve the
    # observed elapsed distribution rather than overwrite it with zero.
    if not pool.loc[~tb, "elapsed_seconds"].between(1, 35).all():
        raise ValueError("Invalid ordinary return clock.")
    if min(tb.sum(), (~tb).sum()) < max(1, min_count // 20):
        raise ValueError("Need both touchbacks and returns.")
    return pool.copy()


def load_frozen_pool(path: Path = FROZEN_PATH) -> pd.DataFrame:
    """Only local, reviewed static data; no live PBP downloads in Streamlit."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing frozen 2025 kickoff data: {path}. "
            "Do not silently substitute 2024 kickoff returns."
        )
    pool = pd.read_csv(path)
    return validate_frozen_pool(pool)


def resolve_post_score_q4_2025(
    states: list[dict], pool: pd.DataFrame, *,
    rng: np.random.Generator, rules_year: int = 2026,
) -> list[dict]:
    """Sample an ENTIRE observed 2025 transition per pending Q4 kickoff."""
    if rules_year < 2025:
        raise ValueError("2025 donor transitions only support 2025+ rules.")
    validate_frozen_pool(pool, min_count=2, min_games=1)
    output = [dict(s) for s in states]
    eligible = [
        i for i, state in enumerate(output)
        if (
            int(state.get("qtr", 0)) == 4
            and float(state.get("is_kickoff", 0)) > 0.5
            and abs(float(state.get("_score_change_original", 0))) > 0
            and float(state.get("game_seconds_remaining", 0)) > 0
        )
    ]
    if not eligible:
        return output
    samples = pool.iloc[rng.integers(0, len(pool), size=len(eligible))]
    for i, (_, donor) in zip(eligible, samples.iterrows()):
        state = output[i]
        own = float(donor["own_yardline"])
        elapsed = float(donor["elapsed_seconds"])
        transition = str(donor["transition_type"])
        # A special 2026 kickoff from midfield has touchback placement own20.
        if (
            rules_year >= 2026
            and state.get("_kickoff_from_50", False)
            and transition == "touchback"
        ):
            own = 20.0
        clock = max(0.0, float(state["game_seconds_remaining"]) - elapsed)
        state.update({
            "is_kickoff": 0.0,
            "yardline_100": 100.0 - own,
            "down": 1.0,
            "ydstogo": min(10.0, 100.0 - own),
            "goal_to_go": float((100.0 - own) <= 10.0),
            "game_seconds_remaining": clock,
            "half_seconds_remaining": max(
                0.0, float(state["half_seconds_remaining"]) - elapsed
            ),
            "final_five_minutes": float(clock <= 300),
            "_kickoff_transition_type": transition,
        })
    return output


@contextmanager
def use_frozen_2025_kickoffs(
    engine: Mapping, pool: pd.DataFrame, *,
    seed: int = 2026, rules_year: int = 2026,
) -> Iterator[None]:
    """Isolated engine override; touchdown XP/2pt children reuse parent draw."""
    validate_frozen_pool(pool, min_count=2, min_games=1)
    original = engine["evaluate_post_play_states"]
    namespace = original.__globals__
    if namespace.get("evaluate_post_play_states") is not original:
        raise RuntimeError("Evaluator already overridden.")
    rng = np.random.default_rng(seed)

    def evaluate_with_2025(base, states):
        if int(base["qtr"]) != 4:
            return original(base, states)
        copied = resolve_post_score_q4_2025(
            states, pool, rng=rng, rules_year=rules_year,
        )
        return original(base, copied)

    namespace["evaluate_post_play_states"] = evaluate_with_2025
    try:
        yield
    finally:
        namespace["evaluate_post_play_states"] = original
