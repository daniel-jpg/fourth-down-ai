"""Opt-in Q4 kickoff continuation for controlled decision-engine experiments.

The production engine is not modified. This context manager temporarily wraps
its state evaluator, preserving the existing touchdown Try logic: parent kickoff
states are sampled exactly once, BEFORE the evaluator constructs XP / 2PT child
states. Those children inherit the SAME kickoff outcome for valid comparisons.

Scope: ordinary Q4 post-score kickoffs only. The donor pool is 2024-only. Rare
return touchdowns, onside kicks/recoveries, penalties, and 2025+ strategy shifts
are not calibrated. Explicit touchback-rate scenarios are NOT live forecasts.
"""

from contextlib import contextmanager
from collections.abc import Iterator, Mapping

import numpy as np
import pandas as pd

from .kickoff_transition import resolve_q4_kickoffs


@contextmanager
def use_q4_kickoff_scenario(
    engine: Mapping,
    pool: pd.DataFrame,
    *,
    touchback_rate: float,
    seed: int = 2026,
    rules_year: int = 2026,
) -> Iterator[None]:
    """Opt into one seeded Q4 kickoff assumption, restoring engine on exit.

    `engine` can be the dict returned by runpy.run_path(...).
    A separate kickoff RNG does not change the original action-outcome draws.
    This is intentionally a temporary experimental hook, not a default rule.
    """
    if not np.isfinite(touchback_rate) or not 0 <= touchback_rate <= 1:
        raise ValueError("touchback_rate must be between 0 and 1")
    if rules_year < 2024:
        raise ValueError("rules_year must be 2024 or later")

    original = engine["evaluate_post_play_states"]
    namespace = original.__globals__

    if namespace.get("evaluate_post_play_states") is not original:
        raise RuntimeError("Evaluator already overridden; nested scenarios unsupported")

    kickoff_rng = np.random.default_rng(seed)

    def experimental_evaluator(base, states):
        # Non-Q4 states are bit-for-bit unchanged. In Q4 the underlying sampler
        # changes ONLY eligible post-score kickoff-pending states.
        if int(base["qtr"]) != 4:
            return original(base, states)
        sampled = resolve_q4_kickoffs(
            states,
            pool,
            touchback_rate=touchback_rate,
            rng=kickoff_rng,
            rules_year=rules_year,
        )
        return original(base, sampled)

    namespace["evaluate_post_play_states"] = experimental_evaluator
    try:
        yield
    finally:
        namespace["evaluate_post_play_states"] = original
