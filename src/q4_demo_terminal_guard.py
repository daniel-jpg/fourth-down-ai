"""Demo-only exact regulation 0:00 result resolution for private Q4 evaluation.

The published frozen engine and its models are never touched. This guard is
intentionally layered OUTSIDE the registered native-kickoff candidate so
future 2026/2027 study code has exactly its previous semantics. Only the
public Streamlit demo invokes it, on a per-session, per-request engine clone.

For unambiguous Q4 terminal states where the original team leads or trails,
the game's final result is 1 or 0, respectively. Ties are not resolved here
because overtime rules apply. Touchdowns needing a Try are left for the
engine's own untimed Try evaluator; resolved Try-child states are checked.
"""
from __future__ import annotations

from contextlib import contextmanager
from types import FunctionType
from collections.abc import Iterator, MutableMapping


def _demo_terminal_evaluate(base, states):
    """Function code is cloned into the isolated engine global namespace."""
    values = _q4_demo_unchanged_evaluator(base, states)
    if int(base["qtr"]) != 4:
        return values
    original_on_home_axis = bool(base["_wp_original_on_home_axis"])
    for index, state in enumerate(states):
        if (
            int(state["qtr"]) != 4
            or float(state["game_seconds_remaining"]) > 0.
            or int(state.get("_touchdown_original", 0)) != 0
        ):
            continue
        home_margin = float(state["home_score_differential"])
        original_margin = home_margin if original_on_home_axis else -home_margin
        if original_margin > 0.:
            values[index] = 1.
        elif original_margin < 0.:
            values[index] = 0.
        # Regulation tie: allow the overtime-aware estimator to handle.
    return values


@contextmanager
def use_q4_demo_terminal_guard(
    private_engine: MutableMapping,
) -> Iterator[None]:
    """Install a reversible *demo-only* rule layer before native input mapping."""
    original = private_engine["evaluate_post_play_states"]
    ns = original.__globals__
    if ns is not private_engine or ns.get("evaluate_post_play_states") is not original:
        raise RuntimeError("Q4 guard requires a clean private engine namespace.")
    if "_q4_demo_unchanged_evaluator" in ns:
        raise RuntimeError("Nested Q4 demo terminal guard is unsupported.")
    ns["_q4_demo_unchanged_evaluator"] = original
    cloned = FunctionType(
        _demo_terminal_evaluate.__code__, ns,
        "_q4_demo_terminal_evaluate",
    )
    ns["evaluate_post_play_states"] = cloned
    try:
        yield
    finally:
        ns["evaluate_post_play_states"] = original
        ns.pop("_q4_demo_unchanged_evaluator", None)
