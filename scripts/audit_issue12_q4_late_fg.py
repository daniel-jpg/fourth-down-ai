"""Issue #12: read-only Q4 near-expiry field-goal action-value diagnostic.

Never fetches football outcomes, refits models or touches protected 2026/2027
evaluations. Uses private copies of the frozen engine Python namespace and
the Q4 native-standard-35 input representation already used by the demo.

Output is model behavior, not validated NFL truth or causal coaching advice.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.kickoff_q4_native_features_candidate import (
    native_q4_normal_kickoff,
    use_q4_native_kickoff_candidate,
)
from src.streamlit_kickoff_preview import (
    isolated_engine_for_preview,
    isolated_engine_from_loaded_engine,
)

CLOCKS = (20, 45, 60, 120)
MARGINS = (-4, -2, -1)
SITES = ("AWAY", "HOME")
SEEDS = (43, 101)
N = 1200
REPORT = Path("artifacts/issue12_q4_late_fg_diagnostic.json")


def make_state(clock: int, margin: int, site: str) -> dict:
    if clock <= 0 or clock > 900 or margin not in MARGINS or site not in SITES:
        raise ValueError("Unsupported synthetic test state.")
    return {
        "qtr": 4, "game_seconds_remaining": float(clock),
        "yardline_100": 1., "ydstogo": 1.,
        "score_differential": float(margin),
        "posteam_timeouts_remaining": 2.,
        "defteam_timeouts_remaining": 2.,
        "site": site, "roof": "outdoors",
    }


def terminal_win_if_decided(margin: float, points: float = 3.) -> float | None:
    """Rule-only at 0:00: no overtime claim for a game ending tied."""
    post = margin + points
    if post > 0:
        return 1.
    if post < 0:
        return 0.
    return None


def group_fg_outcome(state: dict) -> str:
    """Stratify simulator states; special failed kicks remain a separate bucket."""
    score = float(state.get("_score_change_original", 0.))
    original_possession = bool(state.get("_possession_original", False))
    kickoff = bool(state.get("is_kickoff", False))
    if score == 3. and kickoff and not original_possession:
        return "made"
    if score < 0:
        return "opponent_scored_on_kick"
    if original_possession:
        return "kicking_team_recovered"
    return "missed_or_other_failure"


def sample_fg_branches(live_engine: dict, state: dict, seed: int) -> dict:
    """Measure FG result decomposition in an ISOLATED private Q4 context."""
    private = isolated_engine_from_loaded_engine(live_engine)
    base = private["normalize_state"](state)
    observed = []
    with use_q4_native_kickoff_candidate(private):
        original = private["evaluate_post_play_states"]

        def trace(b, states):
            wp = original(b, states)
            # The top-level FG call has exactly N states. Nested Try children
            # are evaluated separately and must not be double counted.
            if len(states) == N and not observed:
                observed.extend(zip(states, map(float, wp)))
            return wp

        private["evaluate_post_play_states"] = trace
        try:
            fg = private["simulate_field_goal"](
                base, N, np.random.default_rng(seed + 500),
            )
        finally:
            private["evaluate_post_play_states"] = original

    assert len(observed) == N, "Failed to capture all and only FG outcomes."
    assert math.isclose(
        float(np.mean([value for _, value in observed])),
        float(fg["q_value"]), abs_tol=1e-12, rel_tol=0.,
    ), "FG branch decomposition fails to reconstruct overall value."
    grouped = defaultdict(list)
    for post_state, wp in observed:
        assert 0 <= wp <= 1 and math.isfinite(wp)
        assert 0 <= float(post_state["game_seconds_remaining"]) <= state["game_seconds_remaining"]
        grouped[group_fg_outcome(post_state)].append(wp)

    return {
        "fg_sample_wp_pct": round(100 * float(fg["q_value"]), 3),
        "fg_monte_carlo_se_pp": round(100 * float(fg["mc_se"]), 3),
        "outcomes": {
            name: {
                "n": len(values),
                "mean_conditional_wp_pct": round(100 * float(np.mean(values)), 3),
                "contribution_to_fg_wp_pp": round(
                    100 * float(sum(values)) / N, 3,
                ),
            } for name, values in sorted(grouped.items())
        },
    }


def conditional_kickoff_probe(live_engine: dict, state: dict) -> dict:
    """Use the same frozen WP artifact, with and without native Q4 encoding."""
    private = isolated_engine_from_loaded_engine(live_engine)
    base = private["normalize_state"](state)
    make = private["make_wp_state"]
    evaluate = private["evaluate_post_play_states"]
    made_5 = make(
        base, possession_original=False, yardline_100=np.nan,
        down=np.nan, ydstogo=np.nan,
        elapsed_seconds=5., score_change_original=3., is_kickoff=True,
    )
    native = native_q4_normal_kickoff(made_5)
    wps = evaluate(base, [made_5, native])
    assert all(0 <= float(wp) <= 1 for wp in wps)
    zero_time = make(
        base, possession_original=False, yardline_100=np.nan,
        down=np.nan, ydstogo=np.nan,
        elapsed_seconds=float(state["game_seconds_remaining"]) + 5.,
        score_change_original=3., is_kickoff=True,
    )
    assert float(zero_time["game_seconds_remaining"]) == 0.
    raw_zero_wp = float(evaluate(base, [zero_time])[0])
    rule = terminal_win_if_decided(float(state["score_differential"]))
    return {
        "made_fg_after_5s_clock": float(native["game_seconds_remaining"]),
        "conditional_legacy_made_fg_wp_pct": round(100 * float(wps[0]), 3),
        "conditional_native_made_fg_wp_pct": round(100 * float(wps[1]), 3),
        "rule_only_if_clock_ends": rule,
        "frozen_wp_at_0s_pct": round(100 * raw_zero_wp, 3),
        "zero_clock_rule_mismatch": (
            rule is not None and abs(raw_zero_wp - rule) > 1e-6
        ),
    }


def run() -> None:
    engine = isolated_engine_for_preview()
    original_recommend = engine["recommend"]
    original_eval = engine["evaluate_post_play_states"]
    cases = []
    print("ISSUE12_READ_ONLY_AUDIT: 48 states, 1,200 simulations/action; model unchanged", flush=True)
    for clock in CLOCKS:
        for margin in MARGINS:
            for site in SITES:
                state = make_state(clock, margin, site)
                kickoff = conditional_kickoff_probe(engine, state)
                for seed in SEEDS:
                    isolated = isolated_engine_from_loaded_engine(engine)
                    with use_q4_native_kickoff_candidate(isolated):
                        result = isolated["recommend"](
                            state, n_simulations=N, seed=seed,
                        )
                    rows = result["results"].set_index("action")
                    fg = rows.loc["FIELD_GOAL"]
                    assert bool(fg["eligible"])
                    eligible = rows.loc[rows["eligible"]]
                    assert eligible["expected_win_probability"].between(0, 1).all()
                    go = eligible.loc[eligible.index.isin(
                        ("NORMAL_GO_PASS", "NORMAL_GO_RUN")
                    ), "expected_win_probability"]
                    assert len(go)
                    branches = sample_fg_branches(engine, state, seed)
                    record = {
                        "clock": clock, "margin_before_fg": margin,
                        "score_margin_after_made_fg": margin + 3,
                        "site": site, "seed": seed,
                        "go_wp_pct": round(100 * float(go.max()), 3),
                        "fg_action_wp_pct": round(
                            100 * float(fg["expected_win_probability"]), 3,
                        ),
                        "fg_action_mc_se_pp": round(100 * float(fg["mc_se"]), 3),
                        "best_strategy": (
                            "GO FOR IT" if "GO" in str(result["recommendation"])
                            else str(result["recommendation"])
                        ),
                        "kickoff": kickoff, "fg_branch_audit": branches,
                    }
                    cases.append(record)
                    print(
                        "LATE_FG_CASE clock={:3d} before={:+d} site={:4s} seed={:3d} "
                        "GO={:6.2f}% FG={:6.2f}% FG_SE={:.2f}pp "
                        "made5_native={:.2f}% zero_clock_frozen={:.2f}% rule={}".format(
                            clock, margin, site, seed,
                            record["go_wp_pct"], record["fg_action_wp_pct"],
                            record["fg_action_mc_se_pp"],
                            kickoff["conditional_native_made_fg_wp_pct"],
                            kickoff["frozen_wp_at_0s_pct"],
                            kickoff["rule_only_if_clock_ends"],
                        ), flush=True,
                    )
    assert len(cases) == 48
    assert engine["recommend"] is original_recommend
    assert engine["evaluate_post_play_states"] is original_eval
    # Regression anchor for the existing demo: no changes to its model.
    anchor = [r for r in cases if (
        r["clock"], r["margin_before_fg"], r["site"], r["seed"]
    ) == (45, -4, "AWAY", 43)]
    assert len(anchor) == 1
    # Distinguish improbable predictions from confirmed rule violations.
    mismatches = [r for r in cases if r["kickoff"]["zero_clock_rule_mismatch"]]
    report = {
        "issue": 12,
        "design": "read-only frozen-engine near-end Q4 FG decomposition",
        "observed_NFL_games": False,
        "2026_or_2027_outcomes_accessed": False,
        "changes_to_models_or_app": False,
        "simulations_per_action": N,
        "anchors": anchor,
        "zero_clock_rule_mismatch_count": len(mismatches),
        "cases": cases,
        "limits": (
            "Simulated modeled continuation is not independent NFL validation. "
            "Onside/rare kickoffs are not directly represented by the standard-35 "
            "pending-kickoff WP feature vector. Monte Carlo SE measures sampling "
            "noise, not model specification or extrapolation error."
        ),
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    print("ISSUE12_REPORT_PATH", REPORT, flush=True)
    print(
        "ISSUE12_SUMMARY cases={} zero_clock_rule_mismatches={} "
        "anchor_fg={:.2f}% anchor_go={:.2f}%".format(
            len(cases), len(mismatches),
            anchor[0]["fg_action_wp_pct"],
            anchor[0]["go_wp_pct"],
        ), flush=True,
    )
    print("ISSUE12_COMPLETED_READ_ONLY: no promotion, no reserved outcomes.", flush=True)


if __name__ == "__main__":
    run()
