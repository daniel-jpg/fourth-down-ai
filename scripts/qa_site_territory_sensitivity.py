"""Focused real-engine QA for aggressive deep-own-territory and site effects.

Diagnostic-only. Does not retrain, modify frozen data, run future gates,
or replace the preregistered 2026/2027 validation protocols. Two seeds
and repeated representative field positions reveal Monte Carlo sensitivity,
NOT total model calibration or causal policy correctness.
"""
from __future__ import annotations

import math

from src.streamlit_kickoff_preview import (
    isolated_engine_for_preview,
    isolated_engine_from_loaded_engine,
    preview_native_kickoff_feature_candidate,
)

NORMAL = {
    "NORMAL_GO_PASS": "GO FOR IT",
    "NORMAL_GO_RUN": "GO FOR IT",
    "FIELD_GOAL": "FIELD GOAL",
    "PUNT": "PUNT",
}


def state(*, quarter, clock, yardline, togo, score, site):
    return {
        "qtr": quarter,
        "game_seconds_remaining": float(clock),
        "yardline_100": float(yardline),
        "ydstogo": float(togo),
        "score_differential": float(score),
        "posteam_timeouts_remaining": 3.,
        "defteam_timeouts_remaining": 3.,
        "site": site,
        "roof": "outdoors",
    }


def value(result):
    rows = result["results"]
    wp = {}
    for _, row in rows.iterrows():
        action = str(row["action"])
        if not bool(row["eligible"]):
            continue
        assert action in NORMAL, "Unsupported action must never be ranked"
        number = float(row["expected_win_probability"])
        assert 0 <= number <= 1 and math.isfinite(number)
        label = NORMAL[action]
        wp[label] = max(wp.get(label, -1.), number)
    ranked = sorted(wp.items(), key=lambda kv: kv[1], reverse=True)
    assert len(ranked) >= 2
    a, b = ranked[:2]
    return a[0], 100 * a[1], b[0], 100 * b[1], 100 * (a[1] - b[1])


def sample(engine, *, state_value, seed, sims, is_q4):
    if is_q4:
        private = isolated_engine_from_loaded_engine(engine)
        result = preview_native_kickoff_feature_candidate(
            state_value, simulations=sims, action_seed=seed,
            isolated_engine=private,
        )
    else:
        result = engine["recommend"](
            state_value, n_simulations=sims, seed=seed,
        )
    return value(result)


def main():
    engine = isolated_engine_for_preview()
    unchanged_recommend = engine["recommend"]
    unchanged_evaluate = engine["evaluate_post_play_states"]

    print("Deep-own-territory Q1, tied, fourth-and-3, 05:30 in Q1", flush=True)
    own_results = {}
    for own_yardline in (5, 10, 20, 35):
        for seed, sims in ((43, 1200), (101, 1200)):
            result = sample(
                engine,
                state_value=state(
                    quarter=1, clock=3030, yardline=100-own_yardline,
                    togo=3, score=0, site="NEUTRAL",
                ),
                seed=seed, sims=sims, is_q4=False,
            )
            own_results[(own_yardline, seed)] = result
            print(
                f"OWN{own_yardline:02d} seed={seed:3d} "
                f"{result[0]:<11} {result[1]:6.2f}% vs "
                f"{result[2]:<11} {result[3]:6.2f}% "
                f"edge {result[4]:+6.2f} pp",
                flush=True,
            )

    print("Q4, 4:59, fourth-and-4 opp6, trailing by 4, three timeouts", flush=True)
    site_results = {}
    for site in ("AWAY", "NEUTRAL", "HOME"):
        for seed in (43, 101):
            result = sample(
                engine,
                state_value=state(
                    quarter=4, clock=299, yardline=6,
                    togo=4, score=-4, site=site,
                ),
                seed=seed, sims=1200, is_q4=True,
            )
            site_results[(site, seed)] = result
            print(
                f"{site:<7} seed={seed:3d} "
                f"{result[0]:<11} {result[1]:6.2f}% vs "
                f"{result[2]:<11} {result[3]:6.2f}% "
                f"edge {result[4]:+6.2f} pp",
                flush=True,
            )

    assert engine["recommend"] is unchanged_recommend
    assert engine["evaluate_post_play_states"] is unchanged_evaluate
    assert len(own_results) == 8
    assert len(site_results) == 6
    assert site_results[("AWAY",43)][0] == "FIELD GOAL"

    # The neutral-site WP evaluation is explicitly the average of the
    # mirrored HOME/AWAY axes. Same frozen RNG and action outcomes must
    # make this identity hold for GO and FG to numerical precision.
    for seed in (43, 101):
        by_site = {}
        for site in ("AWAY", "NEUTRAL", "HOME"):
            row = site_results[(site, seed)]
            by_site[site] = {row[0]: row[1], row[2]: row[3]}
            assert set(by_site[site]) == {"GO FOR IT", "FIELD GOAL"}
        for strategy in ("GO FOR IT", "FIELD GOAL"):
            expected = (
                by_site["HOME"][strategy]
                + by_site["AWAY"][strategy]
            ) / 2.
            assert math.isclose(
                by_site["NEUTRAL"][strategy], expected,
                rel_tol=0., abs_tol=1e-8,
            ), (
                seed, strategy, expected,
                by_site["NEUTRAL"][strategy],
            )
    print(
        "SITE SYMMETRY PASS: NEUTRAL is the exact mean of the "
        "HOME/AWAY WP axes for GO and FG with both frozen seeds.",
        flush=True,
    )

    # Detect, rather than explain away, large site/field-situation effects.
    own5_choices = {own_results[(5, s)][0] for s in (43,101)}
    site_home_choices = {site_results[("HOME",s)][0] for s in (43,101)}
    print(f"Own-5 choice across seeds: {sorted(own5_choices)}", flush=True)
    print(f"Q4 home-site choice across seeds: {sorted(site_home_choices)}", flush=True)
    print(
        "QA PASS: results finite, eligibility respected, model namespaces "
        "unchanged. Plausibility requires separate empirical support review.",
        flush=True,
    )


if __name__ == "__main__":
    main()
