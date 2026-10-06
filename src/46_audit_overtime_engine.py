import math
from pathlib import Path
import runpy
import subprocess
import tempfile

import numpy as np
import pandas as pd


CURRENT_ENGINE = (
    "src/42_build_decision_engine.py"
)


def load_committed_head_engine():

    source = subprocess.run(
        [
            "git",
            "show",
            (
                "HEAD:"
                "src/42_build_decision_engine.py"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout

    temp_path = None

    try:

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            prefix="audit_head_engine_",
            dir=".",
            delete=False,
        ) as handle:

            handle.write(source)

            temp_path = Path(
                handle.name
            )

        return runpy.run_path(
            str(temp_path)
        )

    finally:

        if temp_path is not None:

            temp_path.unlink(
                missing_ok=True
            )


print("=" * 80)
print("LOADING ENGINES")
print("=" * 80)

new = runpy.run_path(
    CURRENT_ENGINE
)

old = load_committed_head_engine()


normalize_state = new["normalize_state"]
make_wp_state = new["make_wp_state"]
evaluate_states = new["evaluate_post_play_states"]
predict_wp = new["predict_original_team_wp"]
recommend = new["recommend"]


PRODUCTION_ACTIONS = [
    "PUNT",
    "FIELD_GOAL",
    "NORMAL_GO_RUN",
    "NORMAL_GO_PASS",
]

GO_ACTIONS = {
    "NORMAL_GO_RUN",
    "NORMAL_GO_PASS",
}


hard_failures = []
review_flags = []
hard_checks = 0


def hard_check(
    name,
    condition,
    detail="",
):
    global hard_checks

    hard_checks += 1

    if condition:
        return

    hard_failures.append(
        (
            name,
            detail,
        )
    )


def action_value(
    table,
    action,
):
    rows = table[
        table["action"].eq(action)
    ]

    if len(rows) != 1:
        return np.nan

    value = rows.iloc[
        0
    ][
        "expected_win_probability"
    ]

    if pd.isna(value):
        return np.nan

    return float(value)


def production_table(
    result,
):
    return (
        result["results"]
        [
            result["results"]["action"]
            .isin(PRODUCTION_ACTIONS)
        ]
        .copy()
    )


def best_production_action(
    result,
):
    table = production_table(
        result
    )

    table = table[
        pd.notna(
            table[
                "expected_win_probability"
            ]
        )
    ]

    if table.empty:
        return None

    index = (
        table[
            "expected_win_probability"
        ]
        .astype(float)
        .idxmax()
    )

    return str(
        table.loc[
            index,
            "action",
        ]
    )


def scoring_state(
    base,
    score_change,
    possession_original=False,
    elapsed=5.0,
):
    return make_wp_state(
        base,
        possession_original=
            possession_original,
        yardline_100=np.nan,
        down=np.nan,
        ydstogo=np.nan,
        elapsed_seconds=elapsed,
        score_change_original=
            score_change,
        is_kickoff=True,
    )


def turnover_state(
    base,
    elapsed=5.0,
):
    return make_wp_state(
        base,
        possession_original=False,
        yardline_100=75.0,
        down=1.0,
        ydstogo=10.0,
        elapsed_seconds=elapsed,
        score_change_original=0.0,
        is_kickoff=False,
    )


print()
print("=" * 80)
print("1. EXACT OT RULE-LAYER TESTS")
print("=" * 80)


# ---------------------------------------------------------
# Missing OT phase must be rejected.
# ---------------------------------------------------------

try:

    normalize_state({
        "qtr": 5,
        "game_seconds_remaining": 300,
        "yardline_100": 50,
        "ydstogo": 2,
        "score_differential": 0,
        "is_home": 1,
    })

    missing_phase_rejected = False

except Exception:

    missing_phase_rejected = True


hard_check(
    "Missing OT phase rejected",
    missing_phase_rejected,
)


# ---------------------------------------------------------
# OT timeout defaults.
# ---------------------------------------------------------

base = normalize_state({
    "qtr": 5,
    "game_seconds_remaining": 300,
    "yardline_100": 50,
    "ydstogo": 2,
    "score_differential": 0,
    "is_home": 1,
    "ot_phase": "OPENING",
})

hard_check(
    "OT offense timeout default = 2",
    float(
        base[
            "posteam_timeouts_remaining"
        ]
    )
    == 2.0,
)

hard_check(
    "OT defense timeout default = 2",
    float(
        base[
            "defteam_timeouts_remaining"
        ]
    )
    == 2.0,
)


# ---------------------------------------------------------
# Opening possession:
# offensive score does NOT automatically win.
# ---------------------------------------------------------

opening_score = scoring_state(
    base,
    +7.0,
)

raw = float(
    predict_wp(
        [opening_score],
        bool(
            base["_wp_original_on_home_axis"]
        ),
        (
            base["site"]
            ==
            "NEUTRAL"
        ),
    )[0]
)

resolved = float(
    evaluate_states(
        base,
        [opening_score],
    )[0]
)

hard_check(
    "Opening offensive score remains nonterminal",
    abs(
        raw - resolved
    )
    < 1e-12,
    (
        f"raw={raw:.12f}, "
        f"resolved={resolved:.12f}"
    ),
)


# ---------------------------------------------------------
# Opening defensive score ends game in loss.
# ---------------------------------------------------------

opening_defensive_score = (
    scoring_state(
        base,
        -7.0,
        possession_original=True,
    )
)

resolved = float(
    evaluate_states(
        base,
        [
            opening_defensive_score
        ],
    )[0]
)

hard_check(
    "Opening defensive score = loss",
    abs(resolved) < 1e-12,
    f"value={resolved}",
)


# ---------------------------------------------------------
# Response possession, down 3.
# Scoreless turnover = loss.
# ---------------------------------------------------------

response_down3 = normalize_state({
    "qtr": 5,
    "game_seconds_remaining": 100,
    "yardline_100": 25,
    "ydstogo": 2,
    "score_differential": -3,
    "is_home": 1,
    "ot_phase": "RESPONSE",
})

miss = turnover_state(
    response_down3
)

resolved = float(
    evaluate_states(
        response_down3,
        [miss],
    )[0]
)

hard_check(
    "Response turnover while down 3 = loss",
    abs(resolved) < 1e-12,
    f"value={resolved}",
)


# ---------------------------------------------------------
# Response down 7:
# +6 loses, +7 continues, +8 wins.
# ---------------------------------------------------------

response_down7 = normalize_state({
    "qtr": 5,
    "game_seconds_remaining": 100,
    "yardline_100": 10,
    "ydstogo": 2,
    "score_differential": -7,
    "is_home": 1,
    "ot_phase": "RESPONSE",
})


plus6 = scoring_state(
    response_down7,
    +6.0,
)

value6 = float(
    evaluate_states(
        response_down7,
        [plus6],
    )[0]
)

hard_check(
    "Response down 7, +6 = loss",
    abs(value6) < 1e-12,
    f"value={value6}",
)


plus7 = scoring_state(
    response_down7,
    +7.0,
)

raw7 = float(
    predict_wp(
        [plus7],
        bool(
            response_down7[
                "_wp_original_on_home_axis"
            ]
        ),
        (
            response_down7["site"]
            ==
            "NEUTRAL"
        ),
    )[0]
)

resolved7 = float(
    evaluate_states(
        response_down7,
        [plus7],
    )[0]
)

hard_check(
    "Response down 7, +7 = continuation",
    abs(
        raw7 - resolved7
    )
    < 1e-12,
    (
        f"raw={raw7:.12f}, "
        f"resolved={resolved7:.12f}"
    ),
)


plus8 = scoring_state(
    response_down7,
    +8.0,
)

value8 = float(
    evaluate_states(
        response_down7,
        [plus8],
    )[0]
)

hard_check(
    "Response down 7, +8 = win",
    abs(
        value8 - 1.0
    )
    < 1e-12,
    f"value={value8}",
)


# ---------------------------------------------------------
# Sudden death score.
# ---------------------------------------------------------

sudden = normalize_state({
    "qtr": 5,
    "game_seconds_remaining": 100,
    "yardline_100": 25,
    "ydstogo": 2,
    "score_differential": 0,
    "is_home": 1,
    "ot_phase": "SUDDEN_DEATH",
})

sd_win = scoring_state(
    sudden,
    +3.0,
)

sd_loss = scoring_state(
    sudden,
    -3.0,
    possession_original=True,
)

sd_values = evaluate_states(
    sudden,
    [
        sd_win,
        sd_loss,
    ],
)

hard_check(
    "Sudden-death score = win",
    abs(
        float(sd_values[0])
        - 1.0
    )
    < 1e-12,
    f"value={sd_values[0]}",
)

hard_check(
    "Sudden-death opponent score = loss",
    abs(
        float(sd_values[1])
    )
    < 1e-12,
    f"value={sd_values[1]}",
)


# ---------------------------------------------------------
# Expiration tie = 0.5.
# ---------------------------------------------------------

expiration = normalize_state({
    "qtr": 5,
    "game_seconds_remaining": 1,
    "yardline_100": 16,
    "ydstogo": 14,
    "score_differential": -3,
    "is_home": 1,
    "ot_phase": "RESPONSE",
})

tie_at_zero = scoring_state(
    expiration,
    +3.0,
    elapsed=5.0,
)

tie_value = float(
    evaluate_states(
        expiration,
        [tie_at_zero],
    )[0]
)

hard_check(
    "Tie at OT expiration = 0.5",
    abs(
        tie_value - 0.5
    )
    < 1e-12,
    f"value={tie_value}",
)


print(
    f"Rule-layer checks executed: "
    f"{hard_checks}"
)


print()
print("-" * 80)
print("1B. CASE #15 OT RESPONSE DOWN-8 REGRESSION")
print("-" * 80)


case15_state = {
    "qtr": 5,
    "game_seconds_remaining": 1.0,
    "yardline_100": 50.0,
    "ydstogo": 10.0,
    "score_differential": -8.0,
    "is_home": 1,
    "ot_phase": "RESPONSE",
    "posteam_timeouts_remaining": 0.0,
    "defteam_timeouts_remaining": 0.0,
    "roof": "outdoors",
}


case15_base = normalize_state(
    case15_state
)


# A historical GO scoring donor may carry +6 or +7.
# At 0:00, either must still receive the two-point Try
# opportunity before the OT result is resolved.

case15_td6 = scoring_state(
    case15_base,
    +6.0,
    elapsed=5.0,
)

case15_td7 = scoring_state(
    case15_base,
    +7.0,
    elapsed=5.0,
)


case15_expected_td_value = (
    0.5
    *
    (
        639.0
        /
        1335.0
    )
)


case15_td6_value = float(
    evaluate_states(
        case15_base,
        [case15_td6],
    )[0]
)

case15_td7_value = float(
    evaluate_states(
        case15_base,
        [case15_td7],
    )[0]
)


hard_check(
    "Case #15 +6 TD receives two-point Try value",
    abs(
        case15_td6_value
        -
        case15_expected_td_value
    )
    < 1e-12,
    (
        f"resolved={case15_td6_value:.12f}, "
        f"expected={case15_expected_td_value:.12f}"
    ),
)

hard_check(
    "Case #15 +7 TD receives two-point Try value",
    abs(
        case15_td7_value
        -
        case15_expected_td_value
    )
    < 1e-12,
    (
        f"resolved={case15_td7_value:.12f}, "
        f"expected={case15_expected_td_value:.12f}"
    ),
)


case15_result = recommend(
    case15_state,
    n_simulations=4000,
    seed=42,
)

case15_table = production_table(
    case15_result
)

case15_pass = action_value(
    case15_table,
    "NORMAL_GO_PASS",
)

case15_punt = action_value(
    case15_table,
    "PUNT",
)


hard_check(
    "Case #15 PASS has positive game value",
    (
        np.isfinite(case15_pass)
        and
        case15_pass > 0.0
    ),
    f"PASS={case15_pass}",
)

hard_check(
    "Case #15 PASS beats PUNT",
    (
        np.isfinite(case15_pass)
        and
        np.isfinite(case15_punt)
        and
        case15_pass > case15_punt
    ),
    (
        f"PASS={case15_pass}, "
        f"PUNT={case15_punt}"
    ),
)

hard_check(
    "Case #15 recommends PASS",
    (
        best_production_action(
            case15_result
        )
        ==
        "NORMAL_GO_PASS"
    ),
    (
        "best="
        f"{best_production_action(case15_result)}"
    ),
)


print(
    "Case #15 values:",
    f"PASS={case15_pass:.8f}",
    f"PUNT={case15_punt:.8f}",
    "best="
    f"{best_production_action(case15_result)}",
)


print()
print("=" * 80)
print("2. RESPONSE-TIED VS SUDDEN-DEATH CONSISTENCY")
print("=" * 80)


common = {
    "qtr": 5,
    "game_seconds_remaining": 300,
    "yardline_100": 25,
    "ydstogo": 2,
    "score_differential": 0,
    "is_home": 1,
    "posteam_timeouts_remaining": 2,
    "defteam_timeouts_remaining": 2,
}


response_result = recommend(
    {
        **common,
        "ot_phase":
            "RESPONSE",
    },
    n_simulations=3000,
    seed=20261005,
)

sudden_result = recommend(
    {
        **common,
        "ot_phase":
            "SUDDEN_DEATH",
    },
    n_simulations=3000,
    seed=20261005,
)


response_table = (
    production_table(
        response_result
    )
    [
        [
            "action",
            "expected_win_probability",
        ]
    ]
    .rename(
        columns={
            "expected_win_probability":
                "response",
        }
    )
)

sudden_table = (
    production_table(
        sudden_result
    )
    [
        [
            "action",
            "expected_win_probability",
        ]
    ]
    .rename(
        columns={
            "expected_win_probability":
                "sudden",
        }
    )
)

phase_compare = response_table.merge(
    sudden_table,
    on="action",
    how="outer",
)

finite = phase_compare[
    ["response", "sudden"]
].notna().all(axis=1)

if finite.any():

    max_phase_diff = float(
        (
            phase_compare.loc[
                finite,
                "response",
            ]
            -
            phase_compare.loc[
                finite,
                "sudden",
            ]
        )
        .abs()
        .max()
    )

else:

    max_phase_diff = np.nan


print(
    phase_compare.to_string(
        index=False
    )
)

print(
    "\nMax response-tied vs "
    "sudden-death difference:",
    max_phase_diff,
)

hard_check(
    "Response tied and sudden death agree",
    (
        np.isfinite(
            max_phase_diff
        )
        and
        max_phase_diff < 1e-12
    ),
    f"max_diff={max_phase_diff}",
)


print()
print("=" * 80)
print("3. LARGE VALID-STATE OT SWEEP")
print("=" * 80)


phase_scores = {
    "OPENING":
        [0],

    "RESPONSE":
        [
            0,
            -3,
            -6,
            -7,
            -8,
        ],

    "SUDDEN_DEATH":
        [0],
}


clocks = [
    600,
    300,
    60,
    15,
    1,
]

yardlines = [
    80,
    50,
    30,
    15,
    8,
]

distances = [
    1,
    2,
    5,
    10,
]

home_values = [
    0,
    1,
]


sweep_states = 0
sweep_exceptions = 0
range_failures = 0
go_nan_failures = 0
no_action_failures = 0
heuristic_rechecks = 0


for phase, scores in (
    phase_scores.items()
):

    for score in scores:

        for clock in clocks:

            for yardline in yardlines:

                for distance in distances:

                    if distance > yardline:
                        continue

                    for is_home in (
                        home_values
                    ):

                        sweep_states += 1

                        state = {
                            "qtr": 5,

                            "game_seconds_remaining":
                                float(clock),

                            "yardline_100":
                                float(yardline),

                            "ydstogo":
                                float(distance),

                            "score_differential":
                                float(score),

                            "is_home":
                                int(is_home),

                            "ot_phase":
                                phase,

                            "posteam_timeouts_remaining":
                                2.0,

                            "defteam_timeouts_remaining":
                                2.0,
                        }

                        try:

                            result = recommend(
                                state,
                                n_simulations=250,
                                seed=(
                                    20261005
                                    +
                                    sweep_states
                                ),
                            )

                        except Exception as exc:

                            sweep_exceptions += 1

                            hard_failures.append(
                                (
                                    "OT sweep exception",
                                    (
                                        f"{state} -> "
                                        f"{type(exc).__name__}: "
                                        f"{exc}"
                                    ),
                                )
                            )

                            continue


                        table = production_table(
                            result
                        )


                        finite_table = table[
                            pd.notna(
                                table[
                                    "expected_win_probability"
                                ]
                            )
                        ]


                        # -----------------------------
                        # At least one action available.
                        # -----------------------------

                        if finite_table.empty:

                            no_action_failures += 1

                            hard_failures.append(
                                (
                                    "No finite production action",
                                    str(state),
                                )
                            )

                            continue


                        # -----------------------------
                        # All finite values in [0, 1].
                        # -----------------------------

                        values = (
                            finite_table[
                                "expected_win_probability"
                            ]
                            .astype(float)
                            .to_numpy()
                        )

                        bad = (
                            (~np.isfinite(values))
                            |
                            (values < -1e-9)
                            |
                            (values > 1.0 + 1e-9)
                        )

                        if bad.any():

                            range_failures += 1

                            hard_failures.append(
                                (
                                    "OT value outside [0,1]",
                                    (
                                        f"{state}\n"
                                        f"{table.to_string(index=False)}"
                                    ),
                                )
                            )


                        # -----------------------------
                        # Normal GO support contract.
                        #
                        # PASS remains available across
                        # normal fourth-down distances.
                        #
                        # Designed RUN is supported only
                        # at ydstogo <= 3 because longer-
                        # yardage run samples are too sparse.
                        # -----------------------------

                        pass_value = action_value(
                            table,
                            "NORMAL_GO_PASS",
                        )

                        if not np.isfinite(
                            pass_value
                        ):

                            go_nan_failures += 1

                            hard_failures.append(
                                (
                                    "GO pass missing/NaN",
                                    f"{state}",
                                )
                            )


                        run_value = action_value(
                            table,
                            "NORMAL_GO_RUN",
                        )

                        run_should_exist = (
                            distance <= 3
                        )

                        run_is_finite = np.isfinite(
                            run_value
                        )

                        if (
                            run_is_finite
                            !=
                            run_should_exist
                        ):

                            go_nan_failures += 1

                            expected = (
                                "finite"
                                if run_should_exist
                                else "unavailable/NaN"
                            )

                            hard_failures.append(
                                (
                                    "GO run support mismatch",
                                    (
                                        f"expected {expected} | "
                                        f"value={run_value} | "
                                        f"{state}"
                                    ),
                                )
                            )


                        # -----------------------------
                        # Late response, down >= 6:
                        # GO should normally dominate.
                        # These are heuristic flags,
                        # not automatic failures.
                        # -----------------------------

                        if (
                            phase
                            == "RESPONSE"
                            and
                            score <= -6
                            and
                            clock <= 15
                        ):

                            cheap_best = (
                                best_production_action(
                                    result
                                )
                            )

                            cheap_fg = action_value(
                                table,
                                "FIELD_GOAL",
                            )

                            cheap_punt = action_value(
                                table,
                                "PUNT",
                            )

                            needs_recheck = (
                                cheap_best
                                not in GO_ACTIONS
                                or
                                (
                                    np.isfinite(
                                        cheap_fg
                                    )
                                    and
                                    cheap_fg > 0.03
                                )
                                or
                                (
                                    np.isfinite(
                                        cheap_punt
                                    )
                                    and
                                    cheap_punt > 0.05
                                )
                            )


                            if needs_recheck:

                                heuristic_rechecks += 1

                                recheck_result = recommend(
                                    state,
                                    n_simulations=4000,
                                    seed=(
                                        4000000
                                        +
                                        sweep_states
                                    ),
                                )

                                recheck_table = (
                                    production_table(
                                        recheck_result
                                    )
                                )

                                best_action = (
                                    best_production_action(
                                        recheck_result
                                    )
                                )

                                fg_value = (
                                    action_value(
                                        recheck_table,
                                        "FIELD_GOAL",
                                    )
                                )

                                punt_value = (
                                    action_value(
                                        recheck_table,
                                        "PUNT",
                                    )
                                )

                                go_run_value = (
                                    action_value(
                                        recheck_table,
                                        "NORMAL_GO_RUN",
                                    )
                                )

                                go_pass_value = (
                                    action_value(
                                        recheck_table,
                                        "NORMAL_GO_PASS",
                                    )
                                )


                                if (
                                    best_action
                                    not in GO_ACTIONS
                                ):

                                    review_flags.append(
                                        (
                                            "Must-score state "
                                            "still did not "
                                            "choose GO after "
                                            "4,000-simulation "
                                            "recheck",
                                            (
                                                f"{state} | "
                                                f"cheap_best="
                                                f"{cheap_best} | "
                                                f"recheck_best="
                                                f"{best_action} | "
                                                f"RUN="
                                                f"{go_run_value:.6f} | "
                                                f"PASS="
                                                f"{go_pass_value:.6f} | "
                                                f"PUNT="
                                                f"{punt_value:.6f}"
                                            ),
                                        )
                                    )


                                if (
                                    np.isfinite(
                                        fg_value
                                    )
                                    and
                                    fg_value > 0.03
                                ):

                                    review_flags.append(
                                        (
                                            "Late trailing FG "
                                            "value > 3% after "
                                            "4,000-simulation "
                                            "recheck",
                                            (
                                                f"{state} | "
                                                f"FG="
                                                f"{fg_value:.6f}"
                                            ),
                                        )
                                    )


                                if (
                                    np.isfinite(
                                        punt_value
                                    )
                                    and
                                    punt_value > 0.05
                                ):

                                    review_flags.append(
                                        (
                                            "Late trailing punt "
                                            "value > 5% after "
                                            "4,000-simulation "
                                            "recheck",
                                            (
                                                f"{state} | "
                                                f"PUNT="
                                                f"{punt_value:.6f}"
                                            ),
                                        )
                                    )



print(
    "OT states tested:",
    sweep_states,
)

print(
    "Exceptions:",
    sweep_exceptions,
)

print(
    "Out-of-range failures:",
    range_failures,
)

print(
    "GO NaN failures:",
    go_nan_failures,
)

print(
    "No-action failures:",
    no_action_failures,
)

print(
    "High-simulation heuristic rechecks:",
    heuristic_rechecks,
)


print()
print("=" * 80)
print("4. TIMEOUT COMBINATION TESTS")
print("=" * 80)


timeout_scenarios = [
    {
        "ot_phase": "OPENING",
        "score_differential": 0,
        "game_seconds_remaining": 420,
    },
    {
        "ot_phase": "RESPONSE",
        "score_differential": -3,
        "game_seconds_remaining": 120,
    },
    {
        "ot_phase": "SUDDEN_DEATH",
        "score_differential": 0,
        "game_seconds_remaining": 60,
    },
]


timeout_cases = 0


for scenario in timeout_scenarios:

    for offense_to in [
        0,
        1,
        2,
    ]:

        for defense_to in [
            0,
            1,
            2,
        ]:

            timeout_cases += 1

            state = {
                "qtr": 5,

                "game_seconds_remaining":
                    scenario[
                        "game_seconds_remaining"
                    ],

                "yardline_100": 40.0,

                "ydstogo": 3.0,

                "score_differential":
                    scenario[
                        "score_differential"
                    ],

                "is_home": 1,

                "ot_phase":
                    scenario[
                        "ot_phase"
                    ],

                "posteam_timeouts_remaining":
                    float(offense_to),

                "defteam_timeouts_remaining":
                    float(defense_to),
            }

            try:

                result = recommend(
                    state,
                    n_simulations=500,
                    seed=(
                        800000
                        +
                        timeout_cases
                    ),
                )

                table = production_table(
                    result
                )

                finite_values = (
                    table[
                        "expected_win_probability"
                    ]
                    .dropna()
                    .astype(float)
                )

                hard_check(
                    "Timeout case finite action",
                    len(
                        finite_values
                    )
                    > 0,
                    str(state),
                )

                hard_check(
                    "Timeout case bounds",
                    (
                        (
                            finite_values
                            >= -1e-9
                        )
                        &
                        (
                            finite_values
                            <= 1.0 + 1e-9
                        )
                    ).all(),
                    str(state),
                )

            except Exception as exc:

                hard_failures.append(
                    (
                        "Timeout test exception",
                        (
                            f"{state} -> "
                            f"{type(exc).__name__}: "
                            f"{exc}"
                        ),
                    )
                )


print(
    "Timeout combinations tested:",
    timeout_cases,
)


print()
print("=" * 80)
print("5. REGULATION REGRESSION AGAINST COMMITTED HEAD")
print("=" * 80)


reg_clock = {
    1: 3300,
    2: 1500,
    3: 1200,
    4: 600,
}


reg_states = []


for qtr in [
    1,
    2,
    3,
    4,
]:

    for score in [
        -7,
        0,
        7,
    ]:

        for yardline in [
            75,
            45,
            25,
        ]:

            for distance in [
                1,
                3,
            ]:

                for is_home in [
                    0,
                    1,
                ]:

                    reg_states.append({
                        "qtr":
                            qtr,

                        "game_seconds_remaining":
                            float(
                                reg_clock[qtr]
                            ),

                        "yardline_100":
                            float(yardline),

                        "ydstogo":
                            float(distance),

                        "score_differential":
                            float(score),

                        "is_home":
                            is_home,

                        "posteam_timeouts_remaining":
                            2.0,

                        "defteam_timeouts_remaining":
                            2.0,
                    })


reg_failures = 0


for index, state in enumerate(
    reg_states,
    1,
):

    seed = (
        900000
        +
        index
    )

    old_result = old[
        "recommend"
    ](
        state,
        n_simulations=300,
        seed=seed,
    )

    new_result = recommend(
        state,
        n_simulations=300,
        seed=seed,
    )


    old_table = (
        old_result[
            "results"
        ]
        [
            [
                "action",
                "expected_win_probability",
            ]
        ]
        .rename(
            columns={
                "expected_win_probability":
                    "old",
            }
        )
    )

    new_table = (
        new_result[
            "results"
        ]
        [
            [
                "action",
                "expected_win_probability",
            ]
        ]
        .rename(
            columns={
                "expected_win_probability":
                    "new",
            }
        )
    )


    compare = old_table.merge(
        new_table,
        on="action",
        how="outer",
    )


    same_values = np.allclose(
        compare["old"]
        .to_numpy(dtype=float),

        compare["new"]
        .to_numpy(dtype=float),

        equal_nan=True,
        atol=1e-12,
        rtol=0,
    )


    same_recommendation = (
        old_result[
            "recommendation"
        ]
        ==
        new_result[
            "recommendation"
        ]
    )


    if not (
        same_values
        and
        same_recommendation
    ):

        reg_failures += 1

        hard_failures.append(
            (
                "Regulation changed",
                (
                    f"{state}\n"
                    f"old rec="
                    f"{old_result['recommendation']}\n"
                    f"new rec="
                    f"{new_result['recommendation']}\n"
                    f"{compare.to_string(index=False)}"
                ),
            )
        )


print(
    "Regulation states compared:",
    len(reg_states),
)

print(
    "Regulation mismatches:",
    reg_failures,
)


print()
print("=" * 80)
print("6. REGULATION CLOCK-KILL RULE TESTS")
print("=" * 80)


clock_kill_cases = [
    {
        "name":
            "1st down, 0 defensive TO, 120s = terminal",
        "seconds": 120,
        "def_timeouts": 0,
        "down": 1,
        "score_diff": 1,
        "possession_original": True,
        "is_home": 1,
        "terminal": True,
    },
    {
        "name":
            "1st down, 1 defensive TO, 80s = terminal",
        "seconds": 80,
        "def_timeouts": 1,
        "down": 1,
        "score_diff": 1,
        "possession_original": True,
        "is_home": 1,
        "terminal": True,
    },
    {
        "name":
            "1st down, 2 defensive TO, 40s = terminal",
        "seconds": 40,
        "def_timeouts": 2,
        "down": 1,
        "score_diff": 1,
        "possession_original": True,
        "is_home": 1,
        "terminal": True,
    },
    {
        "name":
            "Away offense orientation, 2 defensive TO, 40s = terminal",
        "seconds": 40,
        "def_timeouts": 2,
        "down": 1,
        "score_diff": 1,
        "possession_original": True,
        "is_home": 0,
        "terminal": True,
    },
    {
        "name":
            "1st down, 0 defensive TO, 121s = nonterminal",
        "seconds": 121,
        "def_timeouts": 0,
        "down": 1,
        "score_diff": 1,
        "possession_original": True,
        "is_home": 1,
        "terminal": False,
    },
    {
        "name":
            "1st down, 1 defensive TO, 81s = nonterminal",
        "seconds": 81,
        "def_timeouts": 1,
        "down": 1,
        "score_diff": 1,
        "possession_original": True,
        "is_home": 1,
        "terminal": False,
    },
    {
        "name":
            "1st down, 2 defensive TO, 41s = nonterminal",
        "seconds": 41,
        "def_timeouts": 2,
        "down": 1,
        "score_diff": 1,
        "possession_original": True,
        "is_home": 1,
        "terminal": False,
    },
    {
        "name":
            "1st down, 3 defensive TO = nonterminal",
        "seconds": 1,
        "def_timeouts": 3,
        "down": 1,
        "score_diff": 1,
        "possession_original": True,
        "is_home": 1,
        "terminal": False,
    },
    {
        "name":
            "2nd down, 0 defensive TO, 80s = terminal",
        "seconds": 80,
        "def_timeouts": 0,
        "down": 2,
        "score_diff": 1,
        "possession_original": True,
        "is_home": 1,
        "terminal": True,
    },
    {
        "name":
            "3rd down, 0 defensive TO, 40s = terminal",
        "seconds": 40,
        "def_timeouts": 0,
        "down": 3,
        "score_diff": 1,
        "possession_original": True,
        "is_home": 1,
        "terminal": True,
    },
    {
        "name":
            "4th down cannot be clock-killed",
        "seconds": 1,
        "def_timeouts": 0,
        "down": 4,
        "score_diff": 1,
        "possession_original": True,
        "is_home": 1,
        "terminal": False,
    },
    {
        "name":
            "Tied game is not clock-kill terminal",
        "seconds": 40,
        "def_timeouts": 0,
        "down": 1,
        "score_diff": 0,
        "possession_original": True,
        "is_home": 1,
        "terminal": False,
    },
    {
        "name":
            "Trailing offense is not clock-kill terminal",
        "seconds": 40,
        "def_timeouts": 0,
        "down": 1,
        "score_diff": -1,
        "possession_original": True,
        "is_home": 1,
        "terminal": False,
    },
    {
        "name":
            "Opponent possession is not clock-kill terminal",
        "seconds": 40,
        "def_timeouts": 0,
        "down": 1,
        "score_diff": 1,
        "possession_original": False,
        "is_home": 1,
        "terminal": False,
    },
]


for case in clock_kill_cases:

    base = normalize_state({
        "qtr": 4,

        "game_seconds_remaining":
            float(
                case["seconds"]
            ),

        "yardline_100": 57.0,

        "ydstogo": 1.0,

        "score_differential":
            float(
                case["score_diff"]
            ),

        "is_home":
            int(
                case["is_home"]
            ),

        "posteam_timeouts_remaining":
            3.0,

        "defteam_timeouts_remaining":
            float(
                case["def_timeouts"]
            ),
    })


    state = make_wp_state(

        base,

        possession_original=
            bool(
                case[
                    "possession_original"
                ]
            ),

        yardline_100=55.0,

        down=
            float(
                case["down"]
            ),

        ydstogo=10.0,

        elapsed_seconds=0.0,

        score_change_original=0.0,

        is_kickoff=False,
    )


    raw = float(
        predict_wp(
            [state],
            bool(
                base["_wp_original_on_home_axis"]
            ),
            (
                base["site"]
                ==
                "NEUTRAL"
            ),
        )[0]
    )


    resolved = float(
        evaluate_states(
            base,
            [state],
        )[0]
    )


    if case["terminal"]:

        condition = (
            abs(
                resolved
                -
                1.0
            )
            <
            1e-12
        )

    else:

        condition = (
            abs(
                resolved
                -
                raw
            )
            <
            1e-12
        )


    hard_check(
        (
            "Regulation clock-kill: "
            +
            case["name"]
        ),
        condition,
        (
            f"case={case} | "
            f"raw={raw:.12f} | "
            f"resolved={resolved:.12f}"
        ),
    )


print(
    "Regulation clock-kill checks executed:",
    len(
        clock_kill_cases
    ),
)


print()
print("=" * 80)
print("7. HELD-OUT 2025 REGULAR-SEASON TRAILING OT PLAYS")
print("=" * 80)


split = pd.read_parquet(
    "data/fourth_down_modeling_split.parquet"
)


holdout = split[
    (split["season"] == 2025)
    &
    (split["qtr"] >= 5)
    &
    (split["week"] <= 18)
    &
    (split["score_differential"] < 0)
].copy()


holdout = holdout.sort_values(
    [
        "game_id",
        "play_id",
    ]
)


holdout_rows = []


for index, row in (
    holdout.iterrows()
):

    offense_to = (
        row.get(
            "posteam_timeouts_remaining",
            2.0,
        )
    )

    defense_to = (
        row.get(
            "defteam_timeouts_remaining",
            2.0,
        )
    )


    if pd.isna(offense_to):
        offense_to = 2.0

    if pd.isna(defense_to):
        defense_to = 2.0


    state = {
        "qtr":
            int(row["qtr"]),

        "game_seconds_remaining":
            float(
                row[
                    "game_seconds_remaining"
                ]
            ),

        "yardline_100":
            float(
                row[
                    "yardline_100"
                ]
            ),

        "ydstogo":
            float(
                row[
                    "ydstogo"
                ]
            ),

        "score_differential":
            float(
                row[
                    "score_differential"
                ]
            ),

        "is_home":
            int(
                row[
                    "is_home"
                ]
            ),

        "ot_phase":
            "RESPONSE",

        "posteam_timeouts_remaining":
            float(offense_to),

        "defteam_timeouts_remaining":
            float(defense_to),
    }


    result = recommend(
        state,
        n_simulations=3000,
        seed=(
            1000000
            +
            int(row["play_id"])
        ),
    )


    table = production_table(
        result
    )

    best_action = (
        best_production_action(
            result
        )
    )


    values = {
        action:
            action_value(
                table,
                action,
            )
        for action
        in PRODUCTION_ACTIONS
    }


    holdout_rows.append({
        "game_id":
            row["game_id"],

        "play_id":
            row["play_id"],

        "clock":
            row[
                "game_seconds_remaining"
            ],

        "yardline_100":
            row[
                "yardline_100"
            ],

        "ydstogo":
            row[
                "ydstogo"
            ],

        "score_diff":
            row[
                "score_differential"
            ],

        "actual":
            row.get(
                "action",
                None,
            ),

        "engine_best":
            best_action,

        "PUNT":
            values["PUNT"],

        "FG":
            values["FIELD_GOAL"],

        "GO_RUN":
            values["NORMAL_GO_RUN"],

        "GO_PASS":
            values["NORMAL_GO_PASS"],
    })


    if (
        float(
            row[
                "score_differential"
            ]
        )
        <= -6
        and
        best_action
        not in GO_ACTIONS
    ):

        review_flags.append(
            (
                "2025 held-out "
                "down >= 6 did not choose GO",
                (
                    f"{row['game_id']} "
                    f"play {row['play_id']} "
                    f"best={best_action}"
                ),
            )
        )


holdout_report = pd.DataFrame(
    holdout_rows
)


if not holdout_report.empty:

    print(
        holdout_report.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

else:

    print(
        "No trailing regular-season "
        "2025 OT fourth downs found."
    )


print()
print("=" * 80)
print("8. FINAL AUDIT SUMMARY")
print("=" * 80)


print(
    "Hard checks executed:",
    hard_checks,
)

print(
    "Large OT sweep states:",
    sweep_states,
)

print(
    "Timeout cases:",
    timeout_cases,
)

print(
    "Regulation regression states:",
    len(reg_states),
)

print(
    "Held-out trailing OT plays:",
    len(holdout_rows),
)

print()

print(
    "HARD FAILURES:",
    len(hard_failures),
)

print(
    "REVIEW FLAGS:",
    len(review_flags),
)


if hard_failures:

    print()
    print(
        "---------- HARD FAILURES ----------"
    )

    for number, (
        name,
        detail,
    ) in enumerate(
        hard_failures,
        1,
    ):

        print()
        print(
            f"{number}. {name}"
        )

        if detail:
            print(detail)


if review_flags:

    print()
    print(
        "---------- REVIEW FLAGS ----------"
    )

    for number, (
        name,
        detail,
    ) in enumerate(
        review_flags,
        1,
    ):

        print()
        print(
            f"{number}. {name}"
        )

        if detail:
            print(detail)


print()

if hard_failures:

    print(
        "RESULT: HARD FAILURES FOUND"
    )

    raise SystemExit(1)

else:

    print(
        "RESULT: ALL HARD CHECKS PASSED"
    )

    if review_flags:

        print(
            "Heuristic review flags remain; "
            "inspect them before declaring OT complete."
        )

    else:

        print(
            "No heuristic review flags found."
        )
