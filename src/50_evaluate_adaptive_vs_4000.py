import importlib.util
from pathlib import Path

import nflreadpy as nfl
import numpy as np
import pandas as pd

from sklearn.metrics import (
    log_loss,
    roc_auc_score,
)


# =========================================================
# Load frozen production decision engine.
# =========================================================

print("\nLoading frozen decision engine...")

spec = importlib.util.spec_from_file_location(
    "fourth_down_engine",
    "src/42_build_decision_engine.py",
)

engine = importlib.util.module_from_spec(spec)
spec.loader.exec_module(engine)

print("Decision engine loaded.")


# =========================================================
# Load 2023-2024 validation fourth downs.
# =========================================================

validation = pd.read_parquet(
    "data/fourth_down_modeling_split.parquet"
)

validation = validation[
    validation["split"] == "validation"
].copy()

print(
    f"\nValidation fourth downs: "
    f"{len(validation):,}"
)

print(
    validation["season"]
    .value_counts()
    .sort_index()
    .to_string()
)


# =========================================================
# Load roof information from raw 2023-2024 PBP.
# =========================================================

print("\nLoading 2023-2024 PBP...")

pbp = (
    nfl.load_pbp([2023, 2024])
    .sort([
        "game_id",
        "play_id",
    ])
    .to_pandas()
)

if "roof" in pbp.columns:

    game_roof = (
        pbp[
            [
                "game_id",
                "roof",
            ]
        ]
        .dropna(
            subset=["roof"]
        )
        .drop_duplicates(
            subset=["game_id"]
        )
        .set_index(
            "game_id"
        )["roof"]
    )

    if "roof" not in validation.columns:
        validation["roof"] = (
            validation["game_id"]
            .map(game_roof)
        )
    else:
        missing = validation["roof"].isna()

        validation.loc[
            missing,
            "roof",
        ] = (
            validation.loc[
                missing,
                "game_id",
            ]
            .map(game_roof)
        )


# =========================================================
# Load OT phase metadata.
# =========================================================

OT_STATES_PATH = Path(
    "data/overtime_states.parquet"
)

if not OT_STATES_PATH.exists():

    raise FileNotFoundError(
        "Missing data/overtime_states.parquet. "
        "Build it first with: "
        "python src/45_build_overtime_state_dataset.py"
    )


ot_states = pd.read_parquet(
    OT_STATES_PATH
)

ot_states = ot_states[
    ot_states["season"].isin(
        [2023, 2024]
    )
].copy()


OT_PHASE_MAP = {
    "OPENING_POSSESSION": "OPENING",
    "SECOND_POSSESSION": "RESPONSE",
    "SUDDEN_DEATH": "SUDDEN_DEATH",
}


OT_STATE_LOOKUP = {}

for _, row in ot_states.iterrows():

    phase = OT_PHASE_MAP.get(
        str(row["ot_phase"])
    )

    if phase is None:
        continue

    key = (
        str(row["game_id"]),
        int(
            round(
                float(row["play_id"])
            )
        ),
    )

    OT_STATE_LOOKUP[key] = {
        "ot_phase":
            phase,

        "ot_format":
            (
                "POSTSEASON"
                if str(
                    row["season_type"]
                ) == "POST"
                else
                "REGULAR_SEASON"
            ),
    }


# =========================================================
# State conversion.
# =========================================================

def valid_number(value):

    try:
        return bool(
            np.isfinite(
                float(value)
            )
        )
    except (
        TypeError,
        ValueError,
    ):
        return False


def numeric_or(
    value,
    default,
):

    if valid_number(value):
        return float(value)

    return float(default)


def row_to_state(row):

    required = [
        "qtr",
        "game_seconds_remaining",
        "yardline_100",
        "ydstogo",
        "score_differential",
        "is_home",
    ]

    for column in required:

        if (
            column not in row.index
            or
            not valid_number(
                row[column]
            )
        ):
            return None

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
                row["yardline_100"]
            ),

        "ydstogo":
            float(
                row["ydstogo"]
            ),

        "score_differential":
            float(
                row[
                    "score_differential"
                ]
            ),

        "posteam_timeouts_remaining":
            numeric_or(
                row.get(
                    "posteam_timeouts_remaining",
                    3,
                ),
                3,
            ),

        "defteam_timeouts_remaining":
            numeric_or(
                row.get(
                    "defteam_timeouts_remaining",
                    3,
                ),
                3,
            ),

        "is_home":
            int(row["is_home"]),

        "spread_line":
            numeric_or(
                row.get(
                    "spread_line",
                    0,
                ),
                0,
            ),

        "total_line":
            numeric_or(
                row.get(
                    "total_line",
                    45,
                ),
                45,
            ),

        "roof":
            (
                str(
                    row.get(
                        "roof",
                        "outdoors",
                    )
                )
                if pd.notna(
                    row.get(
                        "roof",
                        np.nan,
                    )
                )
                else
                "outdoors"
            ),
    }

    if (
        "goal_to_go" in row.index
        and
        valid_number(
            row["goal_to_go"]
        )
    ):
        state["goal_to_go"] = int(
            row["goal_to_go"]
        )

    if state["qtr"] >= 5:

        key = (
            str(row["game_id"]),
            int(
                round(
                    float(row["play_id"])
                )
            ),
        )

        metadata = (
            OT_STATE_LOOKUP.get(key)
        )

        if metadata is None:
            raise RuntimeError(
                "Missing OT metadata for "
                f"{key[0]} play {key[1]}"
            )

        state["ot_phase"] = (
            metadata["ot_phase"]
        )

        state["ot_format"] = (
            metadata["ot_format"]
        )

        state["ot_period"] = max(
            1,
            state["qtr"] - 4,
        )

    return state


# =========================================================
# Validation-only policy simulation stability study.
#
# Development data: 2023-2024 validation only.
#
# Candidate adaptive rule being evaluated:
#
#   1. Start with 300 simulations/action.
#   2. If top-two gap < 1.96 combined MC SE,
#      rerun from scratch with 1,200 simulations/action.
#
# Nothing here changes production behavior.
# =========================================================

BASE_SIMULATIONS = 300
HIGH_SIMULATIONS = 1200

UNCERTAINTY_Z = 1.96

SAMPLE_PER_SEASON = 150

SEED_SAMPLE_2023 = 480023
SEED_SAMPLE_2024 = 480024

SEED_BASE_A = 481000
SEED_BASE_B = 482000

SEED_HIGH_A = 483000
SEED_HIGH_B = 484000


# =========================================================
# Build validation policy population.
# =========================================================

valid_indices = []


for index, row in validation.iterrows():

    state = row_to_state(row)

    if state is None:
        continue

    try:

        engine.normalize_state(
            state
        )

    except (
        TypeError,
        ValueError,
    ):

        continue

    valid_indices.append(
        index
    )


policy_df = (
    validation.loc[
        valid_indices
    ]
    .copy()
)


sample_parts = []


for season, seed in [
    (
        2023,
        SEED_SAMPLE_2023,
    ),
    (
        2024,
        SEED_SAMPLE_2024,
    ),
]:

    season_rows = (
        policy_df[
            policy_df["season"]
            ==
            season
        ]
    )

    n = min(
        SAMPLE_PER_SEASON,
        len(season_rows),
    )

    sample_parts.append(
        season_rows.sample(
            n=n,
            random_state=seed,
        )
    )


policy_sample = (
    pd.concat(
        sample_parts,
        ignore_index=True,
    )
    .reset_index(
        drop=True
    )
)


print(
    "\nVALIDATION POLICY SIMULATION STABILITY"
)

print(
    f"Eligible validation rows: "
    f"{len(policy_df):,}"
)

print(
    f"Study rows: "
    f"{len(policy_sample):,}"
)

print(
    policy_sample[
        "season"
    ]
    .value_counts()
    .sort_index()
    .to_string()
)


# =========================================================
# Recommendation summary helper.
# =========================================================

def recommendation_summary(
    state,
    n_simulations,
    seed,
):

    result = engine.recommend(

        state,

        n_simulations=
            n_simulations,

        seed=
            seed,
    )


    eligible = (
        result["results"][
            result["results"][
                "eligible"
            ]
        ]
        .sort_values(
            "expected_win_probability",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )


    if len(eligible) == 0:

        return {
            "action":
                None,

            "gap_pct":
                np.nan,

            "combined_se_pct":
                np.nan,

            "z_gap":
                np.nan,
        }


    best = eligible.iloc[0]


    if len(eligible) < 2:

        return {
            "action":
                best["action"],

            "gap_pct":
                np.nan,

            "combined_se_pct":
                np.nan,

            "z_gap":
                np.inf,
        }


    second = eligible.iloc[1]


    gap = float(
        best[
            "expected_win_probability"
        ]
        -
        second[
            "expected_win_probability"
        ]
    )


    best_se = float(
        best["mc_se"]
    )

    second_se = float(
        second["mc_se"]
    )


    if (
        np.isfinite(best_se)
        and
        np.isfinite(second_se)
    ):

        combined_se = float(
            np.sqrt(
                best_se ** 2
                +
                second_se ** 2
            )
        )

    else:

        combined_se = np.nan


    if (
        np.isfinite(combined_se)
        and
        combined_se > 0.0
    ):

        z_gap = float(
            gap
            /
            combined_se
        )

    elif gap > 0.0:

        z_gap = np.inf

    else:

        z_gap = np.nan


    return {
        "action":
            best["action"],

        "gap_pct":
            100.0 * gap,

        "combined_se_pct":
            (
                100.0 * combined_se
                if np.isfinite(
                    combined_se
                )
                else
                np.nan
            ),

        "z_gap":
            z_gap,
    }


# =========================================================
# Two independent realizations of:
#
#   A) fixed 300 simulations
#   B) candidate adaptive procedure
#
# Comparing A vs B seed realizations tells us how stable the
# numerical recommendation is.
# =========================================================


# =========================================================
# Strategy-level adaptive simulation stability study.
#
# Compare:
#
#   A) production-matched ACTION trigger
#      - normal GO pass/run, FG, punt
#      - fake plays excluded
#
#   B) candidate STRATEGY trigger
#      - GO FOR IT = better of normal pass/run
#      - FIELD GOAL
#      - PUNT
#
# Both use:
#
#   300 simulations/action initially
#   1,200 simulations/action on trigger
#   trigger when z < 1.96
#
# Validation only: 2023-2024.
# 2025 remains untouched.
# =========================================================


def production_rows(
    result,
):

    rows = (
        result["results"][
            result["results"][
                "eligible"
            ]
        ]
        .copy()
    )


    rows = rows[
        ~rows[
            "action"
        ]
        .astype(str)
        .str.startswith(
            "FAKE_"
        )
    ]


    return (
        rows
        .sort_values(
            "expected_win_probability",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )


def gap_statistics(
    best_wp,
    best_se,
    second_wp,
    second_se,
):

    gap = float(
        best_wp
        -
        second_wp
    )


    if (
        np.isfinite(best_se)
        and
        np.isfinite(second_se)
    ):

        combined_se = float(
            np.sqrt(
                best_se ** 2
                +
                second_se ** 2
            )
        )

    else:

        combined_se = np.nan


    if (
        np.isfinite(combined_se)
        and
        combined_se > 0.0
    ):

        z_gap = float(
            gap
            /
            combined_se
        )

    elif gap > 0.0:

        z_gap = np.inf

    else:

        z_gap = np.nan


    return {
        "gap_pct":
            100.0 * gap,

        "combined_se_pct":
            (
                100.0 * combined_se
                if np.isfinite(
                    combined_se
                )
                else
                np.nan
            ),

        "z_gap":
            z_gap,
    }


def action_to_strategy(
    action,
):

    if action in {
        "NORMAL_GO_PASS",
        "NORMAL_GO_RUN",
    }:

        return "GO FOR IT"


    if action == "FIELD_GOAL":

        return "FIELD GOAL"


    if action == "PUNT":

        return "PUNT"


    return action


def action_summary_from_result(
    result,
):

    rows = production_rows(
        result
    )


    if len(rows) == 0:

        return {
            "action":
                None,

            "strategy":
                None,

            "gap_pct":
                np.nan,

            "combined_se_pct":
                np.nan,

            "z_gap":
                np.nan,
        }


    best = rows.iloc[0]


    if len(rows) < 2:

        return {
            "action":
                best["action"],

            "strategy":
                action_to_strategy(
                    best["action"]
                ),

            "gap_pct":
                np.nan,

            "combined_se_pct":
                np.nan,

            "z_gap":
                np.inf,
        }


    second = rows.iloc[1]


    stats = gap_statistics(

        float(
            best[
                "expected_win_probability"
            ]
        ),

        float(
            best["mc_se"]
        ),

        float(
            second[
                "expected_win_probability"
            ]
        ),

        float(
            second["mc_se"]
        ),
    )


    return {
        "action":
            best["action"],

        "strategy":
            action_to_strategy(
                best["action"]
            ),

        **stats,
    }


def strategy_summary_from_result(
    result,
):

    rows = production_rows(
        result
    )


    strategy_rows = []


    go = (
        rows[
            rows[
                "action"
            ]
            .isin([
                "NORMAL_GO_PASS",
                "NORMAL_GO_RUN",
            ])
        ]
        .sort_values(
            "expected_win_probability",
            ascending=False,
        )
    )


    if len(go) > 0:

        go_best = go.iloc[0]

        strategy_rows.append({

            "strategy":
                "GO FOR IT",

            "action":
                go_best[
                    "action"
                ],

            "wp":
                float(
                    go_best[
                        "expected_win_probability"
                    ]
                ),

            "mc_se":
                float(
                    go_best[
                        "mc_se"
                    ]
                ),
        })


    for action, label in [
        (
            "FIELD_GOAL",
            "FIELD GOAL",
        ),
        (
            "PUNT",
            "PUNT",
        ),
    ]:

        action_rows = rows[
            rows["action"]
            ==
            action
        ]


        if len(action_rows) == 0:
            continue


        action_row = action_rows.iloc[0]

        strategy_rows.append({

            "strategy":
                label,

            "action":
                action,

            "wp":
                float(
                    action_row[
                        "expected_win_probability"
                    ]
                ),

            "mc_se":
                float(
                    action_row[
                        "mc_se"
                    ]
                ),
        })


    if len(strategy_rows) == 0:

        return {
            "strategy":
                None,

            "action":
                None,

            "gap_pct":
                np.nan,

            "combined_se_pct":
                np.nan,

            "z_gap":
                np.nan,
        }


    strategies = (
        pd.DataFrame(
            strategy_rows
        )
        .sort_values(
            "wp",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )


    best = strategies.iloc[0]


    if len(strategies) < 2:

        return {
            "strategy":
                best[
                    "strategy"
                ],

            "action":
                best[
                    "action"
                ],

            "gap_pct":
                np.nan,

            "combined_se_pct":
                np.nan,

            "z_gap":
                np.inf,
        }


    second = strategies.iloc[1]


    stats = gap_statistics(

        float(
            best["wp"]
        ),

        float(
            best["mc_se"]
        ),

        float(
            second["wp"]
        ),

        float(
            second["mc_se"]
        ),
    )


    return {
        "strategy":
            best[
                "strategy"
            ],

        "action":
            best[
                "action"
            ],

        **stats,
    }


def run_recommendation(
    state,
    n_simulations,
    seed,
):

    return engine.recommend(

        state,

        n_simulations=
            n_simulations,

        seed=
            seed,
    )


def should_trigger(
    summary,
):

    z_gap = summary[
        "z_gap"
    ]


    return bool(
        np.isfinite(
            z_gap
        )
        and
        z_gap
        <
        UNCERTAINTY_Z
    )


def resolved(
    z_gap,
):

    return bool(
        np.isinf(
            z_gap
        )
        or
        (
            np.isfinite(
                z_gap
            )
            and
            z_gap
            >=
            UNCERTAINTY_Z
        )
    )



# =========================================================
# Adaptive production policy vs fixed 4,000-simulation
# reference.
#
# The adaptive recommendations were already generated by
# Script 49 using only 2023-2024 validation data.
#
# This script evaluates whether those recommendations agree
# with a substantially larger independent Monte Carlo
# calculation.
#
# 2025 remains untouched.
# =========================================================

REFERENCE_SIMULATIONS = 4000
REFERENCE_SEED = 485000


adaptive_path = (
    "data/"
    "evaluation_validation_strategy_"
    "simulation_stability.parquet"
)


adaptive = pd.read_parquet(
    adaptive_path
)


if len(adaptive) != len(policy_sample):

    raise RuntimeError(
        "Adaptive results and validation sample "
        "have different lengths."
    )


if not np.array_equal(

    adaptive[
        "season"
    ].astype(int).to_numpy(),

    policy_sample[
        "season"
    ].astype(int).to_numpy(),
):

    raise RuntimeError(
        "Adaptive results do not align with "
        "the validation sample."
    )


def result_value_maps(
    result,
):

    rows = production_rows(
        result
    )


    action_values = {

        str(row["action"]):
            float(
                row[
                    "expected_win_probability"
                ]
            )

        for _, row in (
            rows.iterrows()
        )
    }


    strategy_values = {}


    go_values = [

        action_values[action]

        for action in [
            "NORMAL_GO_PASS",
            "NORMAL_GO_RUN",
        ]

        if action in action_values
    ]


    if len(go_values) > 0:

        strategy_values[
            "GO FOR IT"
        ] = max(
            go_values
        )


    if "FIELD_GOAL" in action_values:

        strategy_values[
            "FIELD GOAL"
        ] = action_values[
            "FIELD_GOAL"
        ]


    if "PUNT" in action_values:

        strategy_values[
            "PUNT"
        ] = action_values[
            "PUNT"
        ]


    return (
        action_values,
        strategy_values,
    )


def regret_pct(
    choice,
    values,
):

    if (
        choice is None
        or
        choice not in values
        or
        len(values) == 0
    ):

        return np.nan


    return 100.0 * (
        max(
            values.values()
        )
        -
        values[
            choice
        ]
    )


def is_resolved(
    z_gap,
):

    return bool(
        np.isinf(
            z_gap
        )
        or
        (
            np.isfinite(
                z_gap
            )
            and
            z_gap
            >=
            UNCERTAINTY_Z
        )
    )


records = []


print(
    "\nADAPTIVE VS FIXED 4,000 REFERENCE"
)

print(
    f"Validation rows: "
    f"{len(policy_sample):,}"
)

print(
    f"Reference simulations/action: "
    f"{REFERENCE_SIMULATIONS:,}"
)


for i, row in (
    policy_sample.iterrows()
):

    if i % 25 == 0:

        print(
            f"Evaluating "
            f"{i:,}/"
            f"{len(policy_sample):,}..."
        )


    state = row_to_state(
        row
    )


    reference_result = (
        run_recommendation(

            state,

            REFERENCE_SIMULATIONS,

            REFERENCE_SEED
            +
            i,
        )
    )


    reference_action = (
        action_summary_from_result(
            reference_result
        )
    )


    reference_strategy = (
        strategy_summary_from_result(
            reference_result
        )
    )


    (
        action_values,
        strategy_values,
    ) = result_value_maps(
        reference_result
    )


    adaptive_row = (
        adaptive.iloc[i]
    )


    a_action = (
        adaptive_row[
            "current_a_action"
        ]
    )

    b_action = (
        adaptive_row[
            "current_b_action"
        ]
    )


    a_strategy = (
        adaptive_row[
            "current_a_strategy"
        ]
    )

    b_strategy = (
        adaptive_row[
            "current_b_strategy"
        ]
    )


    records.append({

        "season":
            int(
                row["season"]
            ),

        "game_id":
            row.get(
                "game_id",
                None,
            ),

        "play_id":
            row.get(
                "play_id",
                None,
            ),


        "reference_action":
            reference_action[
                "action"
            ],

        "reference_strategy":
            reference_strategy[
                "strategy"
            ],

        "reference_action_z":
            reference_action[
                "z_gap"
            ],

        "reference_strategy_z":
            reference_strategy[
                "z_gap"
            ],


        "adaptive_a_action":
            a_action,

        "adaptive_b_action":
            b_action,

        "adaptive_a_strategy":
            a_strategy,

        "adaptive_b_strategy":
            b_strategy,


        "adaptive_a_action_z":
            adaptive_row[
                "current_a_action_z"
            ],

        "adaptive_b_action_z":
            adaptive_row[
                "current_b_action_z"
            ],

        "adaptive_a_strategy_z":
            adaptive_row[
                "current_a_strategy_z"
            ],

        "adaptive_b_strategy_z":
            adaptive_row[
                "current_b_strategy_z"
            ],


        "adaptive_a_action_agrees":
            (
                a_action
                ==
                reference_action[
                    "action"
                ]
            ),

        "adaptive_b_action_agrees":
            (
                b_action
                ==
                reference_action[
                    "action"
                ]
            ),

        "adaptive_a_strategy_agrees":
            (
                a_strategy
                ==
                reference_strategy[
                    "strategy"
                ]
            ),

        "adaptive_b_strategy_agrees":
            (
                b_strategy
                ==
                reference_strategy[
                    "strategy"
                ]
            ),


        "adaptive_a_action_regret_pct":
            regret_pct(
                a_action,
                action_values,
            ),

        "adaptive_b_action_regret_pct":
            regret_pct(
                b_action,
                action_values,
            ),

        "adaptive_a_strategy_regret_pct":
            regret_pct(
                a_strategy,
                strategy_values,
            ),

        "adaptive_b_strategy_regret_pct":
            regret_pct(
                b_strategy,
                strategy_values,
            ),
    })


results = pd.DataFrame(
    records
)


# =========================================================
# Resolution states.
# =========================================================

for name in [
    "adaptive_a_action",
    "adaptive_b_action",
    "adaptive_a_strategy",
    "adaptive_b_strategy",
]:

    results[
        f"{name}_resolved"
    ] = (
        results[
            f"{name}_z"
        ]
        .map(
            is_resolved
        )
    )


results[
    "reference_strategy_resolved"
] = (
    results[
        "reference_strategy_z"
    ]
    .map(
        is_resolved
    )
)


results[
    "adaptive_strategy_consensus"
] = (
    results[
        "adaptive_a_strategy"
    ]
    ==
    results[
        "adaptive_b_strategy"
    ]
)


results[
    "adaptive_action_consensus"
] = (
    results[
        "adaptive_a_action"
    ]
    ==
    results[
        "adaptive_b_action"
    ]
)


results[
    "consensus_strategy_agrees"
] = (
    results[
        "adaptive_strategy_consensus"
    ]
    &
    (
        results[
            "adaptive_a_strategy"
        ]
        ==
        results[
            "reference_strategy"
        ]
    )
)


results[
    "consensus_action_agrees"
] = (
    results[
        "adaptive_action_consensus"
    ]
    &
    (
        results[
            "adaptive_a_action"
        ]
        ==
        results[
            "reference_action"
        ]
    )
)


# =========================================================
# Reporting helpers.
# =========================================================

def pct(
    series,
):

    return (
        100.0
        *
        float(
            series.mean()
        )
    )


def finite_summary(
    series,
):

    values = (
        pd.to_numeric(
            series,
            errors="coerce",
        )
        .dropna()
        .to_numpy(
            dtype=float
        )
    )


    if len(values) == 0:

        return (
            np.nan,
            np.nan,
            np.nan,
        )


    return (
        float(
            np.median(
                values
            )
        ),
        float(
            np.quantile(
                values,
                0.90,
            )
        ),
        float(
            np.max(
                values
            )
        ),
    )


# =========================================================
# Main comparison.
# =========================================================

print(
    "\nAGREEMENT WITH 4,000-SIMULATION REFERENCE"
)

print(
    "Adaptive seed A — strategy agreement: "
    f"{pct(results['adaptive_a_strategy_agrees']):.2f}%"
)

print(
    "Adaptive seed B — strategy agreement: "
    f"{pct(results['adaptive_b_strategy_agrees']):.2f}%"
)

print(
    "Adaptive seed A — action agreement:   "
    f"{pct(results['adaptive_a_action_agrees']):.2f}%"
)

print(
    "Adaptive seed B — action agreement:   "
    f"{pct(results['adaptive_b_action_agrees']):.2f}%"
)


# =========================================================
# Consensus.
# =========================================================

strategy_consensus = (
    results[
        results[
            "adaptive_strategy_consensus"
        ]
    ]
)


action_consensus = (
    results[
        results[
            "adaptive_action_consensus"
        ]
    ]
)


print(
    "\nADAPTIVE SEED CONSENSUS"
)

print(
    "Same strategy across adaptive seeds: "
    f"{100.0 * len(strategy_consensus) / len(results):.2f}%"
)

if len(strategy_consensus) > 0:

    print(
        "4,000 reference agreement when adaptive "
        "seeds agree on strategy: "
        f"{100.0 * strategy_consensus['consensus_strategy_agrees'].mean():.2f}%"
    )


print(
    "Same action across adaptive seeds:   "
    f"{100.0 * len(action_consensus) / len(results):.2f}%"
)

if len(action_consensus) > 0:

    print(
        "4,000 reference agreement when adaptive "
        "seeds agree on action: "
        f"{100.0 * action_consensus['consensus_action_agrees'].mean():.2f}%"
    )


# =========================================================
# Resolved adaptive decisions.
# =========================================================

for label in [
    "a",
    "b",
]:

    resolved_rows = (
        results[
            results[
                f"adaptive_{label}_strategy_resolved"
            ]
        ]
    )


    print(
        f"\nAdaptive seed {label.upper()} "
        f"strategy-resolved rows: "
        f"{len(resolved_rows):,}"
    )


    if len(resolved_rows) > 0:

        print(
            "Agreement with 4,000 reference "
            "when strategy-resolved: "
            f"{100.0 * resolved_rows[f'adaptive_{label}_strategy_agrees'].mean():.2f}%"
        )


# =========================================================
# 4,000-reference uncertainty.
# =========================================================

print(
    "\n4,000-SIMULATION REFERENCE"
)

print(
    "Reference strategy unresolved: "
    f"{100.0 * (~results['reference_strategy_resolved']).mean():.2f}%"
)


# =========================================================
# Fixed-reference regret.
#
# This is NOT causal regret. It is only the WP difference,
# under the fixed 4,000-simulation calculation, between its
# top option and the adaptive option.
# =========================================================

for label in [
    "a",
    "b",
]:

    median_regret, p90_regret, max_regret = (
        finite_summary(
            results[
                f"adaptive_{label}_strategy_regret_pct"
            ]
        )
    )


    print(
        f"\nAdaptive seed {label.upper()} "
        f"strategy reference-gap"
    )

    print(
        f"Median: {median_regret:.3f} pp"
    )

    print(
        f"P90:    {p90_regret:.3f} pp"
    )

    print(
        f"Max:    {max_regret:.3f} pp"
    )


# =========================================================
# Disagreement diagnostics.
# =========================================================

strategy_disagreement = (
    ~results[
        "adaptive_a_strategy_agrees"
    ]
    |
    ~results[
        "adaptive_b_strategy_agrees"
    ]
)


disagreement_rows = (
    results[
        strategy_disagreement
    ]
)


print(
    "\nSTRATEGY DISAGREEMENTS"
)

print(
    f"Rows where either adaptive seed differs "
    f"from 4,000 reference: "
    f"{len(disagreement_rows):,}"
)


if len(disagreement_rows) > 0:

    reference_gaps = (
        disagreement_rows[
            "reference_strategy_z"
        ]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna()
    )


    if len(reference_gaps) > 0:

        print(
            "Median 4,000 reference strategy z "
            "on disagreement rows: "
            f"{reference_gaps.median():.2f}"
        )


print(
    "\nIMPORTANT"
)

print(
    "The 4,000-simulation run is a higher-budget "
    "Monte Carlo reference, not ground truth."
)

print(
    "This experiment uses only 2023-2024 validation data."
)

print(
    "2025 remains held out."
)


output_path = (
    "data/"
    "evaluation_validation_adaptive_vs_4000.parquet"
)


results.to_parquet(
    output_path,
    index=False,
)


print(
    f"\nSaved: {output_path}"
)
