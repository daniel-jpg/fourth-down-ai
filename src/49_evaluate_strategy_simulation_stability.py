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


records = []


print(
    "\nACTION VS STRATEGY ADAPTIVE TRIGGER"
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


    # -----------------------------------------------------
    # Base 300-simulation realizations.
    # -----------------------------------------------------

    base_result_a = run_recommendation(

        state,

        BASE_SIMULATIONS,

        SEED_BASE_A
        +
        i,
    )


    base_result_b = run_recommendation(

        state,

        BASE_SIMULATIONS,

        SEED_BASE_B
        +
        i,
    )


    base_action_a = (
        action_summary_from_result(
            base_result_a
        )
    )

    base_action_b = (
        action_summary_from_result(
            base_result_b
        )
    )


    base_strategy_a = (
        strategy_summary_from_result(
            base_result_a
        )
    )

    base_strategy_b = (
        strategy_summary_from_result(
            base_result_b
        )
    )


    # -----------------------------------------------------
    # Two candidate trigger definitions.
    # -----------------------------------------------------

    action_trigger_a = (
        should_trigger(
            base_action_a
        )
    )

    action_trigger_b = (
        should_trigger(
            base_action_b
        )
    )


    strategy_trigger_a = (
        should_trigger(
            base_strategy_a
        )
    )

    strategy_trigger_b = (
        should_trigger(
            base_strategy_b
        )
    )


    # -----------------------------------------------------
    # One independent high-budget realization per seed
    # whenever either policy needs it.
    #
    # This keeps the comparison paired while the reported
    # computational cost below is calculated separately for
    # each policy.
    # -----------------------------------------------------

    high_result_a = None
    high_result_b = None


    if (
        action_trigger_a
        or
        strategy_trigger_a
    ):

        high_result_a = (
            run_recommendation(

                state,

                HIGH_SIMULATIONS,

                SEED_HIGH_A
                +
                i,
            )
        )


    if (
        action_trigger_b
        or
        strategy_trigger_b
    ):

        high_result_b = (
            run_recommendation(

                state,

                HIGH_SIMULATIONS,

                SEED_HIGH_B
                +
                i,
            )
        )


    if high_result_a is not None:

        high_action_a = (
            action_summary_from_result(
                high_result_a
            )
        )

        high_strategy_a = (
            strategy_summary_from_result(
                high_result_a
            )
        )

    else:

        high_action_a = None
        high_strategy_a = None


    if high_result_b is not None:

        high_action_b = (
            action_summary_from_result(
                high_result_b
            )
        )

        high_strategy_b = (
            strategy_summary_from_result(
                high_result_b
            )
        )

    else:

        high_action_b = None
        high_strategy_b = None


    # -----------------------------------------------------
    # Current production-matched ACTION trigger.
    # -----------------------------------------------------

    current_action_a = (
        high_action_a
        if action_trigger_a
        else base_action_a
    )

    current_action_b = (
        high_action_b
        if action_trigger_b
        else base_action_b
    )


    current_strategy_a = (
        high_strategy_a
        if action_trigger_a
        else base_strategy_a
    )

    current_strategy_b = (
        high_strategy_b
        if action_trigger_b
        else base_strategy_b
    )


    # -----------------------------------------------------
    # Candidate STRATEGY trigger.
    # -----------------------------------------------------

    candidate_action_a = (
        high_action_a
        if strategy_trigger_a
        else base_action_a
    )

    candidate_action_b = (
        high_action_b
        if strategy_trigger_b
        else base_action_b
    )


    candidate_strategy_a = (
        high_strategy_a
        if strategy_trigger_a
        else base_strategy_a
    )

    candidate_strategy_b = (
        high_strategy_b
        if strategy_trigger_b
        else base_strategy_b
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


        # Base.
        "base_a_action":
            base_action_a[
                "action"
            ],

        "base_b_action":
            base_action_b[
                "action"
            ],

        "base_a_strategy":
            base_strategy_a[
                "strategy"
            ],

        "base_b_strategy":
            base_strategy_b[
                "strategy"
            ],

        "base_a_action_z":
            base_action_a[
                "z_gap"
            ],

        "base_b_action_z":
            base_action_b[
                "z_gap"
            ],

        "base_a_strategy_z":
            base_strategy_a[
                "z_gap"
            ],

        "base_b_strategy_z":
            base_strategy_b[
                "z_gap"
            ],


        # Triggers.
        "action_trigger_a":
            action_trigger_a,

        "action_trigger_b":
            action_trigger_b,

        "strategy_trigger_a":
            strategy_trigger_a,

        "strategy_trigger_b":
            strategy_trigger_b,


        # Current action-trigger final.
        "current_a_action":
            current_action_a[
                "action"
            ],

        "current_b_action":
            current_action_b[
                "action"
            ],

        "current_a_strategy":
            current_strategy_a[
                "strategy"
            ],

        "current_b_strategy":
            current_strategy_b[
                "strategy"
            ],

        "current_a_action_z":
            current_action_a[
                "z_gap"
            ],

        "current_b_action_z":
            current_action_b[
                "z_gap"
            ],

        "current_a_strategy_z":
            current_strategy_a[
                "z_gap"
            ],

        "current_b_strategy_z":
            current_strategy_b[
                "z_gap"
            ],


        # Candidate strategy-trigger final.
        "candidate_a_action":
            candidate_action_a[
                "action"
            ],

        "candidate_b_action":
            candidate_action_b[
                "action"
            ],

        "candidate_a_strategy":
            candidate_strategy_a[
                "strategy"
            ],

        "candidate_b_strategy":
            candidate_strategy_b[
                "strategy"
            ],

        "candidate_a_action_z":
            candidate_action_a[
                "z_gap"
            ],

        "candidate_b_action_z":
            candidate_action_b[
                "z_gap"
            ],

        "candidate_a_strategy_z":
            candidate_strategy_a[
                "z_gap"
            ],

        "candidate_b_strategy_z":
            candidate_strategy_b[
                "z_gap"
            ],
    })


results = pd.DataFrame(
    records
)


# =========================================================
# Flip diagnostics.
# =========================================================

results[
    "base_action_flip"
] = (
    results[
        "base_a_action"
    ]
    !=
    results[
        "base_b_action"
    ]
)


results[
    "base_strategy_flip"
] = (
    results[
        "base_a_strategy"
    ]
    !=
    results[
        "base_b_strategy"
    ]
)


results[
    "current_action_flip"
] = (
    results[
        "current_a_action"
    ]
    !=
    results[
        "current_b_action"
    ]
)


results[
    "current_strategy_flip"
] = (
    results[
        "current_a_strategy"
    ]
    !=
    results[
        "current_b_strategy"
    ]
)


results[
    "candidate_action_flip"
] = (
    results[
        "candidate_a_action"
    ]
    !=
    results[
        "candidate_b_action"
    ]
)


results[
    "candidate_strategy_flip"
] = (
    results[
        "candidate_a_strategy"
    ]
    !=
    results[
        "candidate_b_strategy"
    ]
)


# =========================================================
# Strategy-confidence diagnostics.
# =========================================================

for prefix_name in [
    "current_a",
    "current_b",
    "candidate_a",
    "candidate_b",
]:

    results[
        f"{prefix_name}_strategy_resolved"
    ] = (
        results[
            f"{prefix_name}_strategy_z"
        ]
        .map(
            resolved
        )
    )


results[
    "current_both_strategy_resolved"
] = (
    results[
        "current_a_strategy_resolved"
    ]
    &
    results[
        "current_b_strategy_resolved"
    ]
)


results[
    "candidate_both_strategy_resolved"
] = (
    results[
        "candidate_a_strategy_resolved"
    ]
    &
    results[
        "candidate_b_strategy_resolved"
    ]
)


# =========================================================
# Trigger overlap.
# =========================================================

results[
    "action_only_trigger_a"
] = (
    results[
        "action_trigger_a"
    ]
    &
    ~results[
        "strategy_trigger_a"
    ]
)


results[
    "action_only_trigger_b"
] = (
    results[
        "action_trigger_b"
    ]
    &
    ~results[
        "strategy_trigger_b"
    ]
)


results[
    "strategy_only_trigger_a"
] = (
    results[
        "strategy_trigger_a"
    ]
    &
    ~results[
        "action_trigger_a"
    ]
)


results[
    "strategy_only_trigger_b"
] = (
    results[
        "strategy_trigger_b"
    ]
    &
    ~results[
        "action_trigger_b"
    ]
)


# =========================================================
# Summary.
# =========================================================

print(
    "\n300-SIMULATION BASELINE"
)

print(
    "Action flip rate:   "
    f"{100.0 * results['base_action_flip'].mean():.2f}%"
)

print(
    "Strategy flip rate: "
    f"{100.0 * results['base_strategy_flip'].mean():.2f}%"
)


def report_policy(
    name,
    trigger_column_a,
    trigger_column_b,
    action_flip_column,
    strategy_flip_column,
    resolved_column,
    strategy_z_a,
    strategy_z_b,
):

    trigger_rate_a = float(
        results[
            trigger_column_a
        ].mean()
    )

    trigger_rate_b = float(
        results[
            trigger_column_b
        ].mean()
    )

    average_trigger_rate = (
        0.5
        *
        (
            trigger_rate_a
            +
            trigger_rate_b
        )
    )

    average_simulations = (
        BASE_SIMULATIONS
        +
        average_trigger_rate
        *
        HIGH_SIMULATIONS
    )


    print(
        f"\n{name}"
    )

    print(
        "Trigger rate — seed A: "
        f"{100.0 * trigger_rate_a:.2f}%"
    )

    print(
        "Trigger rate — seed B: "
        f"{100.0 * trigger_rate_b:.2f}%"
    )

    print(
        "Adaptive action flip rate: "
        f"{100.0 * results[action_flip_column].mean():.2f}%"
    )

    print(
        "Adaptive strategy flip rate: "
        f"{100.0 * results[strategy_flip_column].mean():.2f}%"
    )

    print(
        "Estimated average simulations/action: "
        f"{average_simulations:.1f}"
    )

    print(
        "Savings versus 1,200 everywhere: "
        f"{100.0 * (1.0 - average_simulations / HIGH_SIMULATIONS):.2f}%"
    )


    both_resolved = (
        results[
            results[
                resolved_column
            ]
        ]
    )


    print(
        "Both seeds strategy-resolved: "
        f"{len(both_resolved):,}"
    )


    if len(both_resolved) > 0:

        print(
            "Strategy flip rate when both resolved: "
            f"{100.0 * both_resolved[strategy_flip_column].mean():.2f}%"
        )


    unresolved_a = (
        np.isfinite(
            results[
                strategy_z_a
            ]
        )
        &
        (
            results[
                strategy_z_a
            ]
            <
            UNCERTAINTY_Z
        )
    )


    unresolved_b = (
        np.isfinite(
            results[
                strategy_z_b
            ]
        )
        &
        (
            results[
                strategy_z_b
            ]
            <
            UNCERTAINTY_Z
        )
    )


    print(
        "Final strategy unresolved — seed A: "
        f"{100.0 * unresolved_a.mean():.2f}%"
    )

    print(
        "Final strategy unresolved — seed B: "
        f"{100.0 * unresolved_b.mean():.2f}%"
    )


report_policy(

    "PRODUCTION-MATCHED ACTION TRIGGER",

    "action_trigger_a",
    "action_trigger_b",

    "current_action_flip",
    "current_strategy_flip",

    "current_both_strategy_resolved",

    "current_a_strategy_z",
    "current_b_strategy_z",
)


report_policy(

    "CANDIDATE STRATEGY TRIGGER",

    "strategy_trigger_a",
    "strategy_trigger_b",

    "candidate_action_flip",
    "candidate_strategy_flip",

    "candidate_both_strategy_resolved",

    "candidate_a_strategy_z",
    "candidate_b_strategy_z",
)


# =========================================================
# PASS/RUN stability under candidate strategy trigger.
# =========================================================

both_candidate_go = (
    (
        results[
            "candidate_a_strategy"
        ]
        ==
        "GO FOR IT"
    )
    &
    (
        results[
            "candidate_b_strategy"
        ]
        ==
        "GO FOR IT"
    )
)


print(
    "\nPLAY-CALL STABILITY"
)

print(
    "Rows where both candidate seeds choose GO FOR IT: "
    f"{int(both_candidate_go.sum())}"
)


if both_candidate_go.any():

    playcall_flip = (
        results.loc[
            both_candidate_go,
            "candidate_a_action",
        ]
        !=
        results.loc[
            both_candidate_go,
            "candidate_b_action",
        ]
    )

    print(
        "PASS/RUN flip rate when both choose GO FOR IT: "
        f"{100.0 * playcall_flip.mean():.2f}%"
    )


# =========================================================
# Trigger comparison.
# =========================================================

action_only_rate = (
    0.5
    *
    (
        results[
            "action_only_trigger_a"
        ].mean()
        +
        results[
            "action_only_trigger_b"
        ].mean()
    )
)


strategy_only_rate = (
    0.5
    *
    (
        results[
            "strategy_only_trigger_a"
        ].mean()
        +
        results[
            "strategy_only_trigger_b"
        ].mean()
    )
)


print(
    "\nTRIGGER COMPARISON"
)

print(
    "Action-trigger-only states: "
    f"{100.0 * action_only_rate:.2f}%"
)

print(
    "Strategy-trigger-only states: "
    f"{100.0 * strategy_only_rate:.2f}%"
)


print(
    "\nIMPORTANT"
)

print(
    "This experiment uses only 2023-2024 validation data."
)

print(
    "2025 remains held out and is not used to select "
    "the adaptive simulation rule."
)


output_path = (
    "data/"
    "evaluation_validation_strategy_"
    "simulation_stability.parquet"
)


results.to_parquet(
    output_path,
    index=False,
)


print(
    f"\nSaved: {output_path}"
)
