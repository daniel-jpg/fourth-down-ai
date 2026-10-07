import importlib.util
from pathlib import Path

import nflreadpy as nfl
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    log_loss,
    roc_auc_score,
)


# =========================================================
# Configuration
# =========================================================

RANDOM_SEED = 2025

# Full 2025 is used for component evaluation.
#
# Policy simulation is more computationally expensive, so
# use a fixed random 1,000-play sample with 300 Monte Carlo
# simulations per action.
POLICY_SAMPLE_SIZE = 1000
POLICY_SIMULATIONS = 300


Path("data").mkdir(
    exist_ok=True
)


# =========================================================
# 1. Load Script 42 as a module.
#
# This reuses the exact frozen decision engine instead of
# duplicating its logic here.
# =========================================================

print("\nLoading frozen decision engine...")


spec = importlib.util.spec_from_file_location(

    "fourth_down_engine",

    "src/42_build_decision_engine.py",
)


engine = importlib.util.module_from_spec(
    spec
)


spec.loader.exec_module(
    engine
)


print(
    "\nDecision engine loaded."
)


# =========================================================
# 2. Load held-out 2025 benchmark play-by-play.
# =========================================================

print(
    "\nLoading held-out 2025 benchmark PBP..."
)


pbp = (
    nfl.load_pbp([2025])
    .sort([
        "game_id",
        "play_id",
    ])
    .to_pandas()
)


print(
    f"2025 PBP rows: "
    f"{len(pbp):,}"
)


# =========================================================
# 3. Evaluate the WP model on 2025.
#
# Reconstruct the exact generic state representation used
# in Script 41.
# =========================================================

STATE_PLAY_TYPES = [

    "run",
    "pass",

    "punt",
    "field_goal",

    "kickoff",

    "qb_kneel",
    "qb_spike",

    "no_play",
]


wp = pbp[
    pbp[
        "play_type"
    ].isin(
        STATE_PLAY_TYPES
    )
].copy()


required_wp_columns = [

    "game_seconds_remaining",

    "qtr",

    "posteam",

    "home_team",
    "away_team",

    "posteam_score",
    "defteam_score",

    "result",
]


wp = wp.dropna(
    subset=
        required_wp_columns
)


# ---------------------------------------------------------
# Target:
#
# 0 = away win
# 1 = tie
# 2 = home win
# ---------------------------------------------------------

wp[
    "outcome_class"
] = np.where(

    wp["result"] > 0,

    2,

    np.where(
        wp["result"] < 0,
        0,
        1,
    ),
)


wp[
    "is_home_posteam"
] = (

    wp["posteam"]
    ==
    wp["home_team"]

).astype(int)


wp[
    "home_score_differential"
] = np.where(

    wp[
        "is_home_posteam"
    ] == 1,

    (
        wp[
            "posteam_score"
        ]
        -
        wp[
            "defteam_score"
        ]
    ),

    (
        wp[
            "defteam_score"
        ]
        -
        wp[
            "posteam_score"
        ]
    ),
)


wp[
    "home_timeouts_remaining"
] = np.where(

    wp[
        "is_home_posteam"
    ] == 1,

    wp[
        "posteam_timeouts_remaining"
    ],

    wp[
        "defteam_timeouts_remaining"
    ],
)


wp[
    "away_timeouts_remaining"
] = np.where(

    wp[
        "is_home_posteam"
    ] == 1,

    wp[
        "defteam_timeouts_remaining"
    ],

    wp[
        "posteam_timeouts_remaining"
    ],
)


wp[
    "is_kickoff"
] = (

    wp["play_type"]
    ==
    "kickoff"

).astype(int)


wp[
    "is_overtime"
] = (

    wp["qtr"] >= 5

).astype(int)


wp[
    "final_five_minutes"
] = (

    (
        wp["qtr"] == 4
    )

    &

    (
        wp[
            "game_seconds_remaining"
        ]
        <= 300
    )

).astype(int)


# Convert WP model features to numeric.
for column in (
    engine.WP_FEATURES
):

    if column not in wp.columns:

        wp[column] = np.nan

    wp[column] = pd.to_numeric(
        wp[column],
        errors="coerce",
    )


WP_CLASS_INDEX = {

    int(value):
        index

    for index, value
    in enumerate(
        engine.wp_model.classes_
    )
}


def evaluate_wp(
    name,
    frame,
):

    if len(frame) == 0:

        print(
            f"\n{name}: "
            "no rows"
        )

        return


    X = (
        frame[
            engine.WP_FEATURES
        ]
        .to_numpy()
    )


    y = (
        frame[
            "outcome_class"
        ]
        .astype(int)
        .to_numpy()
    )


    probabilities = (
        engine.wp_model
        .predict_proba(X)
    )


    prediction = (

        engine.wp_model.classes_[

            np.argmax(
                probabilities,
                axis=1,
            )

        ]

    )


    p_home = probabilities[
        :,
        WP_CLASS_INDEX[2],
    ]


    home_target = (
        y == 2
    ).astype(float)


    multiclass_ll = log_loss(

        y,

        probabilities,

        labels=
            engine.wp_model.classes_,
    )


    accuracy = accuracy_score(
        y,
        prediction,
    )


    home_brier = float(
        np.mean(
            (
                p_home
                -
                home_target
            )
            ** 2
        )
    )


    if len(
        np.unique(
            home_target
        )
    ) > 1:

        home_auc = (
            roc_auc_score(
                home_target,
                p_home,
            )
        )

    else:

        home_auc = (
            float("nan")
        )


    print(
        f"\n{name}"
    )

    print(
        f"Rows:             "
        f"{len(frame):,}"
    )

    print(
        f"Logloss:          "
        f"{multiclass_ll:.5f}"
    )

    print(
        f"Accuracy:         "
        f"{accuracy:.5f}"
    )

    print(
        f"Home Brier:       "
        f"{home_brier:.5f}"
    )

    print(
        f"Home AUC:         "
        f"{home_auc:.5f}"
    )

    print(
        f"Actual home win:  "
        f"{home_target.mean():.5f}"
    )

    print(
        f"Pred home win:    "
        f"{p_home.mean():.5f}"
    )


evaluate_wp(
    "2025 WP — ALL STATES",
    wp,
)


evaluate_wp(

    "2025 WP — FINAL FIVE MINUTES",

    wp[
        (
            wp["qtr"] == 4
        )
        &
        (
            wp[
                "game_seconds_remaining"
            ]
            <= 300
        )
    ],

)


# =========================================================
# 4. Load the frozen fourth-down 2025 test set.
# =========================================================

test = pd.read_parquet(
    "data/fourth_down_modeling_split.parquet"
)


test = test[
    test["split"]
    ==
    "test"
].copy()


print(
    "\n2025 FOURTH-DOWN TEST SET"
)

print(
    f"Rows: "
    f"{len(test):,}"
)


print(
    "\nHistorical actions:"
)

print(
    test[
        "action"
    ]
    .value_counts()
    .to_string()
)


# =========================================================
# Add roof from raw PBP if it was not included in the
# modeling parquet.
# =========================================================

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
        )[
            "roof"
        ]

    )


    if "roof" not in test.columns:

        test[
            "roof"
        ] = (
            test[
                "game_id"
            ]
            .map(
                game_roof
            )
        )

    else:

        missing_roof = (
            test["roof"]
            .isna()
        )


        test.loc[
            missing_roof,
            "roof",
        ] = (

            test.loc[
                missing_roof,
                "game_id",
            ]
            .map(
                game_roof
            )

        )


# =========================================================
# 5. Convert one fourth-down row into Script 42 input.
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



OT_PHASE_MAP = {
    "OPENING_POSSESSION": "OPENING",
    "SECOND_POSSESSION": "RESPONSE",
    "SUDDEN_DEATH": "SUDDEN_DEATH",
}


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

ot_states_2025 = (
    ot_states[
        ot_states["season"] == 2025
    ][
        [
            "game_id",
            "play_id",
            "season_type",
            "ot_phase",
            "both_teams_guaranteed",
        ]
    ]
    .copy()
)


OT_STATE_LOOKUP = {}

for _, ot_row in ot_states_2025.iterrows():

    key = (
        str(ot_row["game_id"]),
        int(round(float(ot_row["play_id"]))),
    )

    phase = OT_PHASE_MAP.get(
        str(ot_row["ot_phase"])
    )

    if phase is None:
        raise RuntimeError(
            f"Unknown OT phase: {ot_row['ot_phase']}"
        )

    OT_STATE_LOOKUP[key] = {
        "ot_phase": phase,
        "ot_format": (
            "POSTSEASON"
            if str(ot_row["season_type"]) == "POST"
            else "REGULAR_SEASON"
        ),
        "both_teams_guaranteed": int(
            ot_row["both_teams_guaranteed"]
        ),
    }


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
            column
            not in row.index
        ):

            return None


        if not valid_number(
            row[column]
        ):

            return None


    state = {

        "qtr":
            int(
                row["qtr"]
            ),

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
            int(
                row["is_home"]
            ),

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
        "goal_to_go"
        in row.index
        and
        valid_number(
            row[
                "goal_to_go"
            ]
        )
    ):

        state[
            "goal_to_go"
        ] = int(
            row[
                "goal_to_go"
            ]
        )


    if state["qtr"] >= 5:

        if (
            "game_id" not in row.index
            or
            "play_id" not in row.index
        ):
            raise RuntimeError(
                "OT evaluation row is missing "
                "game_id or play_id."
            )

        ot_key = (
            str(row["game_id"]),
            int(
                round(
                    float(row["play_id"])
                )
            ),
        )

        ot_metadata = (
            OT_STATE_LOOKUP.get(
                ot_key
            )
        )

        if ot_metadata is None:
            raise RuntimeError(
                "Missing OT metadata for "
                f"{ot_key[0]} play {ot_key[1]}"
            )

        state["ot_phase"] = (
            ot_metadata["ot_phase"]
        )

        state["ot_format"] = (
            ot_metadata["ot_format"]
        )

        state["ot_period"] = max(
            1,
            int(state["qtr"]) - 4,
        )

    return state


# =========================================================
# 6. 2025 GO conversion calibration.
# =========================================================

go_test = test[

    test[
        "action"
    ].isin(
        engine.GO_ACTIONS.keys()
    )

    &

    test[
        "converted"
    ].notna()

].copy()


go_records = []


for _, row in (
    go_test.iterrows()
):

    state = (
        row_to_state(row)
    )


    if state is None:

        continue


    base = (
        engine.normalize_state(
            state
        )
    )


    p_conversion = (

        engine
        .go_conversion_probability(

            base,

            row["action"],
        )

    )


    raw_p_conversion = (
        engine
        .go_conversion_probability(
            base,
            row["action"],
            apply_calibration=False,
        )
    )


    go_records.append({

        "game_id":
            str(
                row["game_id"]
            ),

        "action":
            row["action"],

        "ydstogo":
            float(
                base["ydstogo"]
            ),

        "actual":
            int(
                bool(
                    row[
                        "converted"
                    ]
                )
            ),

        "predicted":
            p_conversion,

        "raw_predicted":
            raw_p_conversion,
    })


go_eval = pd.DataFrame(
    go_records
)


print(
    "\n2025 GO CONVERSION MODEL"
)


if len(go_eval) > 0:

    y = (
        go_eval[
            "actual"
        ]
        .to_numpy()
    )


    p = (
        go_eval[
            "predicted"
        ]
        .to_numpy()
    )


    binary_probs = np.column_stack([
        1.0 - p,
        p,
    ])


    print(
        f"Rows:       "
        f"{len(go_eval):,}"
    )

    print(
        f"Logloss:    "
        f"{log_loss(y, binary_probs, labels=[0, 1]):.5f}"
    )

    print(
        f"Brier:      "
        f"{np.mean((p - y) ** 2):.5f}"
    )


    if len(
        np.unique(y)
    ) > 1:

        print(
            f"AUC:        "
            f"{roc_auc_score(y, p):.5f}"
        )


    print(
        f"Actual conv:"
        f" {y.mean():.5f}"
    )

    print(
        f"Pred conv:  "
        f" {p.mean():.5f}"
    )


    print(
        "\nBY ACTION"
    )


    by_action = (

        go_eval

        .groupby(
            "action"
        )

        .agg(

            plays=(
                "actual",
                "size",
            ),

            actual_conversion=(
                "actual",
                "mean",
            ),

            predicted_conversion=(
                "predicted",
                "mean",
            ),
        )

        .sort_index()

    )


    print(
        by_action.to_string()
    )


# =========================================================
# 7. 2025 FIELD-GOAL make calibration.
#
# The production engine uses the frozen median-kicker
# fallback unless explicit kicker information is supplied.
# =========================================================

fg_test = test[
    test["action"]
    ==
    "FIELD_GOAL"
].copy()


def actual_fg_made(row):

    if (
        "field_goal_result"
        in row.index
    ):

        value = str(
            row[
                "field_goal_result"
            ]
        ).lower()


        if "made" in value:

            return 1


        if (
            "miss" in value
            or
            "block" in value
        ):

            return 0


    for column in [

        "fg_made",

        "field_goal_made",

    ]:

        if (
            column
            in row.index
            and
            pd.notna(
                row[column]
            )
        ):

            return int(
                bool(
                    row[column]
                )
            )


    if (
        "execution_status"
        in row.index
        and
        str(
            row[
                "execution_status"
            ]
        ).upper()
        ==
        "BROKEN"
    ):

        return 0


    return None


fg_records = []


for _, row in (
    fg_test.iterrows()
):

    actual = (
        actual_fg_made(row)
    )


    if actual is None:

        continue


    state = (
        row_to_state(row)
    )


    if state is None:

        continue


    base = (
        engine.normalize_state(
            state
        )
    )


    probabilities = (

        engine
        .field_goal_probabilities(
            base
        )

    )


    fg_records.append({

        "game_id":
            str(
                row["game_id"]
            ),

        "actual":
            int(actual),

        "predicted":
            probabilities[
                "made"
            ],

        "distance":
            base[
                "yardline_100"
            ]
            +
            18.0,
    })


fg_eval = pd.DataFrame(
    fg_records
)


print(
    "\n2025 FIELD-GOAL MODEL"
)


if len(fg_eval) > 0:

    y = (
        fg_eval[
            "actual"
        ]
        .to_numpy()
    )


    p = (
        fg_eval[
            "predicted"
        ]
        .to_numpy()
    )


    binary_probs = np.column_stack([
        1.0 - p,
        p,
    ])


    print(
        f"Rows:       "
        f"{len(fg_eval):,}"
    )

    print(
        f"Logloss:    "
        f"{log_loss(y, binary_probs, labels=[0, 1]):.5f}"
    )

    print(
        f"Brier:      "
        f"{np.mean((p - y) ** 2):.5f}"
    )


    if len(
        np.unique(y)
    ) > 1:

        print(
            f"AUC:        "
            f"{roc_auc_score(y, p):.5f}"
        )


    print(
        f"Actual make:"
        f" {y.mean():.5f}"
    )

    print(
        f"Pred make:  "
        f" {p.mean():.5f}"
    )


# =========================================================
# 8. Policy evaluation.
#
# IMPORTANT:
#
# We cannot observe the counterfactual outcome of an action
# that was not actually chosen.
#
# Therefore this section evaluates:
#
#   - recommendation distribution
#   - agreement with historical decisions
#   - model-estimated Q gap
#
# It does NOT claim causal improvement in actual wins.
# =========================================================

policy_source = []


for index, row in (
    test.iterrows()
):

    state = (
        row_to_state(row)
    )


    if state is not None:

        policy_source.append(
            index
        )


policy_df = test.loc[
    policy_source
].copy()


sample_size = min(
    POLICY_SAMPLE_SIZE,
    len(policy_df),
)


policy_sample = (

    policy_df.sample(
        n=sample_size,
        random_state=
            RANDOM_SEED,
    )

    .reset_index(
        drop=True
    )

)


print(
    "\n2025 POLICY EVALUATION"
)

print(
    f"Eligible source rows: "
    f"{len(policy_df):,}"
)

print(
    f"Policy sample rows:   "
    f"{len(policy_sample):,}"
)

print(
    f"MC simulations/action:"
    f" {POLICY_SIMULATIONS}"
)


policy_records = []


for i, row in (
    policy_sample.iterrows()
):

    if (
        i % 100
        ==
        0
    ):

        print(
            f"Evaluating "
            f"{i:,}/"
            f"{len(policy_sample):,}..."
        )


    state = (
        row_to_state(row)
    )


    result = (
        engine.recommend(

            state,

            n_simulations=
                POLICY_SIMULATIONS,

            seed=
                RANDOM_SEED
                +
                i,
        )
    )


    action_table = (
        result[
            "results"
        ]
    )


    eligible_rows = (
        action_table[
            action_table[
                "eligible"
            ]
        ]
    )


    recommended_action = (

        eligible_rows
        .iloc[0][
            "action"
        ]

    )


    recommended_q = float(

        eligible_rows
        .iloc[0][
            "expected_win_probability"
        ]

    )


    historical_action = (
        row[
            "action"
        ]
    )


    historical_row = (

        action_table[
            action_table[
                "action"
            ]
            ==
            historical_action
        ]

    )


    historical_q = np.nan


    if (
        len(
            historical_row
        )
        ==
        1
        and
        bool(
            historical_row
            .iloc[0][
                "eligible"
            ]
        )
    ):

        historical_q = float(

            historical_row
            .iloc[0][
                "expected_win_probability"
            ]

        )


    if np.isfinite(
        historical_q
    ):

        estimated_gain_pct = (

            100.0
            *
            (
                recommended_q
                -
                historical_q
            )

        )

    else:

        estimated_gain_pct = np.nan


    policy_records.append({

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

        "historical_action":
            historical_action,

        "recommended_action":
            recommended_action,

        "agreement":
            int(
                historical_action
                ==
                recommended_action
            ),

        "pre_decision_wp":
            result[
                "pre_decision_wp"
            ],

        "recommended_q":
            recommended_q,

        "historical_q":
            historical_q,

        "estimated_gain_pct":
            estimated_gain_pct,

        "edge_over_second_best_pct":
            result[
                "edge_over_second_best_pct"
            ],

        "yardline_100":
            row[
                "yardline_100"
            ],

        "ydstogo":
            row[
                "ydstogo"
            ],

        "qtr":
            row[
                "qtr"
            ],

        "game_seconds_remaining":
            row[
                "game_seconds_remaining"
            ],
    })


policy_results = pd.DataFrame(
    policy_records
)


policy_results.to_parquet(

    "data/evaluation_2025_policy_sample.parquet",

    index=False,
)


# =========================================================
# 9. Policy summary.
# =========================================================

print(
    "\nRECOMMENDATION DISTRIBUTION"
)


recommendation_summary = (

    policy_results[
        "recommended_action"
    ]

    .value_counts()

    .rename_axis(
        "action"
    )

    .reset_index(
        name="plays"
    )

)


recommendation_summary[
    "pct"
] = (

    100.0

    *
    recommendation_summary[
        "plays"
    ]

    /
    len(
        policy_results
    )

)


print(
    recommendation_summary
    .to_string(
        index=False
    )
)


print(
    "\nHISTORICAL ACTION DISTRIBUTION — SAME SAMPLE"
)


historical_summary = (

    policy_results[
        "historical_action"
    ]

    .value_counts()

    .rename_axis(
        "action"
    )

    .reset_index(
        name="plays"
    )

)


historical_summary[
    "pct"
] = (

    100.0

    *
    historical_summary[
        "plays"
    ]

    /
    len(
        policy_results
    )

)


print(
    historical_summary
    .to_string(
        index=False
    )
)


agreement_rate = (

    policy_results[
        "agreement"
    ]
    .mean()

)


print(
    "\nPOLICY AGREEMENT"
)

print(
    f"Exact agreement with "
    f"historical action: "
    f"{100 * agreement_rate:.2f}%"
)


comparable = (

    policy_results[
        "estimated_gain_pct"
    ]
    .dropna()

)


print(
    "\nMODEL-ESTIMATED EDGE OVER HISTORICAL ACTION"
)

print(
    f"Comparable plays: "
    f"{len(comparable):,}/"
    f"{len(policy_results):,}"
)


if len(
    comparable
) > 0:

    print(
        f"Mean:   "
        f"{comparable.mean():.3f} pp"
    )

    print(
        f"Median: "
        f"{comparable.median():.3f} pp"
    )

    print(
        f"P90:    "
        f"{comparable.quantile(0.90):.3f} pp"
    )

    print(
        f">1 pp:  "
        f"{100 * (comparable > 1).mean():.2f}%"
    )


close_calls = (

    policy_results[
        "edge_over_second_best_pct"
    ]
    <= 1.0

)


print(
    "\nDECISION CERTAINTY"
)

print(
    f"Top-two actions within "
    f"1 percentage point: "
    f"{100 * close_calls.mean():.2f}%"
)


fake_recommendations = (

    policy_results[
        "recommended_action"
    ]
    .str.startswith(
        "FAKE_"
    )
)


print(
    f"Fake-play recommendations: "
    f"{int(fake_recommendations.sum())}"
)


print(
    "\nSAVED"
)

print(
    "data/evaluation_2025_policy_sample.parquet"
)


print(
    "\nIMPORTANT:"
)

print(
    "The policy edge is model-estimated, "
    "not an observed causal win improvement."
)

# =========================================================
# 9. Calibration diagnostics.
#
# These tables are descriptive holdout diagnostics only.
# Do not tune production models to the 2025 benchmark.
# =========================================================

def print_calibration_table(
    name,
    actual,
    predicted,
    n_bins=10,
):
    calibration = pd.DataFrame({
        "actual": np.asarray(
            actual,
            dtype=float,
        ),
        "predicted": np.asarray(
            predicted,
            dtype=float,
        ),
    })

    calibration = calibration[
        np.isfinite(calibration["actual"])
        &
        np.isfinite(calibration["predicted"])
    ].copy()

    if len(calibration) == 0:
        print(
            f"\n{name} CALIBRATION: no rows"
        )
        return

    edges = np.linspace(
        0.0,
        1.0,
        n_bins + 1,
    )

    calibration["probability_bin"] = pd.cut(
        calibration["predicted"],
        bins=edges,
        include_lowest=True,
    )

    table = (
        calibration
        .groupby(
            "probability_bin",
            observed=True,
        )
        .agg(
            plays=(
                "actual",
                "size",
            ),
            mean_predicted=(
                "predicted",
                "mean",
            ),
            actual_rate=(
                "actual",
                "mean",
            ),
        )
        .reset_index()
    )

    table["calibration_gap"] = (
        table["actual_rate"]
        -
        table["mean_predicted"]
    )

    print(
        f"\n{name} CALIBRATION"
    )

    print(
        table.to_string(
            index=False,
            formatters={
                "mean_predicted":
                    lambda x: f"{x:.3f}",
                "actual_rate":
                    lambda x: f"{x:.3f}",
                "calibration_gap":
                    lambda x: f"{x:+.3f}",
            },
        )
    )


# Win-probability calibration.
wp_X = (
    wp[
        engine.WP_FEATURES
    ]
    .to_numpy()
)

wp_y = (
    wp["outcome_class"]
    .astype(int)
    .to_numpy()
)

wp_probabilities = (
    engine.wp_model.predict_proba(
        wp_X
    )
)

wp_home_probability = (
    wp_probabilities[
        :,
        WP_CLASS_INDEX[2],
    ]
)

wp_home_actual = (
    wp_y == 2
).astype(float)

print_calibration_table(
    "2025 WP — HOME WIN",
    wp_home_actual,
    wp_home_probability,
)


# GO conversion calibration.
if len(go_eval) > 0:
    print_calibration_table(
        "2025 GO CONVERSION",
        go_eval["actual"],
        go_eval["predicted"],
    )


# Field-goal calibration.
if len(fg_eval) > 0:
    print_calibration_table(
        "2025 FIELD GOAL",
        fg_eval["actual"],
        fg_eval["predicted"],
    )


print(
    "\nCalibration gaps are actual rate minus "
    "mean predicted probability."
)

print(
    "Positive = model underprediction; "
    "negative = model overprediction."
)



# =========================================================
# GAME-CLUSTERED BOOTSTRAP UNCERTAINTY
#
# Resample complete NFL games rather than individual plays
# because fourth-down observations within one game are not
# statistically independent.
# =========================================================

BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 2025


def binary_metric_values(
    frame,
    predicted_column,
):

    y = (
        frame["actual"]
        .to_numpy(
            dtype=float
        )
    )

    p = np.clip(
        frame[predicted_column]
        .to_numpy(
            dtype=float
        ),
        1e-9,
        1.0 - 1e-9,
    )

    metrics = {
        "Logloss":
            float(
                log_loss(
                    y,
                    p,
                    labels=[
                        0,
                        1,
                    ],
                )
            ),

        "Brier":
            float(
                np.mean(
                    (p - y) ** 2
                )
            ),

        "Actual rate":
            float(
                np.mean(y)
            ),

        "Predicted rate":
            float(
                np.mean(p)
            ),

        "AUC":
            np.nan,
    }

    if (
        np.unique(y).size
        >
        1
    ):
        metrics["AUC"] = float(
            roc_auc_score(
                y,
                p,
            )
        )

    return metrics


def clustered_bootstrap_samples(
    frame,
    *,
    seed,
):

    frame = (
        frame
        .reset_index(
            drop=True
        )
    )

    groups = {
        game_id:
            group.index.to_numpy(
                dtype=int
            )

        for (
            game_id,
            group,
        )
        in frame.groupby(
            "game_id",
            sort=False,
        )
    }

    game_ids = np.array(
        list(
            groups.keys()
        ),
        dtype=object,
    )

    rng = (
        np.random
        .default_rng(seed)
    )

    for _ in range(
        BOOTSTRAP_REPLICATES
    ):

        sampled_games = rng.choice(
            game_ids,
            size=len(game_ids),
            replace=True,
        )

        sampled_rows = np.concatenate([
            groups[game_id]
            for game_id
            in sampled_games
        ])

        yield frame.iloc[
            sampled_rows
        ]


def print_clustered_ci(
    name,
    frame,
    predicted_column,
    *,
    seed,
):

    point = binary_metric_values(
        frame,
        predicted_column,
    )

    distributions = {
        metric: []
        for metric
        in point
    }

    for sample in clustered_bootstrap_samples(
        frame,
        seed=seed,
    ):

        values = binary_metric_values(
            sample,
            predicted_column,
        )

        for (
            metric,
            value,
        ) in values.items():

            distributions[
                metric
            ].append(value)

    print(
        f"\n{name}"
    )

    print(
        "Game-clustered bootstrap "
        f"replicates: {BOOTSTRAP_REPLICATES:,}"
    )

    for metric in [
        "Logloss",
        "Brier",
        "AUC",
        "Actual rate",
        "Predicted rate",
    ]:

        values = np.asarray(
            distributions[metric],
            dtype=float,
        )

        lower, upper = (
            np.nanpercentile(
                values,
                [
                    2.5,
                    97.5,
                ],
            )
        )

        print(
            f"{metric:<15} "
            f"{point[metric]:.5f} "
            f"[{lower:.5f}, {upper:.5f}]"
        )


# =========================================================
# Current production-model uncertainty.
# =========================================================

if len(go_eval) > 0:

    print_clustered_ci(
        "2025 GO — GAME-CLUSTERED 95% CI",
        go_eval,
        "predicted",
        seed=BOOTSTRAP_SEED,
    )


if len(fg_eval) > 0:

    print_clustered_ci(
        "2025 FIELD GOAL — GAME-CLUSTERED 95% CI",
        fg_eval,
        "predicted",
        seed=(
            BOOTSTRAP_SEED
            +
            1
        ),
    )


# =========================================================
# Paired uncertainty for the short-yardage run calibration.
#
# Raw pre-calibration probabilities are recorded directly
# from the engine with apply_calibration=False.
# =========================================================

if len(go_eval) > 0:

    go_comparison = (
        go_eval.copy()
    )

    old_point = (
        binary_metric_values(
            go_comparison,
            "raw_predicted",
        )
    )

    new_point = (
        binary_metric_values(
            go_comparison,
            "predicted",
        )
    )


    delta_distributions = {
        "Logloss": [],
        "Brier": [],
        "AUC": [],
    }


    for sample in clustered_bootstrap_samples(
        go_comparison,
        seed=(
            BOOTSTRAP_SEED
            +
            2
        ),
    ):

        old_values = (
            binary_metric_values(
                sample,
                "raw_predicted",
            )
        )

        new_values = (
            binary_metric_values(
                sample,
                "predicted",
            )
        )

        for metric in (
            delta_distributions
        ):

            delta_distributions[
                metric
            ].append(
                new_values[metric]
                -
                old_values[metric]
            )


    print(
        "\n2025 GO CALIBRATION CHANGE — "
        "PAIRED GAME-CLUSTERED 95% CI"
    )

    print(
        "Delta = calibrated production "
        "minus raw pre-calibration model."
    )

    print(
        "For Logloss/Brier, negative is better. "
        "For AUC, positive is better."
    )


    for metric in [
        "Logloss",
        "Brier",
        "AUC",
    ]:

        point_delta = (
            new_point[metric]
            -
            old_point[metric]
        )

        values = np.asarray(
            delta_distributions[
                metric
            ],
            dtype=float,
        )

        lower, upper = (
            np.nanpercentile(
                values,
                [
                    2.5,
                    97.5,
                ],
            )
        )

        print(
            f"{metric:<9} "
            f"{point_delta:+.5f} "
            f"[{lower:+.5f}, {upper:+.5f}]"
        )
