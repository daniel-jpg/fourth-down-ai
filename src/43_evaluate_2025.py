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


    go_records.append({

        "action":
            row["action"],

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