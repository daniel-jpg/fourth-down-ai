import polars as pl
import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


# ---------------------------------------------------------
# Load development data only.
# Keep 2025 completely untouched.
# ---------------------------------------------------------

df = pl.read_parquet(
    "data/fourth_downs_labeled.parquet"
)


fg = (
    df
    .filter(
        (pl.col("action") == "FIELD_GOAL")
        &
        (pl.col("season") <= 2024)
    )
    .with_columns([

        (
            pl.col("yardline_100") + 18
        ).alias("fg_distance_estimate"),

        (
            pl.col("field_goal_result")
            .fill_null("")
            .str.to_lowercase()
            == "made"
        )
        .cast(pl.Int8)
        .alias("fg_made"),

    ])
)


pdf = fg.to_pandas()


pdf["game_date"] = pd.to_datetime(
    pdf["game_date"]
)


pdf = (
    pdf
    .sort_values([
        "game_date",
        "game_id",
        "play_id",
    ])
    .reset_index(drop=True)
)


pdf["split"] = np.where(
    pdf["season"] <= 2022,
    "train",
    "validation",
)


# ---------------------------------------------------------
# Actual four-way outcome
# ---------------------------------------------------------

pdf["fg_outcome"] = np.where(
    pdf["execution_status"] == "BROKEN",
    "broken",
    pdf["field_goal_result"]
        .fillna("unknown")
        .str.lower(),
)


# ---------------------------------------------------------
# Pre-play features
# ---------------------------------------------------------

pdf["fg_distance_sq"] = (
    pdf["fg_distance_estimate"] ** 2
)


pdf["roof_dome"] = (
    pdf["roof"] == "dome"
).astype(int)

pdf["roof_closed"] = (
    pdf["roof"] == "closed"
).astype(int)

pdf["roof_open"] = (
    pdf["roof"] == "open"
).astype(int)


# ---------------------------------------------------------
# Leakage-safe distance-adjusted kicker skill
#
# Same logic as Script 20.
# ---------------------------------------------------------

difficulty_features = [
    "fg_distance_estimate",
    "fg_distance_sq",
    "roof_dome",
    "roof_closed",
    "roof_open",
]


pdf["adjusted_kicker_skill"] = 0.0
pdf["adjusted_prior_attempts"] = 0.0


shrink_attempts = 20.0


for season in sorted(
    pdf["season"].unique()
):

    current_mask = (
        pdf["season"] == season
    )

    previous_mask = (
        (pdf["season"] < season)
        &
        (pdf["execution_status"] == "NORMAL")
        &
        pdf["kicker_player_id"].notna()
    )


    if previous_mask.sum() == 0:
        continue


    previous = pdf.loc[
        previous_mask
    ].copy()


    difficulty_model = Pipeline([
        (
            "scaler",
            StandardScaler(),
        ),
        (
            "model",
            LogisticRegression(
                max_iter=3000,
            ),
        ),
    ])


    difficulty_model.fit(
        previous[difficulty_features],
        previous["fg_made"],
    )


    previous_expected = (
        difficulty_model
        .predict_proba(
            previous[difficulty_features]
        )[:, 1]
    )


    previous["residual"] = (
        previous["fg_made"]
        - previous_expected
    )


    history = (
        previous
        .groupby("kicker_player_id")
        .agg(
            residual_sum=(
                "residual",
                "sum",
            ),
            attempts=(
                "fg_made",
                "size",
            ),
        )
    )


    residual_sum = (
        history["residual_sum"]
        .to_dict()
    )

    attempts = (
        history["attempts"]
        .to_dict()
    )


    current_indices = (
        pdf.index[
            current_mask
        ]
        .tolist()
    )


    for idx in current_indices:

        kicker = pdf.at[
            idx,
            "kicker_player_id"
        ]


        if pd.isna(kicker):

            pdf.at[
                idx,
                "adjusted_kicker_skill"
            ] = 0.0

            pdf.at[
                idx,
                "adjusted_prior_attempts"
            ] = 0.0

            continue


        n_prior = attempts.get(
            kicker,
            0
        )

        residual_prior = residual_sum.get(
            kicker,
            0.0
        )


        pdf.at[
            idx,
            "adjusted_prior_attempts"
        ] = n_prior


        pdf.at[
            idx,
            "adjusted_kicker_skill"
        ] = (
            residual_prior
            /
            (
                n_prior
                + shrink_attempts
            )
        )


        # Update only AFTER assigning the current feature.
        if (
            pdf.at[
                idx,
                "execution_status"
            ]
            == "NORMAL"
        ):

            expected = (
                difficulty_model
                .predict_proba(
                    pdf.loc[
                        [idx],
                        difficulty_features,
                    ]
                )[0, 1]
            )


            residual = (
                pdf.at[
                    idx,
                    "fg_made"
                ]
                - expected
            )


            residual_sum[kicker] = (
                residual_sum.get(
                    kicker,
                    0.0
                )
                + residual
            )


            attempts[kicker] = (
                attempts.get(
                    kicker,
                    0
                )
                + 1
            )


pdf["log_adjusted_prior_attempts"] = (
    np.log1p(
        pdf["adjusted_prior_attempts"]
    )
)


# ---------------------------------------------------------
# Splits
# ---------------------------------------------------------

train = pdf[
    pdf["split"] == "train"
].copy()

validation = pdf[
    pdf["split"] == "validation"
].copy()


normal_train = train[
    train["execution_status"] == "NORMAL"
].copy()

normal_validation = validation[
    validation["execution_status"] == "NORMAL"
].copy()


print("\nSIZES")

print(
    f"All train FG decisions:      {len(train):,}"
)

print(
    f"Normal train FG executions:  {len(normal_train):,}"
)

print(
    f"All validation FG decisions: {len(validation):,}"
)

print(
    f"Normal validation executions:{len(normal_validation):,}"
)


# ---------------------------------------------------------
# 1. BROKEN EXECUTION
#
# Too sparse for a predictive model.
# Use training empirical rate.
# ---------------------------------------------------------

p_broken = (
    (
        train["execution_status"]
        == "BROKEN"
    )
    .mean()
)


print("\nBROKEN EXECUTION MODEL")

print(
    f"Training constant: "
    f"{p_broken:.6f}"
)


# ---------------------------------------------------------
# 2. MADE GIVEN NOT BROKEN
#
# Best simple model from our development work:
# distance + roof + adjusted kicker skill.
# ---------------------------------------------------------

make_features = [
    "fg_distance_estimate",
    "fg_distance_sq",

    "roof_dome",
    "roof_closed",
    "roof_open",

    "adjusted_kicker_skill",
    "log_adjusted_prior_attempts",
]


make_model = Pipeline([
    (
        "scaler",
        StandardScaler(),
    ),
    (
        "model",
        LogisticRegression(
            max_iter=3000,
        ),
    ),
])


make_model.fit(
    normal_train[make_features],
    normal_train["fg_made"],
)


# Conditional clean-execution make probability.
p_make_clean = (
    make_model
    .predict_proba(
        validation[make_features]
    )[:, 1]
)


# ---------------------------------------------------------
# 3. BLOCKED GIVEN CLEAN FAILURE
#
# Distance-only logistic model from Script 22.
# ---------------------------------------------------------

failure_train = normal_train[
    normal_train[
        "fg_outcome"
    ].isin([
        "missed",
        "blocked",
    ])
].copy()


failure_train["blocked"] = (
    failure_train["fg_outcome"]
    == "blocked"
).astype(int)


block_features = [
    "fg_distance_estimate",
]


block_model = Pipeline([
    (
        "scaler",
        StandardScaler(),
    ),
    (
        "model",
        LogisticRegression(
            max_iter=2000,
        ),
    ),
])


block_model.fit(
    failure_train[block_features],
    failure_train["blocked"],
)

# ---------------------------------------------------------
# Save frozen field-goal outcome model bundle.
#
# Includes sensible training-data defaults for kicker
# context when the decision engine does not know the
# current kicker.
# ---------------------------------------------------------

fg_outcome_bundle = {

    "make_model":
        make_model,

    "block_model":
        block_model,

    "make_features":
        make_features,

    "block_features":
        block_features,

    "p_broken":
        float(p_broken),

    "default_adjusted_kicker_skill":
        float(
            normal_train[
                "adjusted_kicker_skill"
            ].median()
        ),

    "default_log_adjusted_prior_attempts":
        float(
            normal_train[
                "log_adjusted_prior_attempts"
            ].median()
        ),

}


joblib.dump(
    fg_outcome_bundle,
    "models/field_goal_outcome_model.joblib",
)


print(
    "\nSaved FG outcome bundle:"
)

print(
    "models/field_goal_outcome_model.joblib"
)

p_block_given_failure = (
    block_model
    .predict_proba(
        validation[block_features]
    )[:, 1]
)


# ---------------------------------------------------------
# Combine hierarchical probabilities
# ---------------------------------------------------------

p_clean = (
    1.0
    - p_broken
)


p_made = (
    p_clean
    *
    p_make_clean
)


p_failed_clean = (
    p_clean
    *
    (
        1.0
        - p_make_clean
    )
)


p_blocked = (
    p_failed_clean
    *
    p_block_given_failure
)


p_missed = (
    p_failed_clean
    *
    (
        1.0
        - p_block_given_failure
    )
)


p_broken_vec = np.full(
    len(validation),
    p_broken,
)


probs = np.column_stack([
    p_made,
    p_missed,
    p_blocked,
    p_broken_vec,
])


classes = [
    "made",
    "missed",
    "blocked",
    "broken",
]


# ---------------------------------------------------------
# Probability sanity check
# ---------------------------------------------------------

prob_sums = probs.sum(
    axis=1
)


print("\nPROBABILITY SUM CHECK")

print(
    f"Min sum: "
    f"{prob_sums.min():.12f}"
)

print(
    f"Max sum: "
    f"{prob_sums.max():.12f}"
)

print(
    f"Max |sum - 1|: "
    f"{np.max(np.abs(prob_sums - 1)):.12e}"
)


# ---------------------------------------------------------
# Multiclass log loss
# ---------------------------------------------------------

class_to_index = {
    name: i
    for i, name in enumerate(classes)
}


actual_index = np.array([
    class_to_index[
        outcome
    ]
    for outcome in validation[
        "fg_outcome"
    ]
])


actual_probability = probs[
    np.arange(
        len(validation)
    ),
    actual_index,
]


multiclass_logloss = (
    -np.mean(
        np.log(
            np.clip(
                actual_probability,
                1e-15,
                1.0,
            )
        )
    )
)


# ---------------------------------------------------------
# Multiclass Brier score
#
# Sum squared error across all four classes,
# averaged over plays.
# ---------------------------------------------------------

one_hot = np.zeros_like(
    probs
)


one_hot[
    np.arange(
        len(validation)
    ),
    actual_index,
] = 1.0


multiclass_brier = (
    np.mean(
        np.sum(
            (
                probs
                - one_hot
            ) ** 2,
            axis=1,
        )
    )
)


print("\nFINAL FG OUTCOME MODEL")

print(
    f"Multiclass log loss: "
    f"{multiclass_logloss:.6f}"
)

print(
    f"Multiclass Brier: "
    f"{multiclass_brier:.6f}"
)


# ---------------------------------------------------------
# Overall calibration
# ---------------------------------------------------------

print("\nCLASS CALIBRATION")

for i, name in enumerate(
    classes
):

    actual_rate = (
        validation[
            "fg_outcome"
        ]
        .eq(name)
        .mean()
    )

    predicted_rate = (
        probs[:, i]
        .mean()
    )

    print(
        f"{name:8s} "
        f"actual={actual_rate:.6f} "
        f"predicted={predicted_rate:.6f}"
    )


# ---------------------------------------------------------
# Conditional make-model check
# ---------------------------------------------------------

normal_pred = (
    make_model
    .predict_proba(
        normal_validation[
            make_features
        ]
    )[:, 1]
)


print(
    "\nCLEAN FG MAKE CALIBRATION"
)

print(
    f"Actual: "
    f"{normal_validation['fg_made'].mean():.6f}"
)

print(
    f"Predicted: "
    f"{normal_pred.mean():.6f}"
)


# ---------------------------------------------------------
# Example probabilities by kick distance
#
# These are validation averages, so kicker/roof mix also
# varies by bucket.
# ---------------------------------------------------------

result = validation[
    [
        "fg_distance_estimate",
        "fg_outcome",
    ]
].copy()


result["p_made"] = p_made
result["p_missed"] = p_missed
result["p_blocked"] = p_blocked
result["p_broken"] = p_broken_vec


bins = [
    0,
    29,
    34,
    39,
    44,
    49,
    54,
    59,
    100,
]

labels = [
    "<=29",
    "30-34",
    "35-39",
    "40-44",
    "45-49",
    "50-54",
    "55-59",
    "60+",
]


result["distance_bucket"] = pd.cut(
    result["fg_distance_estimate"],
    bins=bins,
    labels=labels,
)


summary = (
    result
    .groupby(
        "distance_bucket",
        observed=True,
    )
    .agg(
        attempts=(
            "fg_outcome",
            "size",
        ),

        p_made=(
            "p_made",
            "mean",
        ),

        p_missed=(
            "p_missed",
            "mean",
        ),

        p_blocked=(
            "p_blocked",
            "mean",
        ),

        p_broken=(
            "p_broken",
            "mean",
        ),
    )
)


print(
    "\nPREDICTED OUTCOME DISTRIBUTION BY DISTANCE"
)

print(
    summary.to_string()
)