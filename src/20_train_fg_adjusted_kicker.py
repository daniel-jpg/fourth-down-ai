import polars as pl
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    log_loss,
    brier_score_loss,
    roc_auc_score,
)


# ---------------------------------------------------------
# Load data
#
# Development only through 2024.
# Do NOT inspect 2025.
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
# Basic features
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

pdf["is_outdoors"] = (
    pdf["roof"] == "outdoors"
).astype(int)


# ---------------------------------------------------------
# Weather features
#
# Use TRAIN medians only.
# ---------------------------------------------------------

train_mask = (
    pdf["split"] == "train"
)

train_outdoor = (
    train_mask
    &
    (pdf["roof"] == "outdoors")
)


temp_median = (
    pdf.loc[
        train_outdoor,
        "temp"
    ]
    .median()
)

wind_median = (
    pdf.loc[
        train_outdoor,
        "wind"
    ]
    .median()
)


temp_fill = (
    pdf["temp"]
    .fillna(temp_median)
)

wind_fill = (
    pdf["wind"]
    .fillna(wind_median)
)


pdf["outdoor_temp"] = (
    pdf["is_outdoors"]
    *
    (
        temp_fill
        - temp_median
    )
)


pdf["outdoor_wind"] = (
    pdf["is_outdoors"]
    *
    wind_fill
)


pdf["outdoor_weather_missing"] = (
    (
        pdf["is_outdoors"] == 1
    )
    &
    (
        pdf["temp"].isna()
        |
        pdf["wind"].isna()
    )
).astype(int)


# ---------------------------------------------------------
# Difficulty model features
#
# NO kicker identity.
# ---------------------------------------------------------

difficulty_features = [
    "fg_distance_estimate",
    "fg_distance_sq",
    "roof_dome",
    "roof_closed",
    "roof_open",
    "outdoor_temp",
    "outdoor_wind",
    "outdoor_weather_missing",
]


# ---------------------------------------------------------
# Leakage-safe kicker skill
#
# For each season:
#
# 1. Fit a kick-difficulty model using ONLY prior seasons.
# 2. Estimate how difficult each previous kick was.
# 3. Compute kicker residual:
#
#       actual - expected
#
# 4. Process current-season kicks chronologically.
#
# Current kick never contributes to its own feature.
# ---------------------------------------------------------

pdf["adjusted_kicker_skill"] = 0.0
pdf["adjusted_prior_attempts"] = 0.0


seasons = sorted(
    pdf["season"].unique()
)


shrink_attempts = 20.0


for season in seasons:

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


    # -----------------------------------------------------
    # No prior season exists for 2014.
    # Kicker skill stays at neutral 0.
    # -----------------------------------------------------

    if previous_mask.sum() == 0:
        continue


    previous = pdf.loc[
        previous_mask
    ].copy()


    # -----------------------------------------------------
    # Fit expected-make model using only data that was
    # already available before the current season.
    # -----------------------------------------------------

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
        previous[
            difficulty_features
        ],
        previous[
            "fg_made"
        ],
    )


    # -----------------------------------------------------
    # Recalculate historical kicker residuals using the
    # current pre-season difficulty model.
    # -----------------------------------------------------

    previous_expected = (
        difficulty_model
        .predict_proba(
            previous[
                difficulty_features
            ]
        )[:, 1]
    )


    previous["residual"] = (
        previous["fg_made"]
        - previous_expected
    )


    history = (
        previous
        .groupby(
            "kicker_player_id"
        )
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
        history[
            "residual_sum"
        ]
        .to_dict()
    )

    attempts = (
        history[
            "attempts"
        ]
        .to_dict()
    )


    # -----------------------------------------------------
    # Process current season in chronological order.
    #
    # We update history only AFTER assigning the feature.
    # -----------------------------------------------------

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


        # Only actual kicks update kicker ability.
        if (
            pdf.at[
                idx,
                "execution_status"
            ]
            == "NORMAL"
        ):

            row_X = (
                pdf.loc[
                    [idx],
                    difficulty_features,
                ]
            )


            expected = (
                difficulty_model
                .predict_proba(
                    row_X
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
        pdf[
            "adjusted_prior_attempts"
        ]
    )
)


# ---------------------------------------------------------
# Split
# ---------------------------------------------------------

train = pdf[
    pdf["split"] == "train"
].copy()

validation = pdf[
    pdf["split"] == "validation"
].copy()


y_train = train["fg_made"]
y_val = validation["fg_made"]


print("\nSIZES")

print(
    f"Train:      {len(train):,}"
)

print(
    f"Validation: {len(validation):,}"
)


# ---------------------------------------------------------
# Compare feature sets
# ---------------------------------------------------------

feature_sets = {

    "distance_only": [
        "fg_distance_estimate",
        "fg_distance_sq",
    ],

    "distance_roof": [
        "fg_distance_estimate",
        "fg_distance_sq",
        "roof_dome",
        "roof_closed",
        "roof_open",
    ],

    "distance_roof_adjusted_kicker": [
        "fg_distance_estimate",
        "fg_distance_sq",
        "roof_dome",
        "roof_closed",
        "roof_open",
        "adjusted_kicker_skill",
        "log_adjusted_prior_attempts",
    ],

    "full_adjusted": [
        "fg_distance_estimate",
        "fg_distance_sq",
        "roof_dome",
        "roof_closed",
        "roof_open",
        "outdoor_temp",
        "outdoor_wind",
        "outdoor_weather_missing",
        "adjusted_kicker_skill",
        "log_adjusted_prior_attempts",
    ],
}


predictions = {}
rows = []


for name, features in feature_sets.items():

    model = Pipeline([
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


    model.fit(
        train[features],
        y_train,
    )


    pred = (
        model
        .predict_proba(
            validation[
                features
            ]
        )[:, 1]
    )


    predictions[name] = pred


    rows.append({

        "model": name,

        "log_loss": log_loss(
            y_val,
            pred,
        ),

        "brier": brier_score_loss(
            y_val,
            pred,
        ),

        "auc": roc_auc_score(
            y_val,
            pred,
        ),

        "mean_prediction": (
            pred.mean()
        ),
    })


results = (
    pd.DataFrame(rows)
    .sort_values(
        "log_loss"
    )
)


print("\nMODEL COMPARISON")

print(
    results.to_string(
        index=False
    )
)


# ---------------------------------------------------------
# Adjusted kicker-skill distribution
# ---------------------------------------------------------

print(
    "\nVALIDATION ADJUSTED KICKER SKILL"
)


print(
    validation[
        [
            "adjusted_kicker_skill",
            "adjusted_prior_attempts",
        ]
    ]
    .describe()
    .to_string()
)


# ---------------------------------------------------------
# Most positive / negative established kickers
#
# Diagnostic only.
# ---------------------------------------------------------

kicker_diag = (
    validation[
        [
            "kicker_player_name",
            "adjusted_kicker_skill",
            "adjusted_prior_attempts",
        ]
    ]
    .groupby(
        "kicker_player_name"
    )
    .agg(
        skill=(
            "adjusted_kicker_skill",
            "mean",
        ),
        prior_attempts=(
            "adjusted_prior_attempts",
            "max",
        ),
        validation_kicks=(
            "adjusted_kicker_skill",
            "size",
        ),
    )
)


established = kicker_diag[
    kicker_diag[
        "prior_attempts"
    ] >= 30
]


print(
    "\nHIGHEST ADJUSTED KICKER SKILL"
)

print(
    established
    .sort_values(
        "skill",
        ascending=False,
    )
    .head(10)
    .to_string()
)


print(
    "\nLOWEST ADJUSTED KICKER SKILL"
)

print(
    established
    .sort_values(
        "skill",
        ascending=True,
    )
    .head(10)
    .to_string()
)


# ---------------------------------------------------------
# Bootstrap:
#
# full adjusted vs distance + roof
# ---------------------------------------------------------

print(
    "\nGAME-CLUSTER BOOTSTRAP — "
    "ADJUSTED FULL VS DISTANCE+ROOF"
)


base_pred = predictions[
    "distance_roof"
]

adjusted_pred = predictions[
    "full_adjusted"
]


boot = validation[
    [
        "game_id",
        "fg_made",
    ]
].copy()


boot["base_pred"] = (
    base_pred
)

boot["adjusted_pred"] = (
    adjusted_pred
)


games = (
    boot["game_id"]
    .unique()
)


game_rows = {
    game: boot[
        boot["game_id"] == game
    ]
    for game in games
}


rng = np.random.default_rng(
    42
)


ll_improvement = []
brier_improvement = []


for _ in range(2000):

    sampled_games = (
        rng.choice(
            games,
            size=len(games),
            replace=True,
        )
    )


    sample = pd.concat(
        [
            game_rows[game]
            for game in sampled_games
        ],
        ignore_index=True,
    )


    y = sample[
        "fg_made"
    ]


    base = sample[
        "base_pred"
    ]

    adjusted = sample[
        "adjusted_pred"
    ]


    ll_improvement.append(

        log_loss(
            y,
            base,
            labels=[0, 1],
        )
        -
        log_loss(
            y,
            adjusted,
            labels=[0, 1],
        )

    )


    brier_improvement.append(

        brier_score_loss(
            y,
            base,
        )
        -
        brier_score_loss(
            y,
            adjusted,
        )

    )


ll_improvement = np.array(
    ll_improvement
)

brier_improvement = np.array(
    brier_improvement
)


print("\nLog-loss improvement")

print(
    f"Mean: "
    f"{ll_improvement.mean():+.5f}"
)

print(
    f"95% interval: "
    f"["
    f"{np.quantile(ll_improvement, 0.025):+.5f}, "
    f"{np.quantile(ll_improvement, 0.975):+.5f}"
    f"]"
)

print(
    f"P(adjusted better): "
    f"{(ll_improvement > 0).mean():.3f}"
)


print("\nBrier improvement")

print(
    f"Mean: "
    f"{brier_improvement.mean():+.5f}"
)

print(
    f"95% interval: "
    f"["
    f"{np.quantile(brier_improvement, 0.025):+.5f}, "
    f"{np.quantile(brier_improvement, 0.975):+.5f}"
    f"]"
)

print(
    f"P(adjusted better): "
    f"{(brier_improvement > 0).mean():.3f}"
)