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
# Load FULL labeled data because we need kicker/weather
# columns that are not all carried through Script 07.
# ---------------------------------------------------------

df = pl.read_parquet(
    "data/fourth_downs_labeled.parquet"
)


fg = (
    df
    .filter(
        pl.col("action") == "FIELD_GOAL"
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


# ---------------------------------------------------------
# Time split
# ---------------------------------------------------------

fg = (
    fg
    .with_columns(

        pl.when(
            pl.col("season") <= 2022
        )
        .then(pl.lit("train"))

        .when(
            pl.col("season").is_between(
                2023,
                2024,
                closed="both",
            )
        )
        .then(pl.lit("validation"))

        .when(
            pl.col("season") == 2025
        )
        .then(pl.lit("test"))

        .otherwise(pl.lit("other"))
        .alias("split")

    )
)


# ---------------------------------------------------------
# Convert to pandas for chronological kicker features
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# Leakage-safe PRIOR kicker history
#
# Only NORMAL kicks count toward kicker history.
# Broken snaps do not tell us anything about kicking skill.
#
# For each kick, calculate stats from PRIOR kicks only.
# ---------------------------------------------------------

pdf["history_eligible"] = (
    (pdf["execution_status"] == "NORMAL")
    &
    pdf["kicker_player_id"].notna()
)


pdf["prior_attempts"] = 0.0
pdf["prior_makes"] = 0.0


kicker_attempts = {}
kicker_makes = {}


for i, row in pdf.iterrows():

    kicker = row["kicker_player_id"]

    if pd.notna(kicker):

        pdf.at[
            i,
            "prior_attempts"
        ] = kicker_attempts.get(
            kicker,
            0
        )

        pdf.at[
            i,
            "prior_makes"
        ] = kicker_makes.get(
            kicker,
            0
        )

    # Update history AFTER recording prior values.
    # This ensures the current kick cannot leak into itself.
    if row["history_eligible"]:

        kicker_attempts[kicker] = (
            kicker_attempts.get(
                kicker,
                0
            )
            + 1
        )

        kicker_makes[kicker] = (
            kicker_makes.get(
                kicker,
                0
            )
            + row["fg_made"]
        )


# ---------------------------------------------------------
# Smoothed prior kicker make rate
#
# Start a new kicker at an 85% prior with weight 20.
# As attempts accumulate, his actual prior results matter
# increasingly more.
# ---------------------------------------------------------

prior_strength = 20.0
prior_rate = 0.85


pdf["kicker_prior_make_rate"] = (
    pdf["prior_makes"]
    +
    prior_strength * prior_rate
) / (
    pdf["prior_attempts"]
    +
    prior_strength
)


pdf["log_prior_attempts"] = np.log1p(
    pdf["prior_attempts"]
)


# ---------------------------------------------------------
# Roof features
#
# outdoors is the reference category.
# ---------------------------------------------------------

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
# Weather
#
# Numeric temp/wind matter only for true outdoor games.
#
# Missing outdoor weather gets filled using TRAIN outdoor
# medians. Indoor/retractable-roof games receive 0 after
# interaction with is_outdoors.
# ---------------------------------------------------------

train_mask = (
    pdf["split"] == "train"
)

train_outdoor = (
    train_mask
    &
    (pdf["roof"] == "outdoors")
)


train_temp_median = (
    pdf.loc[
        train_outdoor,
        "temp"
    ]
    .median()
)

train_wind_median = (
    pdf.loc[
        train_outdoor,
        "wind"
    ]
    .median()
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


temp_filled = (
    pdf["temp"]
    .fillna(
        train_temp_median
    )
)

wind_filled = (
    pdf["wind"]
    .fillna(
        train_wind_median
    )
)


# Center temp so zero roughly means normal conditions.
pdf["outdoor_temp"] = (
    pdf["is_outdoors"]
    *
    (
        temp_filled
        - train_temp_median
    )
)

pdf["outdoor_wind"] = (
    pdf["is_outdoors"]
    *
    wind_filled
)


# ---------------------------------------------------------
# Distance curvature
# ---------------------------------------------------------

pdf["fg_distance_sq"] = (
    pdf["fg_distance_estimate"]
    ** 2
)


# ---------------------------------------------------------
# Train / validation only
#
# 2025 remains untouched.
# ---------------------------------------------------------

train = pdf[
    pdf["split"] == "train"
].copy()

validation = pdf[
    pdf["split"] == "validation"
].copy()


print("\nSIZES")

print(
    f"Train:      {len(train):,}"
)

print(
    f"Validation: {len(validation):,}"
)


print("\nTRAIN OUTDOOR WEATHER MEDIANS")

print(
    f"Temperature: "
    f"{train_temp_median:.1f}"
)

print(
    f"Wind: "
    f"{train_wind_median:.1f}"
)


# ---------------------------------------------------------
# Feature set
# ---------------------------------------------------------

features = [
    "fg_distance_estimate",
    "fg_distance_sq",

    "roof_dome",
    "roof_closed",
    "roof_open",

    "outdoor_temp",
    "outdoor_wind",
    "outdoor_weather_missing",

    "kicker_prior_make_rate",
    "log_prior_attempts",
]


X_train = train[features]
y_train = train["fg_made"]

X_val = validation[features]
y_val = validation["fg_made"]


# ---------------------------------------------------------
# Model
# ---------------------------------------------------------

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
    X_train,
    y_train,
)


pred = model.predict_proba(
    X_val
)[:, 1]


# ---------------------------------------------------------
# Metrics
# ---------------------------------------------------------

print("\nENHANCED FG VALIDATION")

print(
    f"Log loss: "
    f"{log_loss(y_val, pred):.4f}"
)

print(
    f"Brier score: "
    f"{brier_score_loss(y_val, pred):.4f}"
)

print(
    f"ROC AUC: "
    f"{roc_auc_score(y_val, pred):.4f}"
)

print(
    f"Actual make rate: "
    f"{y_val.mean():.4f}"
)

print(
    f"Mean predicted probability: "
    f"{pred.mean():.4f}"
)


# ---------------------------------------------------------
# Compare with old distance-only baseline
# ---------------------------------------------------------

distance_features = [
    "fg_distance_estimate",
    "fg_distance_sq",
]


baseline = Pipeline([
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


baseline.fit(
    train[distance_features],
    y_train,
)


baseline_pred = baseline.predict_proba(
    validation[distance_features]
)[:, 1]


print("\nDISTANCE-ONLY BASELINE")

print(
    f"Log loss: "
    f"{log_loss(y_val, baseline_pred):.4f}"
)

print(
    f"Brier score: "
    f"{brier_score_loss(y_val, baseline_pred):.4f}"
)

print(
    f"ROC AUC: "
    f"{roc_auc_score(y_val, baseline_pred):.4f}"
)


print("\nIMPROVEMENT VS DISTANCE ONLY")

print(
    f"Log-loss improvement: "
    f"{log_loss(y_val, baseline_pred) - log_loss(y_val, pred):+.4f}"
)

print(
    f"Brier improvement: "
    f"{brier_score_loss(y_val, baseline_pred) - brier_score_loss(y_val, pred):+.4f}"
)


# ---------------------------------------------------------
# Calibration by corrected distance
# ---------------------------------------------------------

result = validation[
    [
        "fg_distance_estimate",
        "fg_made",
        "kicker_player_name",
        "prior_attempts",
        "kicker_prior_make_rate",
        "roof",
    ]
].copy()


result["prediction"] = pred


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


calibration = (
    result
    .groupby(
        "distance_bucket",
        observed=True,
    )
    .agg(
        attempts=("fg_made", "size"),
        actual_make_rate=("fg_made", "mean"),
        predicted_make_rate=("prediction", "mean"),
    )
)


print("\nVALIDATION BY DISTANCE")

print(
    calibration.to_string()
)


# ---------------------------------------------------------
# Kicker-history diagnostics
# ---------------------------------------------------------

print("\nVALIDATION PRIOR KICKER EXPERIENCE")

experience_bins = pd.cut(
    validation["prior_attempts"],
    bins=[
        -1,
        0,
        10,
        25,
        50,
        100,
        10000,
    ],
    labels=[
        "0",
        "1-10",
        "11-25",
        "26-50",
        "51-100",
        "100+",
    ],
)


experience_summary = (
    validation
    .assign(
        experience_bucket=experience_bins
    )
    .groupby(
        "experience_bucket",
        observed=True,
    )
    .agg(
        kicks=("fg_made", "size"),
        actual_make_rate=("fg_made", "mean"),
        mean_prior_rate=("kicker_prior_make_rate", "mean"),
    )
)


print(
    experience_summary.to_string()
)


# ---------------------------------------------------------
# Coefficients
# ---------------------------------------------------------

coef = pd.Series(
    model.named_steps[
        "model"
    ].coef_[0],
    index=features,
).sort_values(
    key=np.abs,
    ascending=False,
)


print("\nSTANDARDIZED COEFFICIENTS")

print(
    coef.to_string()
)

# ---------------------------------------------------------
# Feature-group ablation
#
# Determine which groups actually improve validation.
# ---------------------------------------------------------

print("\nFEATURE GROUP ABLATION")


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

    "distance_roof_weather": [
        "fg_distance_estimate",
        "fg_distance_sq",
        "roof_dome",
        "roof_closed",
        "roof_open",
        "outdoor_temp",
        "outdoor_wind",
        "outdoor_weather_missing",
    ],

    "distance_kicker_rate": [
        "fg_distance_estimate",
        "fg_distance_sq",
        "kicker_prior_make_rate",
    ],

    "distance_kicker_experience": [
        "fg_distance_estimate",
        "fg_distance_sq",
        "log_prior_attempts",
    ],

    "distance_kicker_all": [
        "fg_distance_estimate",
        "fg_distance_sq",
        "kicker_prior_make_rate",
        "log_prior_attempts",
    ],

    "full": features,
}


ablation_predictions = {}
ablation_rows = []


for name, cols in feature_sets.items():

    temp_model = Pipeline([
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

    temp_model.fit(
        train[cols],
        y_train,
    )

    temp_pred = temp_model.predict_proba(
        validation[cols]
    )[:, 1]

    ablation_predictions[name] = temp_pred

    ablation_rows.append({
        "model": name,

        "log_loss": log_loss(
            y_val,
            temp_pred,
        ),

        "brier": brier_score_loss(
            y_val,
            temp_pred,
        ),

        "auc": roc_auc_score(
            y_val,
            temp_pred,
        ),

        "mean_prediction": temp_pred.mean(),
    })


ablation_table = (
    pd.DataFrame(
        ablation_rows
    )
    .sort_values(
        "log_loss"
    )
)


print(
    ablation_table.to_string(
        index=False
    )
)


# ---------------------------------------------------------
# Game-cluster bootstrap:
# full model versus distance-only
#
# Positive improvement means FULL is better.
# ---------------------------------------------------------

print("\nGAME-CLUSTER BOOTSTRAP — FULL VS DISTANCE ONLY")


full_pred = ablation_predictions[
    "full"
]

distance_pred = ablation_predictions[
    "distance_only"
]


validation_boot = validation[
    [
        "game_id",
        "fg_made",
    ]
].copy()

validation_boot[
    "full_pred"
] = full_pred

validation_boot[
    "distance_pred"
] = distance_pred


games = validation_boot[
    "game_id"
].unique()


game_rows = {
    game: validation_boot[
        validation_boot[
            "game_id"
        ] == game
    ]
    for game in games
}


rng = np.random.default_rng(42)

logloss_improvements = []
brier_improvements = []


for _ in range(2000):

    sampled_games = rng.choice(
        games,
        size=len(games),
        replace=True,
    )

    sample = pd.concat(
        [
            game_rows[game]
            for game in sampled_games
        ],
        ignore_index=True,
    )

    y_sample = sample[
        "fg_made"
    ]

    full_sample = sample[
        "full_pred"
    ]

    distance_sample = sample[
        "distance_pred"
    ]


    distance_ll = log_loss(
        y_sample,
        distance_sample,
        labels=[0, 1],
    )

    full_ll = log_loss(
        y_sample,
        full_sample,
        labels=[0, 1],
    )


    distance_brier = brier_score_loss(
        y_sample,
        distance_sample,
    )

    full_brier = brier_score_loss(
        y_sample,
        full_sample,
    )


    logloss_improvements.append(
        distance_ll - full_ll
    )

    brier_improvements.append(
        distance_brier - full_brier
    )


logloss_improvements = np.array(
    logloss_improvements
)

brier_improvements = np.array(
    brier_improvements
)


print(
    "Log-loss improvement:"
)

print(
    f"mean = "
    f"{logloss_improvements.mean():+.5f}"
)

print(
    f"95% interval = "
    f"["
    f"{np.quantile(logloss_improvements, 0.025):+.5f}, "
    f"{np.quantile(logloss_improvements, 0.975):+.5f}"
    f"]"
)

print(
    f"P(full better) = "
    f"{(logloss_improvements > 0).mean():.3f}"
)


print(
    "\nBrier improvement:"
)

print(
    f"mean = "
    f"{brier_improvements.mean():+.5f}"
)

print(
    f"95% interval = "
    f"["
    f"{np.quantile(brier_improvements, 0.025):+.5f}, "
    f"{np.quantile(brier_improvements, 0.975):+.5f}"
    f"]"
)

print(
    f"P(full better) = "
    f"{(brier_improvements > 0).mean():.3f}"
)