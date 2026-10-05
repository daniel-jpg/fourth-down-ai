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
# Load development data only.
# 2025 remains untouched.
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
        &
        (pl.col("execution_status") == "NORMAL")
    )
    .with_columns([

        (
            pl.col("yardline_100") + 18
        ).alias("fg_distance_estimate"),

        pl.col("field_goal_result")
        .fill_null("")
        .str.to_lowercase()
        .alias("fg_outcome"),

        pl.when(
            pl.col("season") <= 2022
        )
        .then(pl.lit("train"))
        .otherwise(pl.lit("validation"))
        .alias("split"),

    ])
)


# ---------------------------------------------------------
# Clean failed kicks only.
# ---------------------------------------------------------

failures = (
    fg
    .filter(
        pl.col("fg_outcome").is_in([
            "missed",
            "blocked",
        ])
    )
    .with_columns(
        (
            pl.col("fg_outcome")
            == "blocked"
        )
        .cast(pl.Int8)
        .alias("blocked")
    )
)


train = (
    failures
    .filter(
        pl.col("split") == "train"
    )
    .to_pandas()
)

validation = (
    failures
    .filter(
        pl.col("split") == "validation"
    )
    .to_pandas()
)


print("\nSIZES")

print(
    f"Train failures:      {len(train):,}"
)

print(
    f"Validation failures: {len(validation):,}"
)


print(
    f"\nTrain block rate: "
    f"{train['blocked'].mean():.4f}"
)

print(
    f"Validation block rate: "
    f"{validation['blocked'].mean():.4f}"
)


# ---------------------------------------------------------
# Constant baseline
# ---------------------------------------------------------

train_block_rate = (
    train["blocked"].mean()
)


constant_pred = np.full(
    len(validation),
    train_block_rate,
)


# ---------------------------------------------------------
# Linear distance model
#
# Keep this intentionally simple because there are only
# 159 blocks in training.
# ---------------------------------------------------------

features = [
    "fg_distance_estimate",
]


model = Pipeline([
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


model.fit(
    train[features],
    train["blocked"],
)


distance_pred = (
    model
    .predict_proba(
        validation[features]
    )[:, 1]
)


# ---------------------------------------------------------
# Metrics
# ---------------------------------------------------------

def metrics(
    name,
    y,
    pred,
):

    print(
        f"\n{name}"
    )

    print(
        f"Log loss: "
        f"{log_loss(y, pred):.5f}"
    )

    print(
        f"Brier: "
        f"{brier_score_loss(y, pred):.5f}"
    )

    print(
        f"Mean prediction: "
        f"{pred.mean():.5f}"
    )


y_val = validation[
    "blocked"
]


metrics(
    "CONSTANT BLOCK RATE",
    y_val,
    constant_pred,
)


metrics(
    "DISTANCE LOGISTIC",
    y_val,
    distance_pred,
)


print(
    f"\nDistance-model ROC AUC: "
    f"{roc_auc_score(y_val, distance_pred):.4f}"
)


# ---------------------------------------------------------
# Coefficient direction
# ---------------------------------------------------------

coef = (
    model
    .named_steps["model"]
    .coef_[0][0]
)


print(
    f"\nStandardized distance coefficient: "
    f"{coef:+.4f}"
)


# ---------------------------------------------------------
# Validation calibration by distance
# ---------------------------------------------------------

result = validation[
    [
        "fg_distance_estimate",
        "blocked",
    ]
].copy()


result[
    "prediction"
] = distance_pred


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


result[
    "distance_bucket"
] = pd.cut(
    result[
        "fg_distance_estimate"
    ],
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
        failures=(
            "blocked",
            "size",
        ),
        actual_block_rate=(
            "blocked",
            "mean",
        ),
        predicted_block_rate=(
            "prediction",
            "mean",
        ),
    )
)


print(
    "\nVALIDATION BY DISTANCE"
)

print(
    calibration.to_string()
)


# ---------------------------------------------------------
# Game-cluster bootstrap:
# distance model vs constant.
#
# Positive = distance model better.
# ---------------------------------------------------------

print(
    "\nGAME-CLUSTER BOOTSTRAP"
)


boot = validation[
    [
        "game_id",
        "blocked",
    ]
].copy()


boot[
    "constant_pred"
] = constant_pred

boot[
    "distance_pred"
] = distance_pred


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

    y = sample[
        "blocked"
    ]

    constant = sample[
        "constant_pred"
    ]

    distance = sample[
        "distance_pred"
    ]


    ll_improvement.append(
        log_loss(
            y,
            constant,
            labels=[0, 1],
        )
        -
        log_loss(
            y,
            distance,
            labels=[0, 1],
        )
    )


    brier_improvement.append(
        brier_score_loss(
            y,
            constant,
        )
        -
        brier_score_loss(
            y,
            distance,
        )
    )


ll_improvement = np.array(
    ll_improvement
)

brier_improvement = np.array(
    brier_improvement
)


print(
    "\nLog-loss improvement"
)

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
    f"P(distance better): "
    f"{(ll_improvement > 0).mean():.3f}"
)


print(
    "\nBrier improvement"
)

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
    f"P(distance better): "
    f"{(brier_improvement > 0).mean():.3f}"
)