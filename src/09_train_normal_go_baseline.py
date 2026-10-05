import polars as pl
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    log_loss,
    brier_score_loss,
    roc_auc_score,
    accuracy_score,
)


# ---------------------------------------------------------
# Load split data
# ---------------------------------------------------------

df = pl.read_parquet(
    "data/fourth_down_modeling_split.parquet"
)

go = df.filter(
    pl.col("action").is_in([
        "NORMAL_GO_RUN",
        "NORMAL_GO_PASS",
    ])
)

print(f"Normal go plays: {go.height:,}")


# ---------------------------------------------------------
# Add action indicator
#
# 0 = run
# 1 = pass
# ---------------------------------------------------------

go = go.with_columns(
    (pl.col("action") == "NORMAL_GO_PASS")
    .cast(pl.Int8)
    .alias("is_pass")
)


# ---------------------------------------------------------
# ONLY pre-play features
#
# No yards_gained, touchdown, converted, field_goal_result,
# or anything else that happens after the snap.
# ---------------------------------------------------------

features = [
    "ydstogo",
    "yardline_100",
    "goal_to_go",

    "qtr",
    "game_seconds_remaining",

    "score_differential",

    "posteam_timeouts_remaining",
    "defteam_timeouts_remaining",

    "is_home",

    "spread_line",
    "total_line",

    "is_pass",
]

target = "converted"


# ---------------------------------------------------------
# Split
# ---------------------------------------------------------

train = go.filter(
    pl.col("split") == "train"
)

validation = go.filter(
    pl.col("split") == "validation"
)

test = go.filter(
    pl.col("split") == "test"
)


print("\nSIZES")
print(f"Train:      {train.height:,}")
print(f"Validation: {validation.height:,}")
print(f"Test:       {test.height:,}")


# ---------------------------------------------------------
# Convert to pandas for scikit-learn
# ---------------------------------------------------------

X_train = train.select(features).to_pandas()
y_train = train[target].to_pandas()

X_val = validation.select(features).to_pandas()
y_val = validation[target].to_pandas()

X_test = test.select(features).to_pandas()
y_test = test[target].to_pandas()


# ---------------------------------------------------------
# Baseline logistic regression
# ---------------------------------------------------------

model = Pipeline([
    (
        "scaler",
        StandardScaler()
    ),
    (
        "model",
        LogisticRegression(
            max_iter=2000
        )
    ),
])

model.fit(
    X_train,
    y_train,
)


# ---------------------------------------------------------
# Evaluation function
# ---------------------------------------------------------

def evaluate(name, X, y):

    probabilities = model.predict_proba(X)[:, 1]

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    print(f"\n{name}")

    print(
        f"Log loss: "
        f"{log_loss(y, probabilities):.4f}"
    )

    print(
        f"Brier score: "
        f"{brier_score_loss(y, probabilities):.4f}"
    )

    print(
        f"ROC AUC: "
        f"{roc_auc_score(y, probabilities):.4f}"
    )

    print(
        f"Accuracy: "
        f"{accuracy_score(y, predictions):.4f}"
    )

    print(
        f"Actual conversion rate: "
        f"{y.mean():.4f}"
    )

    print(
        f"Mean predicted probability: "
        f"{probabilities.mean():.4f}"
    )


# ---------------------------------------------------------
# Evaluate validation only for now
#
# Do NOT touch the 2025 test set yet.
# ---------------------------------------------------------

evaluate(
    "VALIDATION RESULTS",
    X_val,
    y_val,
)


# ---------------------------------------------------------
# Basic run/pass summaries
# ---------------------------------------------------------

print("\nVALIDATION CONVERSION RATE BY ACTION")

print(
    validation
    .group_by("action")
    .agg([
        pl.len().alias("attempts"),
        pl.col("converted").mean().alias(
            "conversion_rate"
        ),
        pl.col("ydstogo").mean().alias(
            "avg_ydstogo"
        ),
    ])
    .sort("action")
)


print("\nMODEL FEATURES")

for feature in features:
    print(feature)