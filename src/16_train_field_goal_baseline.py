import polars as pl
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    log_loss,
    brier_score_loss,
    roc_auc_score,
    accuracy_score,
)


# ---------------------------------------------------------
# Load split modeling data
# ---------------------------------------------------------

df = pl.read_parquet(
    "data/fourth_down_modeling_split.parquet"
)


fg = (
    df
    .filter(
        pl.col("action") == "FIELD_GOAL"
    )
    .with_columns([

        (
            pl.col("field_goal_result")
            .fill_null("")
            .str.to_lowercase()
            == "made"
        )
        .cast(pl.Int8)
        .alias("fg_made"),

        pl.when(
            pl.col("execution_status") == "BROKEN"
        )
        .then(
            pl.lit("broken")
        )
        .otherwise(
            pl.col("field_goal_result")
            .fill_null("unknown")
            .str.to_lowercase()
        )
        .alias("fg_outcome"),

    ])
)


print(
    f"Field-goal decisions: {fg.height:,}"
)


# ---------------------------------------------------------
# Outcome audit
# ---------------------------------------------------------

print("\nFG OUTCOMES")

print(
    fg
    .group_by("fg_outcome")
    .len()
    .sort("len", descending=True)
)


print("\nFG OUTCOMES BY SPLIT")

print(
    fg
    .group_by([
        "split",
        "fg_outcome",
    ])
    .len()
    .sort([
        "split",
        "fg_outcome",
    ])
)


# ---------------------------------------------------------
# Train / validation
#
# Do NOT evaluate test yet.
# ---------------------------------------------------------

train = (
    fg
    .filter(
        pl.col("split") == "train"
    )
    .to_pandas()
)

validation = (
    fg
    .filter(
        pl.col("split") == "validation"
    )
    .to_pandas()
)


print("\nSIZES")

print(
    f"Train:      {len(train):,}"
)

print(
    f"Validation: {len(validation):,}"
)


# ---------------------------------------------------------
# Distance support
# ---------------------------------------------------------

print("\nDISTANCE SUPPORT")

print(
    "Train:",
    f"{train['fg_distance_estimate'].min():.0f}",
    "to",
    f"{train['fg_distance_estimate'].max():.0f}",
)

print(
    "Validation:",
    f"{validation['fg_distance_estimate'].min():.0f}",
    "to",
    f"{validation['fg_distance_estimate'].max():.0f}",
)


# ---------------------------------------------------------
# Baseline model
#
# Start with distance only.
#
# Quadratic term allows the success curve to bend rather
# than forcing a straight-line effect on log-odds.
# ---------------------------------------------------------

features = [
    "fg_distance_estimate",
]


X_train = train[features]
y_train = train["fg_made"]

X_val = validation[features]
y_val = validation["fg_made"]


model = Pipeline([
    (
        "poly",
        PolynomialFeatures(
            degree=2,
            include_bias=False,
        ),
    ),
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
    X_train,
    y_train,
)


pred = model.predict_proba(
    X_val
)[:, 1]


# ---------------------------------------------------------
# Validation metrics
# ---------------------------------------------------------

print("\nVALIDATION RESULTS")

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
    f"Accuracy: "
    f"{accuracy_score(y_val, pred >= 0.5):.4f}"
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
# Calibration by kick distance
# ---------------------------------------------------------

result = validation[
    [
        "fg_distance_estimate",
        "fg_made",
        "execution_status",
        "fg_outcome",
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
# Broken execution audit
# ---------------------------------------------------------

print("\nBROKEN FG DECISIONS IN VALIDATION")

broken_val = result[
    result["execution_status"] == "BROKEN"
]

print(
    broken_val[
        [
            "fg_distance_estimate",
            "fg_outcome",
            "prediction",
        ]
    ].to_string(
        index=False
    )
)