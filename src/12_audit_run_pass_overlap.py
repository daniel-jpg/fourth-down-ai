import polars as pl
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss


# ---------------------------------------------------------
# Load data
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


# ---------------------------------------------------------
# Historical treatment/action
#
# 0 = run
# 1 = pass
#
# Here it is OK to predict historical coach choice.
# We are NOT using this as the recommendation model.
# This is only to measure selection / overlap.
# ---------------------------------------------------------

go = go.with_columns(
    (pl.col("action") == "NORMAL_GO_PASS")
    .cast(pl.Int8)
    .alias("is_pass")
)


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
]


# ---------------------------------------------------------
# Split
# ---------------------------------------------------------

train = go.filter(
    pl.col("split") == "train"
)

validation = go.filter(
    pl.col("split") == "validation"
)


X_train = train.select(features).to_pandas()
t_train = train["is_pass"].to_pandas()

X_val = validation.select(features).to_pandas()
t_val = validation["is_pass"].to_pandas()


# ---------------------------------------------------------
# Propensity model
#
# e(x) = P(PASS | pre-play state)
#
# High predictive ability here means coaches strongly select
# run/pass based on state.
# ---------------------------------------------------------

propensity_model = Pipeline([
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

propensity_model.fit(
    X_train,
    t_train,
)


propensity = propensity_model.predict_proba(
    X_val
)[:, 1]


print("\nPROPENSITY MODEL")

print(
    f"ROC AUC: "
    f"{roc_auc_score(t_val, propensity):.4f}"
)

print(
    f"Log loss: "
    f"{log_loss(t_val, propensity):.4f}"
)

print(
    f"Actual pass rate: "
    f"{t_val.mean():.4f}"
)

print(
    f"Mean predicted pass probability: "
    f"{propensity.mean():.4f}"
)


# ---------------------------------------------------------
# Overall common support
# ---------------------------------------------------------

print("\nCOMMON SUPPORT")

for low in [
    0.01,
    0.05,
    0.10,
    0.20,
]:

    high = 1 - low

    inside = (
        (propensity >= low)
        &
        (propensity <= high)
    )

    print(
        f"{low:.2f} <= propensity <= {high:.2f}: "
        f"{inside.mean():.4f} "
        f"({inside.sum()} / {len(inside)})"
    )


# ---------------------------------------------------------
# Support separately for actual RUN and PASS
# ---------------------------------------------------------

audit = validation.select([
    "action",
    "ydstogo",
    "yardline_100",
]).to_pandas()

audit["is_pass"] = t_val.to_numpy()

audit["propensity_pass"] = propensity


print("\nPROPENSITY BY ACTUAL ACTION")

summary = (
    audit
    .groupby("action")
    .agg(
        plays=("propensity_pass", "size"),
        mean_propensity=("propensity_pass", "mean"),
        median_propensity=("propensity_pass", "median"),
        min_propensity=("propensity_pass", "min"),
        max_propensity=("propensity_pass", "max"),
    )
)

print(
    summary.to_string()
)


# ---------------------------------------------------------
# Propensity quantiles by actual action
# ---------------------------------------------------------

print("\nPROPENSITY QUANTILES BY ACTUAL ACTION")

for action in [
    "NORMAL_GO_RUN",
    "NORMAL_GO_PASS",
]:

    values = (
        audit
        .loc[
            audit["action"] == action,
            "propensity_pass"
        ]
    )

    quantiles = values.quantile([
        0.05,
        0.10,
        0.25,
        0.50,
        0.75,
        0.90,
        0.95,
    ])

    print(f"\n{action}")

    print(
        quantiles.to_string()
    )


# ---------------------------------------------------------
# Distance buckets
# ---------------------------------------------------------

audit["distance_bucket"] = pd.cut(
    audit["ydstogo"],
    bins=[
        0,
        1,
        2,
        3,
        5,
        10,
        100,
    ],
    labels=[
        "1",
        "2",
        "3",
        "4-5",
        "6-10",
        "11+",
    ],
)


distance = (
    audit
    .groupby(
        "distance_bucket",
        observed=True,
    )
    .agg(
        plays=("is_pass", "size"),
        actual_pass_rate=("is_pass", "mean"),
        mean_propensity=("propensity_pass", "mean"),
    )
)


print("\nRUN/PASS SELECTION BY YARDS TO GO")

print(
    distance.to_string()
)


# ---------------------------------------------------------
# Extreme propensity states
# ---------------------------------------------------------

audit["poor_overlap"] = (
    (audit["propensity_pass"] < 0.10)
    |
    (audit["propensity_pass"] > 0.90)
)


print("\nPOOR-OVERLAP STATES")

print(
    f"Rows outside 0.10-0.90: "
    f"{audit['poor_overlap'].sum()}"
)

print(
    f"Fraction outside 0.10-0.90: "
    f"{audit['poor_overlap'].mean():.4f}"
)


print("\nPOOR OVERLAP BY DISTANCE")

poor_by_distance = (
    audit
    .groupby(
        "distance_bucket",
        observed=True,
    )
    .agg(
        plays=("poor_overlap", "size"),
        poor_overlap_rate=("poor_overlap", "mean"),
        actual_pass_rate=("is_pass", "mean"),
    )
)

print(
    poor_by_distance.to_string()
)