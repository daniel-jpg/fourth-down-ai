import polars as pl
import pandas as pd

from xgboost import XGBClassifier

from sklearn.metrics import (
    log_loss,
    brier_score_loss,
    roc_auc_score,
    accuracy_score,
)


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

go = go.with_columns(
    (pl.col("action") == "NORMAL_GO_PASS")
    .cast(pl.Int8)
    .alias("is_pass")
)


# ---------------------------------------------------------
# Features
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


X_train = train.select(features).to_pandas()
y_train = train[target].to_pandas()

X_val = validation.select(features).to_pandas()
y_val = validation[target].to_pandas()


# ---------------------------------------------------------
# XGBoost model
#
# Conservative parameters because we only have
# ~5,000 training normal-go plays.
# ---------------------------------------------------------

model = XGBClassifier(

    objective="binary:logistic",
    eval_metric="logloss",

    n_estimators=500,
    learning_rate=0.03,

    max_depth=3,
    min_child_weight=15,

    subsample=0.9,
    colsample_bytree=0.9,

    reg_lambda=10.0,
    reg_alpha=0.5,

    tree_method="hist",

    random_state=42,
    n_jobs=-1,
)

model.fit(
    X_train,
    y_train,
)


# ---------------------------------------------------------
# Validation evaluation
# ---------------------------------------------------------

probabilities = model.predict_proba(
    X_val
)[:, 1]

predictions = (
    probabilities >= 0.5
).astype(int)


print("\nXGBOOST VALIDATION RESULTS")

print(
    f"Log loss: "
    f"{log_loss(y_val, probabilities):.4f}"
)

print(
    f"Brier score: "
    f"{brier_score_loss(y_val, probabilities):.4f}"
)

print(
    f"ROC AUC: "
    f"{roc_auc_score(y_val, probabilities):.4f}"
)

print(
    f"Accuracy: "
    f"{accuracy_score(y_val, predictions):.4f}"
)

print(
    f"Actual conversion rate: "
    f"{y_val.mean():.4f}"
)

print(
    f"Mean predicted probability: "
    f"{probabilities.mean():.4f}"
)


# ---------------------------------------------------------
# Calibration table
# ---------------------------------------------------------

calibration = pd.DataFrame({
    "actual": y_val.to_numpy(),
    "predicted": probabilities,
})

calibration["bin"] = pd.qcut(
    calibration["predicted"],
    q=10,
    duplicates="drop",
)

calibration_table = (
    calibration
    .groupby(
        "bin",
        observed=True,
    )
    .agg(
        plays=("actual", "size"),
        avg_prediction=("predicted", "mean"),
        actual_rate=("actual", "mean"),
    )
)


print("\nCALIBRATION BY PREDICTED-PROBABILITY DECILE")

print(
    calibration_table.to_string()
)


# ---------------------------------------------------------
# Feature importance
# ---------------------------------------------------------

importance = pd.DataFrame({
    "feature": features,
    "importance": model.feature_importances_,
}).sort_values(
    "importance",
    ascending=False,
)


print("\nFEATURE IMPORTANCE")

print(
    importance.to_string(
        index=False
    )
)


# ---------------------------------------------------------
# Exploratory counterfactual predictions
#
# For every validation state:
#
#   What does this model predict if RUN?
#   What does this model predict if PASS?
#
# IMPORTANT:
# These are NOT yet causal recommendations.
# We will audit selection bias / overlap next.
# ---------------------------------------------------------

X_run = X_val.copy()
X_run["is_pass"] = 0

X_pass = X_val.copy()
X_pass["is_pass"] = 1


p_run = model.predict_proba(
    X_run
)[:, 1]

p_pass = model.predict_proba(
    X_pass
)[:, 1]


counterfactual = validation.select([
    "ydstogo",
    "yardline_100",
]).to_pandas()

counterfactual["p_run"] = p_run
counterfactual["p_pass"] = p_pass

counterfactual["run_advantage"] = (
    counterfactual["p_run"]
    -
    counterfactual["p_pass"]
)


print("\nSAME-STATE RUN VS PASS")

print(
    f"Mean predicted RUN conversion: "
    f"{p_run.mean():.4f}"
)

print(
    f"Mean predicted PASS conversion: "
    f"{p_pass.mean():.4f}"
)

print(
    f"States favoring RUN: "
    f"{(p_run > p_pass).mean():.4f}"
)

print(
    f"States favoring PASS: "
    f"{(p_pass > p_run).mean():.4f}"
)


# ---------------------------------------------------------
# Run/pass comparison by yards to go
# ---------------------------------------------------------

counterfactual["distance_bucket"] = pd.cut(
    counterfactual["ydstogo"],
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


distance_summary = (
    counterfactual
    .groupby(
        "distance_bucket",
        observed=True,
    )
    .agg(
        plays=("p_run", "size"),
        p_run=("p_run", "mean"),
        p_pass=("p_pass", "mean"),
        run_advantage=("run_advantage", "mean"),
    )
)


print("\nCOUNTERFACTUAL BY YARDS TO GO")

print(
    distance_summary.to_string()
)