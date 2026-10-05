import polars as pl
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    log_loss,
    brier_score_loss,
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


train = go.filter(
    pl.col("split") == "train"
)

validation = go.filter(
    pl.col("split") == "validation"
)


# ---------------------------------------------------------
# Convert to pandas
# ---------------------------------------------------------

train_pd = train.select(
    features
    + [
        "is_pass",
        "converted",
        "game_id",
    ]
).to_pandas()

val_pd = validation.select(
    features
    + [
        "is_pass",
        "converted",
        "game_id",
    ]
).to_pandas()


X_train = train_pd[features]
t_train = train_pd["is_pass"]
y_train = train_pd["converted"]

X_val = val_pd[features]
t_val = val_pd["is_pass"].to_numpy()
y_val = val_pd["converted"].to_numpy()


# ---------------------------------------------------------
# Propensity model
#
# e(x) = P(PASS | X)
# ---------------------------------------------------------

propensity_model = Pipeline([
    (
        "scaler",
        StandardScaler(),
    ),
    (
        "model",
        LogisticRegression(
            max_iter=2000
        ),
    ),
])

propensity_model.fit(
    X_train,
    t_train,
)

e = propensity_model.predict_proba(
    X_val
)[:, 1]


# ---------------------------------------------------------
# Separate outcome models
#
# m0(x) = P(convert | RUN, X)
# m1(x) = P(convert | PASS, X)
#
# Important:
# We fit separate models rather than forcing one global
# run/pass coefficient.
# ---------------------------------------------------------

run_train = train_pd[
    train_pd["is_pass"] == 0
]

pass_train = train_pd[
    train_pd["is_pass"] == 1
]


run_model = Pipeline([
    (
        "scaler",
        StandardScaler(),
    ),
    (
        "model",
        LogisticRegression(
            max_iter=2000
        ),
    ),
])

pass_model = Pipeline([
    (
        "scaler",
        StandardScaler(),
    ),
    (
        "model",
        LogisticRegression(
            max_iter=2000
        ),
    ),
])


run_model.fit(
    run_train[features],
    run_train["converted"],
)

pass_model.fit(
    pass_train[features],
    pass_train["converted"],
)


m0 = run_model.predict_proba(
    X_val
)[:, 1]

m1 = pass_model.predict_proba(
    X_val
)[:, 1]


# ---------------------------------------------------------
# Evaluate each outcome model ONLY on plays where that
# action actually happened
# ---------------------------------------------------------

actual_run = (
    t_val == 0
)

actual_pass = (
    t_val == 1
)


print("\nOUTCOME MODEL VALIDATION")

print("\nRUN MODEL")

print(
    f"Plays: "
    f"{actual_run.sum()}"
)

print(
    f"Log loss: "
    f"{log_loss(
        y_val[actual_run],
        m0[actual_run]
    ):.4f}"
)

print(
    f"Brier: "
    f"{brier_score_loss(
        y_val[actual_run],
        m0[actual_run]
    ):.4f}"
)


print("\nPASS MODEL")

print(
    f"Plays: "
    f"{actual_pass.sum()}"
)

print(
    f"Log loss: "
    f"{log_loss(
        y_val[actual_pass],
        m1[actual_pass]
    ):.4f}"
)

print(
    f"Brier: "
    f"{brier_score_loss(
        y_val[actual_pass],
        m1[actual_pass]
    ):.4f}"
)


# ---------------------------------------------------------
# Common support restriction
#
# Only compare RUN vs PASS when both actions have
# reasonable historical probability.
# ---------------------------------------------------------

support = (
    (e >= 0.10)
    &
    (e <= 0.90)
)


print("\nCOMMON SUPPORT")

print(
    f"Supported rows: "
    f"{support.sum()} / {len(support)}"
)

print(
    f"Supported fraction: "
    f"{support.mean():.4f}"
)


# ---------------------------------------------------------
# Doubly robust pseudo-outcome
#
# Treatment:
#   T = 1 means PASS
#   T = 0 means RUN
#
# tau > 0 => PASS improves conversion probability
# tau < 0 => RUN improves conversion probability
# ---------------------------------------------------------

e_safe = np.clip(
    e,
    0.10,
    0.90,
)

tau_dr = (
    m1
    - m0
    + (
        t_val
        * (y_val - m1)
        / e_safe
    )
    - (
        (1 - t_val)
        * (y_val - m0)
        / (1 - e_safe)
    )
)


result = val_pd.copy()

result["propensity_pass"] = e
result["p_run"] = m0
result["p_pass"] = m1
result["tau_dr"] = tau_dr
result["supported"] = support


supported = result[
    result["supported"]
].copy()


# ---------------------------------------------------------
# Overall effect
# ---------------------------------------------------------

print("\nDOUBLY ROBUST EFFECT")

effect = supported["tau_dr"].mean()

print(
    f"PASS minus RUN conversion effect: "
    f"{effect:+.4f}"
)

if effect > 0:

    print(
        "Interpretation: PASS is estimated to "
        "increase conversion probability."
    )

else:

    print(
        "Interpretation: RUN is estimated to "
        "increase conversion probability."
    )


# ---------------------------------------------------------
# Distance buckets
# ---------------------------------------------------------

supported["distance_bucket"] = pd.cut(
    supported["ydstogo"],
    bins=[
        0,
        1,
        2,
        3,
        5,
        100,
    ],
    labels=[
        "1",
        "2",
        "3",
        "4-5",
        "6+",
    ],
)


distance = (
    supported
    .groupby(
        "distance_bucket",
        observed=True,
    )
    .agg(
        plays=("tau_dr", "size"),
        avg_propensity=("propensity_pass", "mean"),
        p_run=("p_run", "mean"),
        p_pass=("p_pass", "mean"),
        dr_pass_minus_run=("tau_dr", "mean"),
    )
)


print("\nDOUBLY ROBUST EFFECT BY DISTANCE")

print(
    distance.to_string()
)


# ---------------------------------------------------------
# Simple uncertainty estimate
#
# Cluster bootstrap by game so plays from the same game
# stay together.
# ---------------------------------------------------------

rng = np.random.default_rng(42)

games = supported[
    "game_id"
].unique()

bootstrap_effects = []


for _ in range(1000):

    sampled_games = rng.choice(
        games,
        size=len(games),
        replace=True,
    )

    pieces = []

    for game in sampled_games:

        pieces.append(
            supported[
                supported["game_id"] == game
            ]
        )

    sample = pd.concat(
        pieces,
        ignore_index=True,
    )

    bootstrap_effects.append(
        sample["tau_dr"].mean()
    )


lower = np.quantile(
    bootstrap_effects,
    0.025,
)

upper = np.quantile(
    bootstrap_effects,
    0.975,
)


print("\n95% CLUSTER-BOOTSTRAP INTERVAL")

print(
    f"Effect: "
    f"{effect:+.4f}"
)

print(
    f"95% interval: "
    f"[{lower:+.4f}, {upper:+.4f}]"
)