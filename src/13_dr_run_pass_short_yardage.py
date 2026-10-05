import polars as pl
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, brier_score_loss


df = pl.read_parquet(
    "data/fourth_down_modeling_split.parquet"
)

go = (
    df
    .filter(
        pl.col("action").is_in([
            "NORMAL_GO_RUN",
            "NORMAL_GO_PASS",
        ])
    )
    .filter(
        pl.col("ydstogo") <= 2
    )
    .with_columns(
        (pl.col("action") == "NORMAL_GO_PASS")
        .cast(pl.Int8)
        .alias("is_pass")
    )
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


train = (
    go
    .filter(pl.col("split") == "train")
    .select(
        features
        + [
            "is_pass",
            "converted",
            "game_id",
            "action",
        ]
    )
    .to_pandas()
)

validation = (
    go
    .filter(pl.col("split") == "validation")
    .select(
        features
        + [
            "is_pass",
            "converted",
            "game_id",
            "action",
        ]
    )
    .to_pandas()
)


X_train = train[features]
t_train = train["is_pass"]
y_train = train["converted"]

X_val = validation[features]
t_val = validation["is_pass"].to_numpy()
y_val = validation["converted"].to_numpy()


print("\nSIZES")
print(f"Train: {len(train):,}")
print(f"Validation: {len(validation):,}")


print("\nVALIDATION ACTION COUNTS")
print(
    validation
    .groupby(["ydstogo", "action"])
    .size()
    .to_string()
)


# ---------------------------------------------------------
# Propensity model
# e(x) = P(PASS | X)
# ---------------------------------------------------------

propensity_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=2000)),
])

propensity_model.fit(
    X_train,
    t_train,
)

e = propensity_model.predict_proba(
    X_val
)[:, 1]


# ---------------------------------------------------------
# Outcome models
# ---------------------------------------------------------

run_train = train[
    train["is_pass"] == 0
]

pass_train = train[
    train["is_pass"] == 1
]


run_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=2000)),
])

pass_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=2000)),
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
# Validation quality on observed actions
# ---------------------------------------------------------

actual_run = t_val == 0
actual_pass = t_val == 1


print("\nOUTCOME MODEL VALIDATION")

print("\nRUN")
print(f"Plays: {actual_run.sum()}")
print(
    f"Log loss: "
    f"{log_loss(y_val[actual_run], m0[actual_run]):.4f}"
)
print(
    f"Brier: "
    f"{brier_score_loss(y_val[actual_run], m0[actual_run]):.4f}"
)

print("\nPASS")
print(f"Plays: {actual_pass.sum()}")
print(
    f"Log loss: "
    f"{log_loss(y_val[actual_pass], m1[actual_pass]):.4f}"
)
print(
    f"Brier: "
    f"{brier_score_loss(y_val[actual_pass], m1[actual_pass]):.4f}"
)


# ---------------------------------------------------------
# Common support
# ---------------------------------------------------------

support = (
    (e >= 0.10)
    &
    (e <= 0.90)
)


print("\nCOMMON SUPPORT")
print(
    f"Supported: {support.sum()} / {len(support)}"
)
print(
    f"Fraction: {support.mean():.4f}"
)


# ---------------------------------------------------------
# Doubly robust pseudo-outcome
#
# positive = PASS better
# negative = RUN better
# ---------------------------------------------------------

e_safe = np.clip(
    e,
    0.10,
    0.90,
)

tau = (
    m1
    - m0
    + t_val * (y_val - m1) / e_safe
    - (1 - t_val) * (y_val - m0) / (1 - e_safe)
)


result = validation.copy()

result["propensity_pass"] = e
result["p_run"] = m0
result["p_pass"] = m1
result["tau_dr"] = tau
result["supported"] = support


supported = result[
    result["supported"]
].copy()


print("\nSUPPORTED ACTION COUNTS")

print(
    supported
    .groupby(["ydstogo", "action"])
    .size()
    .to_string()
)


# ---------------------------------------------------------
# Overall short-yardage effect
# ---------------------------------------------------------

effect = supported[
    "tau_dr"
].mean()


print("\nOVERALL DR EFFECT — 4TH AND 1-2")

print(
    f"PASS minus RUN: {effect:+.4f}"
)


# ---------------------------------------------------------
# Exact distance
# ---------------------------------------------------------

by_distance = (
    supported
    .groupby("ydstogo")
    .agg(
        plays=("tau_dr", "size"),
        p_run=("p_run", "mean"),
        p_pass=("p_pass", "mean"),
        dr_pass_minus_run=("tau_dr", "mean"),
    )
)


print("\nDR EFFECT BY DISTANCE")

print(
    by_distance.to_string()
)


# ---------------------------------------------------------
# Game-cluster bootstrap
# ---------------------------------------------------------

rng = np.random.default_rng(42)

games = supported[
    "game_id"
].unique()

effects = []


for _ in range(1000):

    sampled_games = rng.choice(
        games,
        size=len(games),
        replace=True,
    )

    sample = pd.concat(
        [
            supported[
                supported["game_id"] == game
            ]
            for game in sampled_games
        ],
        ignore_index=True,
    )

    effects.append(
        sample["tau_dr"].mean()
    )


lower = np.quantile(
    effects,
    0.025,
)

upper = np.quantile(
    effects,
    0.975,
)


print("\n95% CLUSTER BOOTSTRAP")

print(
    f"Effect: {effect:+.4f}"
)

print(
    f"95% interval: "
    f"[{lower:+.4f}, {upper:+.4f}]"
)

print("\n95% CLUSTER BOOTSTRAP BY DISTANCE")

rng_by_distance = np.random.default_rng(123)

for distance in [1.0, 2.0]:

    sub = supported[
        supported["ydstogo"] == distance
    ].copy()

    games = sub[
        "game_id"
    ].unique()

    distance_effects = []

    for _ in range(1000):

        sampled_games = rng_by_distance.choice(
            games,
            size=len(games),
            replace=True,
        )

        sample = pd.concat(
            [
                sub[
                    sub["game_id"] == game
                ]
                for game in sampled_games
            ],
            ignore_index=True,
        )

        distance_effects.append(
            sample["tau_dr"].mean()
        )

    effect_d = sub[
        "tau_dr"
    ].mean()

    lower_d = np.quantile(
        distance_effects,
        0.025,
    )

    upper_d = np.quantile(
        distance_effects,
        0.975,
    )

    print(
        f"4th-and-{int(distance)}: "
        f"{effect_d:+.4f} "
        f"[{lower_d:+.4f}, {upper_d:+.4f}]"
    )