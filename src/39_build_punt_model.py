import json
from pathlib import Path

import joblib
import nflreadpy as nfl
import numpy as np
import polars as pl

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)


pl.Config.set_tbl_rows(100)
pl.Config.set_fmt_str_lengths(120)

Path("models").mkdir(
    exist_ok=True
)


# =========================================================
# 1. Load punt audit from Scripts 37-38.
# =========================================================

punts = (
    pl.read_parquet(
        "data/punt_outcome_audit.parquet"
    )
    .sort([
        "game_id",
        "play_id",
    ])
)

print(
    f"Loaded {punts.height:,} punt decisions."
)


# =========================================================
# 2. Load development PBP ONLY.
#
# 2025 remains untouched.
# =========================================================

print("\nLoading 2014-2024 PBP...")

pbp = (
    nfl.load_pbp(
        list(range(2014, 2025))
    )
    .sort([
        "game_id",
        "play_id",
    ])
)

print(
    f"Loaded {pbp.height:,} PBP rows."
)


# =========================================================
# 3. Find first subsequent offensive state.
#
# This gives us the actual post-punt field position,
# including returns, touchbacks, penalties, etc.
# =========================================================

state_rows = (
    pbp
    .filter(
        pl.col("posteam").is_not_null()
        &
        pl.col("down").is_not_null()
        &
        pl.col("yardline_100").is_not_null()
    )
    .select([
        "game_id",
        "play_id",

        "qtr",
        "game_seconds_remaining",

        "posteam",
        "defteam",

        "down",
        "ydstogo",
        "yardline_100",

        "posteam_score",
        "defteam_score",

        "play_type",
        "desc",
    ])
    .rename({

        "play_id":
            "state_play_id",

        "qtr":
            "state_qtr",

        "game_seconds_remaining":
            "state_game_seconds_remaining",

        "posteam":
            "state_posteam",

        "defteam":
            "state_defteam",

        "down":
            "state_down",

        "ydstogo":
            "state_ydstogo",

        "yardline_100":
            "state_yardline_100",

        "posteam_score":
            "state_posteam_score",

        "defteam_score":
            "state_defteam_score",

        "play_type":
            "state_play_type",

        "desc":
            "state_desc",
    })
    .sort([
        "game_id",
        "state_play_id",
    ])
)


punts = (
    punts
    .with_columns(
        (
            pl.col("play_id")
            + 0.000001
        )
        .alias("search_play_id")
    )
    .sort([
        "game_id",
        "search_play_id",
    ])
    .join_asof(
        state_rows,

        left_on="search_play_id",
        right_on="state_play_id",

        by="game_id",

        strategy="forward",
    )
)


# =========================================================
# 4. Verify future-state extraction.
# =========================================================

bad_future = punts.filter(
    pl.col("state_play_id").is_not_null()
    &
    (
        pl.col("state_play_id")
        <= pl.col("play_id")
    )
)


print("\nSTRICT FUTURE CHECK")

print(
    f"Bad future states: {bad_future.height}"
)

assert bad_future.height == 0


# =========================================================
# 5. Score differential from ORIGINAL kicking team's
# perspective.
# =========================================================

punts = punts.with_columns(

    pl.when(
        pl.col("state_posteam")
        ==
        pl.col("posteam")
    )
    .then(
        pl.col("state_posteam_score")
        -
        pl.col("state_defteam_score")
    )

    .when(
        pl.col("state_posteam")
        ==
        pl.col("defteam")
    )
    .then(
        pl.col("state_defteam_score")
        -
        pl.col("state_posteam_score")
    )

    .otherwise(None)

    .alias(
        "state_original_score_differential"
    )

)


punts = punts.with_columns([

    (
        pl.col(
            "state_original_score_differential"
        )
        -
        pl.col(
            "score_differential"
        )
    )
    .alias("score_change"),

    (
        pl.col(
            "game_seconds_remaining"
        )
        -
        pl.col(
            "state_game_seconds_remaining"
        )
    )
    .alias("seconds_to_state"),

])


# =========================================================
# 6. Period changes.
# =========================================================

punts = punts.with_columns([

    (
        (pl.col("qtr") == 2)
        &
        (pl.col("state_qtr") == 3)
    )
    .fill_null(False)
    .alias("crossed_halftime"),

    (
        (pl.col("qtr") == 4)
        &
        (pl.col("state_qtr") >= 5)
    )
    .fill_null(False)
    .alias("entered_overtime"),

])


# =========================================================
# 7. Physical punt branch.
#
# BROKEN:
#   bad snap / execution failure
#
# BLOCKED:
#   actual punt attempt gets blocked
#
# REGULAR:
#   everything else
# =========================================================

punts = punts.with_columns(

    pl.when(
        pl.col("execution_status")
        == "BROKEN"
    )
    .then(
        pl.lit("BROKEN")
    )

    .when(
        pl.col("pbp_punt_blocked")
        .fill_null(0)
        == 1
    )
    .then(
        pl.lit("BLOCKED")
    )

    .otherwise(
        pl.lit("REGULAR")
    )

    .alias("punt_branch")

)


# =========================================================
# 8. Final transition class.
#
# Score changes take priority over possession because a
# punt-return TD is followed by a kickoff and therefore
# possession may have flipped again by the next state.
# =========================================================

punts = punts.with_columns(

    pl.when(
        pl.col("score_change") < 0
    )
    .then(
        pl.lit("opponent_scored")
    )

    .when(
        pl.col("score_change") > 0
    )
    .then(
        pl.lit("kicking_team_scored")
    )

    .when(
        (
            pl.col("touchdown") == 1
        )
        &
        (
            pl.col("pbp_td_team")
            ==
            pl.col("defteam")
        )
    )
    .then(
        pl.lit("opponent_scored")
    )

    .when(
        (
            pl.col("touchdown") == 1
        )
        &
        (
            pl.col("pbp_td_team")
            ==
            pl.col("posteam")
        )
    )
    .then(
        pl.lit("kicking_team_scored")
    )

    .when(
        pl.col("state_play_id")
        .is_null()
    )
    .then(
        pl.lit("no_later_state")
    )

    .when(
        pl.col("crossed_halftime")
    )
    .then(
        pl.lit("crossed_halftime")
    )

    .when(
        pl.col("entered_overtime")
    )
    .then(
        pl.lit("entered_overtime")
    )

    .when(
        pl.col("state_posteam")
        ==
        pl.col("defteam")
    )
    .then(
        pl.lit(
            "opponent_ball_no_score"
        )
    )

    .when(
        pl.col("state_posteam")
        ==
        pl.col("posteam")
    )
    .then(
        pl.lit(
            "kicking_team_ball_no_score"
        )
    )

    .otherwise(
        pl.lit("other")
    )

    .alias("transition_class")

)


# =========================================================
# 9. Training branch rates.
# =========================================================

train = punts.filter(
    pl.col("split")
    == "train"
)

validation = punts.filter(
    pl.col("split")
    == "validation"
)


print("\nPUNT BRANCHES — TRAIN")

print(
    train
    .group_by("punt_branch")
    .len()
    .with_columns(
        (
            pl.col("len")
            /
            train.height
        )
        .alias("rate")
    )
    .sort(
        "len",
        descending=True,
    )
)


print("\nPUNT BRANCHES — VALIDATION")

print(
    validation
    .group_by("punt_branch")
    .len()
    .with_columns(
        (
            pl.col("len")
            /
            validation.height
        )
        .alias("rate")
    )
    .sort(
        "len",
        descending=True,
    )
)


# =========================================================
# 10. Regular-punt transition distribution.
# =========================================================

train_regular = train.filter(
    pl.col("punt_branch")
    == "REGULAR"
)


print("\nREGULAR PUNT TRANSITIONS — TRAIN")

print(
    train_regular
    .group_by(
        "transition_class"
    )
    .len()
    .with_columns(
        (
            pl.col("len")
            /
            train_regular.height
        )
        .alias("rate")
    )
    .sort(
        "len",
        descending=True,
    )
)


# =========================================================
# 11. Ordinary punt field-position model.
#
# We model the final receiving team's yardline DIRECTLY.
#
# This automatically incorporates:
# - raw punt distance
# - return yards
# - touchbacks
# - fair catches
# - downed punts
# - out of bounds
# - most penalty effects
# =========================================================

ordinary_train = (
    train
    .filter(
        (pl.col("punt_branch") == "REGULAR")
        &
        (
            pl.col("transition_class")
            ==
            "opponent_ball_no_score"
        )
    )
)


ordinary_val = (
    validation
    .filter(
        (pl.col("punt_branch") == "REGULAR")
        &
        (
            pl.col("transition_class")
            ==
            "opponent_ball_no_score"
        )
    )
)


print("\nORDINARY PUNT SAMPLE")

print(
    f"Train:      {ordinary_train.height:,}"
)

print(
    f"Validation: {ordinary_val.height:,}"
)


FEATURES = [

    "yardline_100",
    "ydstogo",

    "qtr",
    "game_seconds_remaining",

    "score_differential",

    "posteam_timeouts_remaining",
    "defteam_timeouts_remaining",

    "site_advantage",
]


X_train = (
    ordinary_train
    .select(FEATURES)
    .to_numpy()
)

y_train = (
    ordinary_train[
        "state_yardline_100"
    ]
    .to_numpy()
)


X_val = (
    ordinary_val
    .select(FEATURES)
    .to_numpy()
)

y_val = (
    ordinary_val[
        "state_yardline_100"
    ]
    .to_numpy()
)


# =========================================================
# 12. Out-of-fold predictions.
#
# These residuals become our empirical uncertainty pool.
# Using OOF predictions prevents an unrealistically narrow
# residual distribution.
# =========================================================

groups = (
    ordinary_train[
        "game_id"
    ]
    .to_numpy()
)


group_kfold = GroupKFold(
    n_splits=5
)


oof_pred = np.zeros(
    ordinary_train.height
)


print("\nBuilding OOF residual pool...")


for fold, (
    train_idx,
    holdout_idx,
) in enumerate(
    group_kfold.split(
        X_train,
        y_train,
        groups=groups,
    ),
    start=1,
):

    fold_model = (
        HistGradientBoostingRegressor(

            learning_rate=0.05,

            max_iter=300,

            max_leaf_nodes=31,

            min_samples_leaf=30,

            l2_regularization=2.0,

            random_state=42,
        )
    )

    fold_model.fit(
        X_train[train_idx],
        y_train[train_idx],
    )

    oof_pred[
        holdout_idx
    ] = fold_model.predict(
        X_train[
            holdout_idx
        ]
    )

    print(
        f"Fold {fold}/5 complete."
    )


# =========================================================
# 13. Final model.
# =========================================================

model = HistGradientBoostingRegressor(

    learning_rate=0.05,

    max_iter=300,

    max_leaf_nodes=31,

    min_samples_leaf=30,

    l2_regularization=2.0,

    random_state=42,
)


model.fit(
    X_train,
    y_train,
)


val_pred = model.predict(
    X_val
)


# Valid football field bounds.
val_pred = np.clip(
    val_pred,
    0,
    100,
)


val_mae = mean_absolute_error(
    y_val,
    val_pred,
)


val_rmse = np.sqrt(
    mean_squared_error(
        y_val,
        val_pred,
    )
)


print("\nORDINARY PUNT FIELD-POSITION MODEL")

print(
    f"Validation MAE:  {val_mae:.4f} yards"
)

print(
    f"Validation RMSE: {val_rmse:.4f} yards"
)

print(
    f"Actual mean:     {y_val.mean():.4f}"
)

print(
    f"Predicted mean:  {val_pred.mean():.4f}"
)


# =========================================================
# 14. Build empirical residual + clock pool.
#
# We sample these JOINTLY later.
# =========================================================

ordinary_train = (
    ordinary_train
    .with_columns([

        pl.Series(
            "oof_predicted_yardline",
            oof_pred,
        ),

        pl.Series(
            "yardline_residual",
            y_train - oof_pred,
        ),

    ])
    .with_columns(

        (
            (
                pl.col("yardline_100")
                / 10
            )
            .floor()
            * 10
        )
        .cast(pl.Int32)
        .alias(
            "start_yardline_bucket"
        )

    )
)


bad_clock = ordinary_train.filter(
    pl.col("seconds_to_state")
    .is_null()
    |
    (pl.col("seconds_to_state") < 0)
    |
    (pl.col("seconds_to_state") > 30)
)


print("\nORDINARY PUNT CLOCK")

print(
    ordinary_train
    .select([

        pl.col(
            "seconds_to_state"
        )
        .mean()
        .alias("mean"),

        pl.col(
            "seconds_to_state"
        )
        .median()
        .alias("median"),

        pl.col(
            "seconds_to_state"
        )
        .quantile(0.10)
        .alias("p10"),

        pl.col(
            "seconds_to_state"
        )
        .quantile(0.90)
        .alias("p90"),

        pl.col(
            "seconds_to_state"
        )
        .max()
        .alias("max"),

    ])
)


print(
    f"Clock rows outside 0-30 sec: "
    f"{bad_clock.height}"
)


residual_pool = (
    ordinary_train
    .filter(
        pl.col("seconds_to_state")
        .is_not_null()
        &
        (pl.col("seconds_to_state") >= 0)
        &
        (pl.col("seconds_to_state") <= 30)
    )
    .select([

        "game_id",
        "play_id",
        "season",

        "yardline_100",
        "start_yardline_bucket",

        "oof_predicted_yardline",
        "state_yardline_100",
        "yardline_residual",

        "seconds_to_state",

    ])
)


print(
    f"Residual pool rows: "
    f"{residual_pool.height:,}"
)


# =========================================================
# 15. Rare-transition pool.
#
# Contains:
# - bad snaps / broken punts
# - blocked punts
# - return TDs
# - muff recoveries
# - possession-retaining penalties
# - period-end cases
#
# These are too rare to deserve individual tiny models.
# =========================================================

rare_train = train.filter(

    ~(
        (pl.col("punt_branch") == "REGULAR")
        &
        (
            pl.col("transition_class")
            ==
            "opponent_ball_no_score"
        )
    )

)


rare_columns = [

    "game_id",
    "play_id",
    "season",

    "yardline_100",
    "game_seconds_remaining",
    "score_differential",

    "punt_branch",
    "transition_class",

    "score_change",
    "seconds_to_state",

    "state_posteam",
    "state_down",
    "state_ydstogo",
    "state_yardline_100",

    "pbp_punt_blocked",
    "pbp_fumble_lost",

    "touchdown",
    "pbp_td_team",

    "desc",
]


rare_train.select(
    rare_columns
).write_parquet(
    "data/punt_rare_transition_pool.parquet"
)


# =========================================================
# 16. Save ordinary residual pool and model.
# =========================================================

residual_pool.write_parquet(
    "data/punt_ordinary_transition_pool.parquet"
)


joblib.dump(
    model,
    "models/punt_field_position_model.joblib",
)


# =========================================================
# 17. Save model specification.
# =========================================================

train_broken = train.filter(
    pl.col("punt_branch")
    == "BROKEN"
).height


train_normal = train.filter(
    pl.col("execution_status")
    == "NORMAL"
)


train_blocked = train_normal.filter(
    pl.col("pbp_punt_blocked")
    .fill_null(0)
    == 1
).height


regular_train = train.filter(
    pl.col("punt_branch")
    == "REGULAR"
)


regular_rates = {

    row["transition_class"]:
        row["len"]
        /
        regular_train.height

    for row in (
        regular_train
        .group_by(
            "transition_class"
        )
        .len()
        .iter_rows(
            named=True
        )
    )

}


spec = {

    "training_seasons":
        "2014-2022",

    "validation_seasons":
        "2023-2024",

    "test_season":
        2025,

    "broken_probability":
        train_broken
        /
        train.height,

    "blocked_probability_given_normal_execution":
        train_blocked
        /
        train_normal.height,

    "regular_transition_probabilities":
        regular_rates,

    "ordinary_field_position_model":
        "models/punt_field_position_model.joblib",

    "ordinary_transition_pool":
        "data/punt_ordinary_transition_pool.parquet",

    "rare_transition_pool":
        "data/punt_rare_transition_pool.parquet",

    "features":
        FEATURES,

    "ordinary_transition_policy":
        (
            "predict opponent yardline, then sample "
            "OOF yardline residual and clock cost jointly "
            "from similar starting-field-position bucket"
        ),

    "rare_transition_policy":
        "empirical pooled sampling",

}


with open(
    "data/punt_transition_spec.json",
    "w",
) as f:

    json.dump(
        spec,
        f,
        indent=2,
    )


print("\nSAVED")

print(
    "models/punt_field_position_model.joblib"
)

print(
    "data/punt_ordinary_transition_pool.parquet"
)

print(
    "data/punt_rare_transition_pool.parquet"
)

print(
    "data/punt_transition_spec.json"
)
