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
# 3. Attach exact SAME-PLAY punt metadata.
#
# Do not infer score / possession from a later play.
#
# play_id is not reliably chronological across every game,
# so the punt play itself is the source of truth for:
#
# - TD / safety scoring
# - lost fumbles
# - multi-fumble recovery sequence
# =========================================================

punt_meta = (
    pbp
    .filter(
        pl.col("punt_attempt")
        .fill_null(0)
        ==
        1
    )
    .select([
        "game_id",
        "play_id",

        "fixed_drive",

        "posteam_score",
        "defteam_score",
        "posteam_score_post",
        "defteam_score_post",

        "fumble_lost",

        "fumbled_1_team",
        "fumble_recovery_1_team",

        "fumbled_2_team",
        "fumble_recovery_2_team",

        "touchdown",
        "td_team",
        "safety",
    ])
    .rename({
        "fixed_drive":
            "pbp_fixed_drive",

        "posteam_score":
            "same_play_posteam_score_before",

        "defteam_score":
            "same_play_defteam_score_before",

        "posteam_score_post":
            "same_play_posteam_score_after",

        "defteam_score_post":
            "same_play_defteam_score_after",

        "fumble_lost":
            "same_play_fumble_lost",

        "fumbled_1_team":
            "same_play_fumbled_1_team",

        "fumble_recovery_1_team":
            "same_play_fumble_recovery_1_team",

        "fumbled_2_team":
            "same_play_fumbled_2_team",

        "fumble_recovery_2_team":
            "same_play_fumble_recovery_2_team",

        "touchdown":
            "same_play_touchdown",

        "td_team":
            "same_play_td_team",

        "safety":
            "same_play_safety",
    })
)


before_meta_join = punts.height

punts = punts.join(
    punt_meta,
    on=[
        "game_id",
        "play_id",
    ],
    how="left",
)

assert (
    punts.height
    ==
    before_meta_join
)


# =========================================================
# 4. Reconstruct next offensive state by FIXED DRIVE.
#
# fixed_drive + 1 is far safer than assuming a larger
# play_id means a later football state.
#
# We still validate the resulting state before allowing it
# into the field-position / clock model.
# =========================================================

drive_states = (
    pbp
    .filter(
        pl.col("fixed_drive")
        .is_not_null()
        &
        pl.col("posteam")
        .is_not_null()
        &
        pl.col("down")
        .is_not_null()
        &
        pl.col("yardline_100")
        .is_not_null()
    )
    .select([
        "game_id",
        "fixed_drive",

        "play_id",
        "qtr",
        "game_seconds_remaining",

        "posteam",
        "defteam",

        "down",
        "ydstogo",
        "yardline_100",

        "play_type",
        "desc",
    ])
    .sort(
        [
            "game_id",
            "fixed_drive",
            "qtr",
            "game_seconds_remaining",
            "play_id",
        ],
        descending=[
            False,
            False,
            False,
            True,
            False,
        ],
    )
    .group_by(
        [
            "game_id",
            "fixed_drive",
        ],
        maintain_order=True,
    )
    .agg([
        pl.col("play_id")
        .first()
        .alias("state_play_id"),

        pl.col("qtr")
        .first()
        .alias("state_qtr"),

        pl.col(
            "game_seconds_remaining"
        )
        .first()
        .alias(
            "state_game_seconds_remaining"
        ),

        pl.col("posteam")
        .first()
        .alias("state_posteam"),

        pl.col("defteam")
        .first()
        .alias("state_defteam"),

        pl.col("down")
        .first()
        .alias("state_down"),

        pl.col("ydstogo")
        .first()
        .alias("state_ydstogo"),

        pl.col("yardline_100")
        .first()
        .alias("state_yardline_100"),

        pl.col("play_type")
        .first()
        .alias("state_play_type"),

        pl.col("desc")
        .first()
        .alias("state_desc"),
    ])
    .rename({
        "fixed_drive":
            "target_fixed_drive"
    })
)


punts = (
    punts
    .with_columns(
        (
            pl.col("pbp_fixed_drive")
            + 1.0
        )
        .alias(
            "target_fixed_drive"
        )
    )
    .join(
        drive_states,
        on=[
            "game_id",
            "target_fixed_drive",
        ],
        how="left",
    )
)


# =========================================================
# 5. Same-play score and final loose-ball recovery.
# =========================================================

punts = punts.with_columns([

    (
        (
            pl.col(
                "same_play_posteam_score_after"
            )
            -
            pl.col(
                "same_play_defteam_score_after"
            )
        )
        -
        (
            pl.col(
                "same_play_posteam_score_before"
            )
            -
            pl.col(
                "same_play_defteam_score_before"
            )
        )
    )
    .alias(
        "same_play_score_change"
    ),

    pl.when(
        pl.col(
            "same_play_fumble_lost"
        )
        .fill_null(0)
        ==
        1
    )
    .then(
        pl.coalesce([
            pl.col(
                "same_play_fumble_recovery_2_team"
            ),

            # A second fumble that goes out of bounds stays
            # with the second fumbler.
            pl.col(
                "same_play_fumbled_2_team"
            ),

            pl.col(
                "same_play_fumble_recovery_1_team"
            ),
        ])
    )
    .otherwise(
        pl.lit(
            None,
            dtype=pl.String,
        )
    )
    .alias(
        "final_recovery_team"
    ),

    (
        pl.col(
            "game_seconds_remaining"
        )
        -
        pl.col(
            "state_game_seconds_remaining"
        )
    )
    .alias(
        "seconds_to_state"
    ),

])


# score_change now means score ON THE PUNT PLAY itself.
punts = punts.with_columns(
    pl.col(
        "same_play_score_change"
    )
    .alias(
        "score_change"
    )
)


# =========================================================
# 6. Period / state validation.
# =========================================================

punts = punts.with_columns([

    (
        (pl.col("qtr") == 2)
        &
        (pl.col("state_qtr") == 3)
    )
    .fill_null(False)
    .alias(
        "crossed_halftime"
    ),

    (
        (pl.col("qtr") == 4)
        &
        (pl.col("state_qtr") >= 5)
    )
    .fill_null(False)
    .alias(
        "entered_overtime"
    ),

    (
        (
            pl.col("state_qtr")
            ==
            pl.col("qtr")
        )
        |
        (
            (pl.col("qtr") == 1)
            &
            (pl.col("state_qtr") == 2)
        )
        |
        (
            (pl.col("qtr") == 3)
            &
            (pl.col("state_qtr") == 4)
        )
    )
    .fill_null(False)
    .alias(
        "valid_period_transition"
    ),

])


punts = punts.with_columns(
    (
        pl.col("state_play_id")
        .is_not_null()
        &
        pl.col(
            "valid_period_transition"
        )
        &
        pl.col(
            "seconds_to_state"
        )
        .is_not_null()
        &
        (
            pl.col(
                "seconds_to_state"
            )
            >=
            0
        )
        &
        (
            pl.col(
                "seconds_to_state"
            )
            <=
            30
        )
    )
    .fill_null(False)
    .alias(
        "state_chronological"
    )
)


# =========================================================
# 7. Physical punt branch.
# =========================================================

punts = punts.with_columns(

    pl.when(
        pl.col("execution_status")
        ==
        "BROKEN"
    )
    .then(
        pl.lit("BROKEN")
    )

    .when(
        pl.col("pbp_punt_blocked")
        .fill_null(0)
        ==
        1
    )
    .then(
        pl.lit("BLOCKED")
    )

    .otherwise(
        pl.lit("REGULAR")
    )

    .alias(
        "punt_branch"
    )

)


# =========================================================
# 8. Transition class.
#
# Scoring comes from the punt play itself.
#
# For an ordinary punt:
# - absent a lost return-team fumble, the receiving team
#   gets the ball;
# - if the return team loses the ball, use the FINAL
#   recovery on the play.
#
# Blocked / broken punts may not populate normal fumble
# fields, so only use the next-drive state when that state
# passes the chronology guard.
# =========================================================

punts = punts.with_columns(

    pl.when(
        pl.col("score_change")
        <
        0
    )
    .then(
        pl.lit(
            "opponent_scored"
        )
    )

    .when(
        pl.col("score_change")
        >
        0
    )
    .then(
        pl.lit(
            "kicking_team_scored"
        )
    )

    .when(
        pl.col("state_play_id")
        .is_null()
    )
    .then(
        pl.lit(
            "no_later_state"
        )
    )

    .when(
        pl.col(
            "crossed_halftime"
        )
    )
    .then(
        pl.lit(
            "crossed_halftime"
        )
    )

    .when(
        pl.col(
            "entered_overtime"
        )
    )
    .then(
        pl.lit(
            "entered_overtime"
        )
    )

    # Ordinary punt where receiving team lost the ball
    # and the kicking team finished with it.
    .when(
        (
            pl.col("punt_branch")
            ==
            "REGULAR"
        )
        &
        (
            pl.col(
                "same_play_fumble_lost"
            )
            .fill_null(0)
            ==
            1
        )
        &
        (
            pl.col(
                "final_recovery_team"
            )
            ==
            pl.col("posteam")
        )
    )
    .then(
        pl.lit(
            "kicking_team_ball_no_score"
        )
    )

    # All other non-scoring regular punts belong to the
    # receiving team.
    .when(
        pl.col("punt_branch")
        ==
        "REGULAR"
    )
    .then(
        pl.lit(
            "opponent_ball_no_score"
        )
    )

    # Blocked / broken punt: trust next-drive possession
    # only if the state passed the chronology guard.
    .when(
        pl.col(
            "state_chronological"
        )
        &
        (
            pl.col("state_posteam")
            ==
            pl.col("defteam")
        )
    )
    .then(
        pl.lit(
            "opponent_ball_no_score"
        )
    )

    .when(
        pl.col(
            "state_chronological"
        )
        &
        (
            pl.col("state_posteam")
            ==
            pl.col("posteam")
        )
    )
    .then(
        pl.lit(
            "kicking_team_ball_no_score"
        )
    )

    .otherwise(
        pl.lit("other")
    )

    .alias(
        "transition_class"
    )

)


# =========================================================
# 8B. Is the reconstructed state safe to use?
# =========================================================

punts = punts.with_columns(

    pl.when(
        pl.col(
            "transition_class"
        )
        ==
        "opponent_ball_no_score"
    )
    .then(
        pl.col(
            "state_chronological"
        )
        &
        (
            pl.col("state_posteam")
            ==
            pl.col("defteam")
        )
    )

    .when(
        pl.col(
            "transition_class"
        )
        ==
        "kicking_team_ball_no_score"
    )
    .then(
        pl.col(
            "state_chronological"
        )
        &
        (
            pl.col("state_posteam")
            ==
            pl.col("posteam")
        )
    )

    .when(
        pl.col(
            "transition_class"
        )
        .is_in([
            "opponent_scored",
            "kicking_team_scored",
        ])
    )
    .then(
        pl.col(
            "state_chronological"
        )
    )

    .otherwise(
        pl.lit(False)
    )

    .fill_null(False)
    .alias(
        "state_usable"
    )

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
        &
        pl.col("state_usable")
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
        &
        pl.col("state_usable")
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

        "pbp_touchback",

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
    "safety",

    "same_play_score_change",
    "final_recovery_team",
    "state_usable",

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
