import json
from pathlib import Path

import joblib
import nflreadpy as nfl
import numpy as np
import polars as pl

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    log_loss,
    brier_score_loss,
    roc_auc_score,
)


pl.Config.set_tbl_rows(100)
pl.Config.set_fmt_str_lengths(120)

Path("models").mkdir(
    exist_ok=True
)


# =========================================================
# 1. Load modeling data.
#
# We train ONE pooled fourth-down conversion model using:
#
# - normal designed runs
# - normal dropbacks
# - fake punt runs/passes
# - fake FG runs/passes
#
# Fake indicators get regularized toward the much larger
# normal-go sample.
# =========================================================

df = pl.read_parquet(
    "data/fourth_down_modeling_split.parquet"
)


GO_ACTIONS = [

    "NORMAL_GO_RUN",
    "NORMAL_GO_PASS",

    "FAKE_PUNT_RUN",
    "FAKE_PUNT_PASS",

    "FAKE_FG_RUN",
    "FAKE_FG_PASS",
]


go = (
    df
    .filter(
        pl.col("action")
        .is_in(GO_ACTIONS)
    )
)


print(
    f"Loaded {go.height:,} normal/fake go plays."
)


# =========================================================
# 2. Build pooled-action indicators.
#
# IMPORTANT:
#
# execution_status is NOT a predictor.
#
# Broken execution is part of the observed outcome risk of
# calling the fake, so it stays in the target data but is
# never revealed to the model beforehand.
# =========================================================

go = go.with_columns([

    pl.col("action")
    .is_in([
        "NORMAL_GO_PASS",
        "FAKE_PUNT_PASS",
        "FAKE_FG_PASS",
    ])
    .cast(pl.Int8)
    .alias("is_pass"),

    pl.col("action")
    .str.starts_with(
        "FAKE_PUNT"
    )
    .cast(pl.Int8)
    .alias("fake_punt"),

    pl.col("action")
    .str.starts_with(
        "FAKE_FG"
    )
    .cast(pl.Int8)
    .alias("fake_fg"),

])


go = go.with_columns(

    (
        (pl.col("fake_punt") == 1)
        |
        (pl.col("fake_fg") == 1)
    )
    .cast(pl.Int8)
    .alias("is_fake")

)


# =========================================================
# 3. Samples.
#
# 2025 test stays completely untouched.
# =========================================================

train = go.filter(
    pl.col("split")
    == "train"
)

validation = go.filter(
    pl.col("split")
    == "validation"
)


print("\nACTION COUNTS — TRAIN")

print(
    train
    .group_by("action")
    .len()
    .sort("action")
)


print("\nACTION COUNTS — VALIDATION")

print(
    validation
    .group_by("action")
    .len()
    .sort("action")
)


print("\nCONVERSION RATES — TRAIN")

print(
    train
    .group_by("action")
    .agg([

        pl.len()
        .alias("plays"),

        pl.col("converted")
        .mean()
        .alias("conversion_rate"),

    ])
    .sort("action")
)


# =========================================================
# 4. Pooled conversion model.
#
# Fixed regularization strength.
#
# We are NOT tuning C on the tiny fake validation sample.
# Stronger regularization intentionally shrinks fake
# adjustments toward normal fourth-down behavior.
# =========================================================

FEATURES = [

    "ydstogo",

    "goal_to_go",

    "qtr",
    "game_seconds_remaining",

    "score_differential",

    "posteam_timeouts_remaining",
    "defteam_timeouts_remaining",

    "short_yardage",
    "inside_10",
    "inside_20",
    "final_two_minutes",

    "is_pass",

    "fake_punt",
    "fake_fg",
]


X_train = (
    train
    .select(FEATURES)
    .to_numpy()
)

y_train = (
    train["converted"]
    .to_numpy()
)


X_val = (
    validation
    .select(FEATURES)
    .to_numpy()
)

y_val = (
    validation["converted"]
    .to_numpy()
)


model = Pipeline([

    (
        "scaler",
        StandardScaler(),
    ),

    (
        "logistic",
        LogisticRegression(

            C=0.25,

            max_iter=3000,

            random_state=42,
        ),
    ),

])


model.fit(
    X_train,
    y_train,
)


val_pred = model.predict_proba(
    X_val
)[:, 1]


# =========================================================
# 5. Overall validation.
# =========================================================

val_ll = log_loss(
    y_val,
    val_pred,
)

val_brier = brier_score_loss(
    y_val,
    val_pred,
)

val_auc = roc_auc_score(
    y_val,
    val_pred,
)


print("\nPOOLED GO MODEL — VALIDATION")

print(
    f"Rows:             {validation.height:,}"
)

print(
    f"Actual conversion:{y_val.mean():.4f}"
)

print(
    f"Mean prediction:  {val_pred.mean():.4f}"
)

print(
    f"Log loss:         {val_ll:.4f}"
)

print(
    f"Brier:            {val_brier:.4f}"
)

print(
    f"AUC:              {val_auc:.4f}"
)


# =========================================================
# 6. Fake-only validation.
#
# This is an AUDIT only.
#
# We do not tune anything from these tiny samples.
# =========================================================

validation = validation.with_columns(

    pl.Series(
        "conversion_prediction",
        val_pred,
    )

)


fake_val = validation.filter(
    pl.col("is_fake")
    == 1
)


print("\nFAKE-ONLY VALIDATION")

print(
    f"Rows:             {fake_val.height}"
)


if fake_val.height > 0:

    print(
        f"Actual conversion:"
        f"{fake_val['converted'].mean():.4f}"
    )

    print(
        f"Mean prediction:  "
        f"{fake_val['conversion_prediction'].mean():.4f}"
    )


    fake_y = (
        fake_val["converted"]
        .to_numpy()
    )

    fake_p = (
        fake_val[
            "conversion_prediction"
        ]
        .to_numpy()
    )


    if len(
        np.unique(fake_y)
    ) > 1:

        print(
            f"Fake log loss:    "
            f"{log_loss(fake_y, fake_p):.4f}"
        )

        print(
            f"Fake Brier:       "
            f"{brier_score_loss(fake_y, fake_p):.4f}"
        )

        print(
            f"Fake AUC:         "
            f"{roc_auc_score(fake_y, fake_p):.4f}"
        )


# =========================================================
# 7. Calibration by action.
#
# Tiny fake-FG validation groups may contain only 1 play.
# Do NOT interpret those individually.
# =========================================================

print("\nVALIDATION BY ACTION")

print(
    validation
    .group_by("action")
    .agg([

        pl.len()
        .alias("plays"),

        pl.col("converted")
        .mean()
        .alias(
            "actual_conversion"
        ),

        pl.col(
            "conversion_prediction"
        )
        .mean()
        .alias(
            "mean_prediction"
        ),

        pl.col("ydstogo")
        .mean()
        .alias(
            "avg_ydstogo"
        ),

    ])
    .sort("action")
)


# =========================================================
# 8. Save conversion model.
# =========================================================

joblib.dump(
    model,
    "models/fake_conversion_model.joblib",
)


# =========================================================
# 9. Build shared transition data.
#
# Conversion probability is the fake-specific part.
#
# Once the ball is snapped and we know whether the play
# succeeds/fails, we borrow transition behavior from
# ordinary runs/dropbacks because fake samples are tiny.
#
# We still retain actual fake rows inside this pool.
# =========================================================

print(
    "\nLoading 2014-2024 PBP for transition states..."
)


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
# 10. Raw fields for touchdowns / turnovers.
# =========================================================

raw_fields = [
    "game_id",
    "play_id",
]


for column in [
    "td_team",
    "fumble",
    "fumble_lost",
]:

    if column in pbp.columns:

        raw_fields.append(
            column
        )


raw = pbp.select(
    raw_fields
)


rename_map = {

    column:
        f"pbp_{column}"

    for column in raw_fields

    if column
    not in [
        "game_id",
        "play_id",
    ]

}


raw = raw.rename(
    rename_map
)


go = go.join(
    raw,

    on=[
        "game_id",
        "play_id",
    ],

    how="left",
)


# =========================================================
# 11. First subsequent offensive state.
#
# no_play rows are allowed because their pre-play state can
# be the immediate football state following the decision.
# =========================================================

state_rows = (
    pbp
    .filter(
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

    })
    .sort([
        "game_id",
        "state_play_id",
    ])
)


go = (
    go
    .with_columns(

        (
            pl.col("play_id")
            + 0.000001
        )
        .alias(
            "search_play_id"
        )

    )
    .sort([
        "game_id",
        "search_play_id",
    ])
    .join_asof(

        state_rows,

        left_on=
            "search_play_id",

        right_on=
            "state_play_id",

        by="game_id",

        strategy="forward",
    )
)


# =========================================================
# 12. Join sanity check.
# =========================================================

bad_future = go.filter(

    pl.col("state_play_id")
    .is_not_null()

    &

    (
        pl.col("state_play_id")
        <=
        pl.col("play_id")
    )

)


print("\nSTRICT FUTURE CHECK")

print(
    f"Bad future states: "
    f"{bad_future.height}"
)


assert bad_future.height == 0


# =========================================================
# 13. Score differential from ORIGINAL offense's
# perspective.
# =========================================================

go = go.with_columns(

    pl.when(
        pl.col("state_posteam")
        ==
        pl.col("posteam")
    )
    .then(

        pl.col(
            "state_posteam_score"
        )
        -
        pl.col(
            "state_defteam_score"
        )

    )

    .when(
        pl.col("state_posteam")
        ==
        pl.col("defteam")
    )
    .then(

        pl.col(
            "state_defteam_score"
        )
        -
        pl.col(
            "state_posteam_score"
        )

    )

    .otherwise(None)

    .alias(
        "state_original_score_differential"
    )

)


go = go.with_columns([

    (
        pl.col(
            "state_original_score_differential"
        )
        -
        pl.col(
            "score_differential"
        )
    )
    .alias(
        "score_change"
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


# =========================================================
# 14. Period-transition flags.
# =========================================================

go = go.with_columns([

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

])


# =========================================================
# 15. Possession after the play.
# =========================================================

go = go.with_columns(

    pl.when(
        pl.col("state_posteam")
        ==
        pl.col("posteam")
    )
    .then(
        pl.lit("same_offense")
    )

    .when(
        pl.col("state_posteam")
        ==
        pl.col("defteam")
    )
    .then(
        pl.lit("opponent")
    )

    .otherwise(
        pl.lit("unknown")
    )

    .alias(
        "next_possession"
    )

)


# =========================================================
# 16. Transition class.
#
# Raw touchdown team gets first priority where available.
# =========================================================

offensive_td_condition = (
    (pl.col("touchdown") == 1)
)


defensive_td_condition = (
    (pl.col("touchdown") == 1)
)


if "pbp_td_team" in go.columns:

    offensive_td_condition = (

        (pl.col("touchdown") == 1)

        &

        (
            pl.col("pbp_td_team")
            ==
            pl.col("posteam")
        )

    )


    defensive_td_condition = (

        (pl.col("touchdown") == 1)

        &

        (
            pl.col("pbp_td_team")
            ==
            pl.col("defteam")
        )

    )


go = go.with_columns(

    pl.when(
        offensive_td_condition
    )
    .then(
        pl.lit(
            "offense_scored"
        )
    )

    .when(
        defensive_td_condition
    )
    .then(
        pl.lit(
            "opponent_scored"
        )
    )

    .when(
        pl.col("score_change") > 0
    )
    .then(
        pl.lit(
            "offense_scored"
        )
    )

    .when(
        pl.col("score_change") < 0
    )
    .then(
        pl.lit(
            "opponent_scored"
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

    .when(
        pl.col("next_possession")
        ==
        "same_offense"
    )
    .then(
        pl.lit(
            "same_offense_no_score"
        )
    )

    .when(
        pl.col("next_possession")
        ==
        "opponent"
    )
    .then(
        pl.lit(
            "opponent_ball_no_score"
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
# 17. Training transition pool.
#
# This is shared between ordinary go plays and fake plays.
# That is intentional shrinkage.
# =========================================================

transition_train = go.filter(
    pl.col("split")
    == "train"
)


print("\nTRAIN TRANSITIONS BY PLAY STYLE")

print(
    transition_train
    .group_by([
        "is_pass",
        "converted",
        "transition_class",
    ])
    .len()
    .sort([
        "is_pass",
        "converted",
        "transition_class",
    ])
)


# =========================================================
# 18. Fake-specific transition audit.
# =========================================================

print("\nFAKE TRAIN TRANSITIONS")

print(
    transition_train
    .filter(
        pl.col("is_fake")
        == 1
    )
    .group_by([
        "action",
        "transition_class",
    ])
    .len()
    .sort([
        "action",
        "transition_class",
    ])
)


# =========================================================
# 19. Shared transition pool.
#
# After conversion is sampled, borrow an observed
# run/dropback transition conditional on:
#
#   - run vs pass
#   - converted vs failed
#
# IMPORTANT:
# Keep scoring transitions too. Otherwise a successful
# fake could never generate a touchdown.
#
# We allow up to 60 seconds because successful plays can
# leave the same offense on the field with the game clock
# running before the next snap.
# =========================================================

shared_pool = (
    transition_train
    .filter(

        pl.col("state_play_id")
        .is_not_null()

        &

        ~pl.col("crossed_halftime")

        &

        ~pl.col("entered_overtime")

        &

        pl.col("seconds_to_state")
        .is_not_null()

        &

        (
            pl.col("seconds_to_state")
            >= 0
        )

        &

        (
            pl.col("seconds_to_state")
            <= 60
        )

        &

        pl.col("transition_class")
        .is_in([
            "same_offense_no_score",
            "opponent_ball_no_score",
            "offense_scored",
            "opponent_scored",
        ])

    )
)


pool_columns = [

    "game_id",
    "play_id",
    "season",

    "qtr",
    "game_seconds_remaining",

    "action",

    "is_pass",
    "is_fake",
    "fake_punt",
    "fake_fg",

    "execution_status",

    "converted",

    "yardline_100",
    "ydstogo",

    "yards_gained",

    "seconds_to_state",

    "transition_class",
    "next_possession",

    "score_change",

    "touchdown",
    "safety",

    "state_down",
    "state_ydstogo",
    "state_yardline_100",

]


for column in [
    "pbp_td_team",
    "pbp_fumble",
    "pbp_fumble_lost",
]:

    if column in shared_pool.columns:

        pool_columns.append(
            column
        )


shared_pool = shared_pool.select(
    pool_columns
)


print("\nSHARED TRANSITION POOL")

print(
    shared_pool
    .group_by([
        "is_pass",
        "converted",
        "transition_class",
    ])
    .len()
    .sort([
        "is_pass",
        "converted",
        "transition_class",
    ])
)


print(
    f"\nShared transition rows: "
    f"{shared_pool.height:,}"
)


shared_pool.write_parquet(
    "data/go_shared_transition_pool.parquet"
)

# =========================================================
# 20. Rare FAKE outcomes.
#
# Keep actual fake:
# - touchdowns
# - defensive scores
# - broken executions
# - weird possession outcomes
# - period-ending situations
#
# We don't fit tiny models to these.
# =========================================================

fake_train = transition_train.filter(
    pl.col("is_fake")
    == 1
)


fake_rare = fake_train.filter(

    (
        pl.col("execution_status")
        == "BROKEN"
    )

    |

    ~pl.col(
        "transition_class"
    )
    .is_in([
        "same_offense_no_score",
        "opponent_ball_no_score",
    ])

)


rare_columns = [

    "game_id",
    "play_id",
    "season",

    "action",

    "is_pass",
    "fake_punt",
    "fake_fg",

    "execution_status",

    "converted",

    "yardline_100",
    "ydstogo",

    "yards_gained",

    "touchdown",
    "safety",

    "transition_class",
    "score_change",

    "seconds_to_state",

    "next_possession",

    "state_down",
    "state_ydstogo",
    "state_yardline_100",

    "desc",
]


for column in [
    "pbp_td_team",
    "pbp_fumble",
    "pbp_fumble_lost",
]:

    if column in fake_rare.columns:

        rare_columns.append(
            column
        )


fake_rare.select(
    rare_columns
).write_parquet(
    "data/fake_rare_transition_pool.parquet"
)


print(
    f"\nFake rare transition rows: "
    f"{fake_rare.height}"
)


# =========================================================
# 21. Save specification.
# =========================================================

spec = {

    "training_seasons":
        "2014-2022",

    "validation_seasons":
        "2023-2024",

    "test_season":
        2025,

    "conversion_model":
        "models/fake_conversion_model.joblib",

    "conversion_features":
        FEATURES,

    "fake_effect_policy":
        (
            "pooled regularized logistic model; "
            "fake-punt and fake-FG indicators borrow "
            "strength from normal designed-run/dropback plays"
        ),

    "run_pass_policy":
        (
            "run and dropback share one model using is_pass"
        ),

    "broken_execution_policy":
        (
            "broken fake executions remain in conversion "
            "training outcomes but execution_status is never "
            "used as a pre-play predictor"
        ),

    "ordinary_transition_pool":
        "data/go_shared_transition_pool.parquet",

    "rare_fake_transition_pool":
        "data/fake_rare_transition_pool.parquet",

    "transition_policy":
        (
            "sample conversion first; then borrow run/dropback "
            "transition behavior conditional on play style and "
            "conversion outcome; use empirical fake rare pool "
            "for rare special-teams execution outcomes"
        ),

}


with open(
    "data/fake_transition_spec.json",
    "w",
) as f:

    json.dump(
        spec,
        f,
        indent=2,
    )


print("\nSAVED")

print(
    "models/fake_conversion_model.joblib"
)

print(
    "data/go_shared_transition_pool.parquet"
)

print(
    "data/fake_rare_transition_pool.parquet"
)

print(
    "data/fake_transition_spec.json"
)