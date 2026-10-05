import polars as pl


pl.Config.set_tbl_rows(100)
pl.Config.set_fmt_str_lengths(140)


df = pl.read_parquet(
    "data/fg_failed_immediate_state_audit.parquet"
)


# ---------------------------------------------------------
# Blocked + broken FG live-ball outcomes.
# ---------------------------------------------------------

live = (
    df
    .filter(
        pl.col("fg_outcome").is_in([
            "blocked",
            "broken",
        ])
    )
    .with_columns(

        pl.when(
            pl.col("season") <= 2022
        )
        .then(pl.lit("train"))
        .otherwise(pl.lit("validation"))
        .alias("split")

    )
)


# ---------------------------------------------------------
# Transition class.
# ---------------------------------------------------------

live = live.with_columns(

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
        pl.lit("original_offense_scored")
    )

    .when(
        pl.col("state_posteam")
        == pl.col("defteam")
    )
    .then(
        pl.lit("opponent_ball_no_score")
    )

    .when(
        pl.col("state_posteam")
        == pl.col("posteam")
    )
    .then(
        pl.lit("original_ball_no_score")
    )

    .otherwise(
        pl.lit("other")
    )

    .alias("transition_class")

)


# ---------------------------------------------------------
# Training blocked-FG class probabilities.
# ---------------------------------------------------------

train_blocks = live.filter(
    (pl.col("split") == "train")
    &
    (pl.col("fg_outcome") == "blocked")
)


print(
    "\nTRAIN BLOCKED TRANSITION MODEL"
)

print(
    train_blocks
    .group_by(
        "transition_class"
    )
    .agg(
        pl.len()
        .alias("plays")
    )
    .with_columns(
        (
            pl.col("plays")
            /
            pl.col("plays").sum()
        )
        .alias("rate")
    )
    .sort(
        "plays",
        descending=True,
    )
)


# ---------------------------------------------------------
# Opponent scoring after blocked / broken FG.
#
# We need to know whether these are essentially all
# defensive TDs and what the resulting score changes are.
# ---------------------------------------------------------

print(
    "\nOPPONENT-SCORED SCORE CHANGES"
)

print(
    live
    .filter(
        pl.col("transition_class")
        == "opponent_scored"
    )
    .group_by([
        "split",
        "fg_outcome",
        "score_change",
    ])
    .len()
    .sort([
        "split",
        "fg_outcome",
        "score_change",
    ])
)


print(
    "\nOPPONENT-SCORED DETAILS"
)

print(
    live
    .filter(
        pl.col("transition_class")
        == "opponent_scored"
    )
    .select([
    "split",
    "season",
    "week",
    "game_id",
    "play_id",

    "fg_outcome",

    "qtr",
    "game_seconds_remaining",

    "yardline_100",

    "seconds_to_state",

        "state_posteam",
        "state_down",
        "state_ydstogo",
        "state_yardline_100",

        "desc",
        "state_desc",
    ])
    .sort([
        "split",
        "season",
        "week",
    ])
)


# ---------------------------------------------------------
# Kicking team gets the ball again without a score.
#
# Very rare, but potentially valuable if it represents
# recovery + first down rather than a data artifact.
# ---------------------------------------------------------

print(
    "\nORIGINAL-OFFENSE BALL, NO SCORE"
)

print(
    live
    .filter(
        pl.col("transition_class")
        == "original_ball_no_score"
    )
    .select([
        "split",
        "season",
        "week",
        "game_id",
        "play_id",

        "fg_outcome",

        "qtr",
        "game_seconds_remaining",

        "yardline_100",

        "seconds_to_state",

        "state_posteam",
        "state_down",
        "state_ydstogo",
        "state_yardline_100",

        "desc",
        "state_desc",
    ])
    .sort([
        "split",
        "season",
        "week",
    ])
)


# ---------------------------------------------------------
# Any unclassified case.
# ---------------------------------------------------------

print(
    "\nOTHER LIVE-BALL CASES"
)

print(
    live
    .filter(
        pl.col("transition_class")
        == "other"
    )
    .select([
        "split",
        "season",
        "week",
        "game_id",
        "play_id",

        "fg_outcome",

        "qtr",
        "game_seconds_remaining",

        "score_change",
        "seconds_to_state",

        "state_posteam",
        "state_down",
        "state_ydstogo",
        "state_yardline_100",

        "desc",
        "state_desc",
    ])
)


# ---------------------------------------------------------
# Halftime-crossing blocked/broken kicks.
#
# These should not be treated as ordinary field-position
# transitions.
# ---------------------------------------------------------

print(
    "\nLIVE FAILURES CROSSING HALFTIME"
)

print(
    live
    .filter(
        pl.col("crossed_halftime")
    )
    .select([
        "split",
        "season",
        "week",
        "game_id",
        "play_id",

        "fg_outcome",

        "qtr",
        "game_seconds_remaining",

        "score_change",
        "seconds_to_state",

        "state_qtr",
        "state_game_seconds_remaining",
        "state_posteam",
        "state_yardline_100",

        "desc",
        "state_desc",
    ])
    .sort([
        "season",
        "week",
    ])
)


# ---------------------------------------------------------
# Ordinary training block pool.
#
# This is the pool we are likely to sample JOINTLY from:
#
#   yardline residual
#   elapsed time
#
# rather than fitting separate models.
# ---------------------------------------------------------

ordinary_train = (
    train_blocks
    .filter(
        (pl.col("transition_class")
            == "opponent_ball_no_score")
        &
        ~pl.col("crossed_halftime")
    )
    .with_columns(
        (
            pl.col("state_yardline_100")
            -
            pl.col("miss_rule_yardline")
        )
        .alias("yardline_residual")
    )
)


print(
    "\nORDINARY TRAIN BLOCK POOL"
)

print(
    ordinary_train.select([

        pl.len()
        .alias("plays"),

        pl.col("yardline_residual")
        .mean()
        .alias("mean_residual"),

        pl.col("yardline_residual")
        .median()
        .alias("median_residual"),

        pl.col("seconds_to_state")
        .mean()
        .alias("mean_seconds"),

        pl.col("seconds_to_state")
        .median()
        .alias("median_seconds"),

    ])
)


# Save the exact training rows so the final transition
# code can reuse them without recomputing the audit.

ordinary_train.select([
    "game_id",
    "play_id",
    "season",

    "yardline_100",
    "miss_rule_yardline",
    "yardline_residual",

    "seconds_to_state",

    "state_yardline_100",
]).write_parquet(
    "data/fg_block_ordinary_transition_pool.parquet"
)


print(
    "\nSaved:"
)

print(
    "data/fg_block_ordinary_transition_pool.parquet"
)