import polars as pl


pl.Config.set_tbl_rows(100)


df = pl.read_parquet(
    "data/fg_failed_immediate_state_audit.parquet"
)


# ---------------------------------------------------------
# Blocked FGs only.
#
# Broken executions will eventually borrow the block
# transition distribution because there are only 9 train
# + 1 validation examples.
# ---------------------------------------------------------

blocks = (
    df
    .filter(
        pl.col("fg_outcome")
        == "blocked"
    )
    .with_columns([

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

        .alias(
            "transition_class"
        ),

        pl.when(
            pl.col("season") <= 2018
        )
        .then(
            pl.lit("2014-2018")
        )

        .when(
            pl.col("season") <= 2022
        )
        .then(
            pl.lit("2019-2022")
        )

        .otherwise(
            pl.lit("2023-2024")
        )

        .alias(
            "era"
        ),

    ])
)


# ---------------------------------------------------------
# Residual versus the deterministic missed-FG spot.
# ---------------------------------------------------------

blocks = blocks.with_columns(

    (
        pl.col("state_yardline_100")
        -
        pl.col("miss_rule_yardline")
    )
    .alias(
        "yardline_residual"
    )

)


# ---------------------------------------------------------
# Transition classes by era.
# ---------------------------------------------------------

print(
    "\nBLOCKED TRANSITION CLASSES BY ERA"
)

print(
    blocks
    .group_by([
        "era",
        "transition_class",
    ])
    .len()
    .sort([
        "era",
        "transition_class",
    ])
)


print(
    "\nBLOCKED TRANSITION RATES BY ERA"
)

print(
    blocks
    .group_by("era")
    .agg([

        pl.len()
        .alias("blocks"),

        (
            pl.col("transition_class")
            == "opponent_ball_no_score"
        )
        .mean()
        .alias(
            "opponent_ball_no_score_rate"
        ),

        (
            pl.col("transition_class")
            == "opponent_scored"
        )
        .mean()
        .alias(
            "opponent_score_rate"
        ),

        (
            pl.col("transition_class")
            == "original_ball_no_score"
        )
        .mean()
        .alias(
            "original_ball_rate"
        ),

    ])
    .sort("era")
)


# ---------------------------------------------------------
# Ordinary blocks:
#
# opponent gets ball
# no score
# same half
# ---------------------------------------------------------

ordinary = blocks.filter(

    (
        pl.col("transition_class")
        == "opponent_ball_no_score"
    )
    &
    ~pl.col("crossed_halftime")

)


print(
    "\nORDINARY BLOCK RESIDUAL BY ERA"
)

print(
    ordinary
    .group_by("era")
    .agg([

        pl.len()
        .alias("blocks"),

        pl.col("yardline_residual")
        .mean()
        .alias("mean"),

        pl.col("yardline_residual")
        .median()
        .alias("median"),

        pl.col("yardline_residual")
        .quantile(0.10)
        .alias("p10"),

        pl.col("yardline_residual")
        .quantile(0.25)
        .alias("p25"),

        pl.col("yardline_residual")
        .quantile(0.75)
        .alias("p75"),

        pl.col("yardline_residual")
        .quantile(0.90)
        .alias("p90"),

        (
            pl.col("yardline_residual")
            .abs()
            <= 1
        )
        .mean()
        .alias(
            "within_1_rate"
        ),

        (
            pl.col("yardline_residual")
            <= -5
        )
        .mean()
        .alias(
            "return_5plus_rate"
        ),

        (
            pl.col("yardline_residual")
            <= -15
        )
        .mean()
        .alias(
            "return_15plus_rate"
        ),

        pl.col("seconds_to_state")
        .mean()
        .alias(
            "mean_seconds"
        ),

        pl.col("seconds_to_state")
        .median()
        .alias(
            "median_seconds"
        ),

    ])
    .sort("era")
)


# ---------------------------------------------------------
# Season-by-season diagnostic.
#
# We are looking for a real trend, not one weird era bucket.
# ---------------------------------------------------------

print(
    "\nORDINARY BLOCK RESIDUAL BY SEASON"
)

print(
    ordinary
    .group_by("season")
    .agg([

        pl.len()
        .alias("blocks"),

        pl.col("yardline_residual")
        .mean()
        .alias("mean_residual"),

        pl.col("yardline_residual")
        .median()
        .alias("median_residual"),

        (
            pl.col("yardline_residual")
            <= -5
        )
        .mean()
        .alias(
            "return_5plus_rate"
        ),

        (
            pl.col("yardline_residual")
            <= -15
        )
        .mean()
        .alias(
            "return_15plus_rate"
        ),

        pl.col("seconds_to_state")
        .median()
        .alias(
            "median_seconds"
        ),

    ])
    .sort("season")
)


# ---------------------------------------------------------
# Compare FULL TRAINING versus RECENT TRAINING.
#
# If recent training resembles validation more closely in
# a stable way, we may prefer 2019-2022 as the empirical
# sampling pool.
# ---------------------------------------------------------

comparison = ordinary.with_columns(

    pl.when(
        pl.col("season") <= 2022
    )
    .then(
        pl.lit("all_train")
    )
    .otherwise(
        pl.lit("validation")
    )
    .alias(
        "sample_group"
    )

)


recent_train = (
    ordinary
    .filter(
        pl.col("season")
        .is_between(
            2019,
            2022,
        )
    )
    .with_columns(
        pl.lit(
            "recent_train"
        )
        .alias(
            "sample_group"
        )
    )
)


comparison = pl.concat([
    comparison,
    recent_train,
])


print(
    "\nFULL VS RECENT TRAIN COMPARISON"
)

print(
    comparison
    .group_by(
        "sample_group"
    )
    .agg([

        pl.len()
        .alias("blocks"),

        pl.col("yardline_residual")
        .mean()
        .alias("mean_residual"),

        pl.col("yardline_residual")
        .median()
        .alias("median_residual"),

        (
            pl.col("yardline_residual")
            <= -5
        )
        .mean()
        .alias(
            "return_5plus_rate"
        ),

        (
            pl.col("yardline_residual")
            <= -15
        )
        .mean()
        .alias(
            "return_15plus_rate"
        ),

        pl.col("seconds_to_state")
        .median()
        .alias(
            "median_seconds"
        ),

    ])
    .sort(
        "sample_group"
    )
)
