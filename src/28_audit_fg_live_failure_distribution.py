import polars as pl


pl.Config.set_tbl_rows(100)


# ---------------------------------------------------------
# Load the cleaned immediate-state audit from Script 27.
# ---------------------------------------------------------

df = pl.read_parquet(
    "data/fg_failed_immediate_state_audit.parquet"
)


# ---------------------------------------------------------
# Development split.
#
# 2025 is still completely untouched.
# ---------------------------------------------------------

df = df.with_columns(

    pl.when(
        pl.col("season") <= 2022
    )
    .then(pl.lit("train"))
    .otherwise(pl.lit("validation"))
    .alias("split")

)


# ---------------------------------------------------------
# Live-ball failures only.
# ---------------------------------------------------------

live = df.filter(
    pl.col("fg_outcome").is_in([
        "blocked",
        "broken",
    ])
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

    .alias(
        "transition_class"
    )

)


# ---------------------------------------------------------
# Estimated kick distance.
# ---------------------------------------------------------

live = live.with_columns(
    (
        pl.col("yardline_100")
        + 18
    )
    .alias(
        "fg_distance_estimate"
    )
)


live = live.with_columns(

    pl.when(
        pl.col("fg_distance_estimate")
        <= 39
    )
    .then(pl.lit("<=39"))

    .when(
        pl.col("fg_distance_estimate")
        <= 49
    )
    .then(pl.lit("40-49"))

    .when(
        pl.col("fg_distance_estimate")
        <= 59
    )
    .then(pl.lit("50-59"))

    .otherwise(
        pl.lit("60+")
    )

    .alias(
        "distance_bucket"
    )

)


# ---------------------------------------------------------
# 1. Transition-class stability.
# ---------------------------------------------------------

print(
    "\nLIVE FAILURE TRANSITION CLASSES BY SPLIT"
)

print(
    live
    .group_by([
        "split",
        "fg_outcome",
        "transition_class",
    ])
    .len()
    .sort([
        "split",
        "fg_outcome",
        "transition_class",
    ])
)


print(
    "\nBLOCKED TRANSITION RATES BY SPLIT"
)

print(
    live
    .filter(
        pl.col("fg_outcome")
        == "blocked"
    )
    .group_by("split")
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
    .sort("split")
)


# ---------------------------------------------------------
# Ordinary scoreless opponent-possession live failures.
# ---------------------------------------------------------

ordinary = live.filter(

    (pl.col("transition_class")
        == "opponent_ball_no_score")

    &
    ~pl.col("crossed_halftime")

)


# ---------------------------------------------------------
# Residual relative to normal missed-FG placement.
#
# Negative residual =
# opponent got BETTER field position, usually because of
# a return / loose-ball advance.
# ---------------------------------------------------------

ordinary = ordinary.with_columns(
    (
        pl.col("state_yardline_100")
        -
        pl.col("miss_rule_yardline")
    )
    .alias("yardline_residual")
)


# ---------------------------------------------------------
# Overall residual distribution.
# ---------------------------------------------------------

print(
    "\nLIVE FAILURE RESIDUAL DISTRIBUTION BY SPLIT"
)

print(
    ordinary
    .group_by([
        "split",
        "fg_outcome",
    ])
    .agg([

        pl.len()
        .alias("plays"),

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

    ])
    .sort([
        "split",
        "fg_outcome",
    ])
)


# ---------------------------------------------------------
# Does kick distance materially change the residual?
# ---------------------------------------------------------

print(
    "\nBLOCKED RESIDUAL BY DISTANCE"
)

print(
    ordinary
    .filter(
        pl.col("fg_outcome")
        == "blocked"
    )
    .group_by([
        "split",
        "distance_bucket",
    ])
    .agg([

        pl.len()
        .alias("plays"),

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
        .quantile(0.90)
        .alias("p90"),

        (
            pl.col("yardline_residual")
            <= -5
        )
        .mean()
        .alias(
            "return_5plus_rate"
        ),

    ])
    .sort([
        "split",
        "distance_bucket",
    ])
)


# ---------------------------------------------------------
# Does original field position correlate with residual?
# ---------------------------------------------------------

print(
    "\nBLOCKED RESIDUAL CORRELATION"
)

for split in [
    "train",
    "validation",
]:

    temp = ordinary.filter(
        (pl.col("fg_outcome") == "blocked")
        &
        (pl.col("split") == split)
    )

    print(
        split,
        temp.select(
            pl.corr(
                "yardline_100",
                "yardline_residual",
            )
            .alias("correlation")
        )
    )


# ---------------------------------------------------------
# Clock consumption.
#
# We need this later because WP depends on time remaining.
# ---------------------------------------------------------

print(
    "\nLIVE FAILURE CLOCK CONSUMPTION"
)

print(
    ordinary
    .group_by([
        "split",
        "fg_outcome",
    ])
    .agg([

        pl.len()
        .alias("plays"),

        pl.col("seconds_to_state")
        .mean()
        .alias("mean_seconds"),

        pl.col("seconds_to_state")
        .median()
        .alias("median_seconds"),

        pl.col("seconds_to_state")
        .quantile(0.10)
        .alias("p10_seconds"),

        pl.col("seconds_to_state")
        .quantile(0.90)
        .alias("p90_seconds"),

    ])
    .sort([
        "split",
        "fg_outcome",
    ])
)


# ---------------------------------------------------------
# Training residual frequencies.
#
# If distance conditioning is unnecessary, this empirical
# distribution can directly become our transition model.
# ---------------------------------------------------------

print(
    "\nTRAIN BLOCKED RESIDUAL FREQUENCIES"
)

print(
    ordinary
    .filter(
        (pl.col("split") == "train")
        &
        (pl.col("fg_outcome") == "blocked")
    )
    .group_by(
        "yardline_residual"
    )
    .len()
    .sort(
        "yardline_residual"
    )
)