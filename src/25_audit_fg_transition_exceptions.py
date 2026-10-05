import polars as pl


pl.Config.set_tbl_rows(100)
pl.Config.set_fmt_str_lengths(120)


# ---------------------------------------------------------
# We already built this in Script 24.
# No need to reload all 531k PBP rows.
# ---------------------------------------------------------

df = pl.read_parquet(
    "data/fg_post_state_audit.parquet"
)


df = df.with_columns([

    pl.when(
        pl.col("season") <= 2022
    )
    .then(pl.lit("train"))
    .otherwise(pl.lit("validation"))
    .alias("split"),

    (
        (pl.col("qtr") == 2)
        &
        (pl.col("next_qtr") == 3)
    )
    .fill_null(False)
    .alias("crossed_halftime"),

    (
        pl.min_horizontal(
            pl.lit(80.0),
            (
                pl.lit(92.0)
                - pl.col("yardline_100")
            ),
        )
    )
    .alias("miss_rule_yardline"),

])


# ---------------------------------------------------------
# MISSED FGs
# ---------------------------------------------------------

missed = (
    df
    .filter(
        (pl.col("fg_outcome") == "missed")
        &
        pl.col("next_state_found")
    )
    .with_columns(

        pl.when(
            pl.col("score_differential_change") != 0
        )
        .then(
            pl.lit("score_changed")
        )

        .when(
            pl.col("crossed_halftime")
        )
        .then(
            pl.lit("crossed_halftime")
        )

        .when(
            pl.col("offense_kept_ball")
        )
        .then(
            pl.lit("original_offense_next")
        )

        .when(
            pl.col("possession_flipped")
        )
        .then(
            pl.lit("standard_opponent_ball")
        )

        .otherwise(
            pl.lit("other")
        )

        .alias("transition_class")

    )
    .with_columns(
        (
            pl.col("next_yardline_100")
            -
            pl.col("miss_rule_yardline")
        )
        .alias("rule_error")
    )
)


print("\nMISSED FG TRANSITION CLASSES")

print(
    missed
    .group_by([
        "split",
        "transition_class",
    ])
    .len()
    .sort([
        "split",
        "transition_class",
    ])
)


# ---------------------------------------------------------
# Evaluate the NFL-rule baseline ONLY on the ordinary case:
#
# no score
# no halftime transition
# opponent gets the ball
# ---------------------------------------------------------

standard_miss = missed.filter(
    pl.col("transition_class")
    == "standard_opponent_ball"
)


print("\nSTANDARD MISSED-FG RULE PERFORMANCE")

print(
    standard_miss
    .group_by("split")
    .agg([

        pl.len()
        .alias("misses"),

        (
            pl.col("rule_error")
            .abs()
            < 0.5
        )
        .mean()
        .alias("exact_match_rate"),

        pl.col("rule_error")
        .abs()
        .mean()
        .alias("mean_absolute_error"),

        pl.col("rule_error")
        .mean()
        .alias("mean_error"),

        pl.col("rule_error")
        .abs()
        .quantile(0.90)
        .alias("p90_absolute_error"),

        pl.col("rule_error")
        .abs()
        .quantile(0.95)
        .alias("p95_absolute_error"),

    ])
    .sort("split")
)


# ---------------------------------------------------------
# Are there still strange rule deviations once we remove
# halftime / scoring / possession exceptions?
# ---------------------------------------------------------

print("\nLARGEST STANDARD MISSED-FG ERRORS")

print(
    standard_miss
    .filter(
        pl.col("rule_error").abs() >= 3
    )
    .select([
        "season",
        "week",
        "game_id",
        "play_id",

        "qtr",
        "game_seconds_remaining",
        "next_qtr",
        "next_game_seconds_remaining",

        "yardline_100",
        "miss_rule_yardline",
        "next_yardline_100",
        "rule_error",

        "seconds_to_next_state",

        "desc",
        "next_desc",
    ])
    .sort(
        pl.col("rule_error")
        .abs(),
        descending=True,
    )
    .head(30)
)


# ---------------------------------------------------------
# Summarize the nonstandard missed-FG cases.
# ---------------------------------------------------------

print("\nNONSTANDARD MISSED-FG TRANSITIONS")

print(
    missed
    .filter(
        pl.col("transition_class")
        != "standard_opponent_ball"
    )
    .group_by(
        "transition_class"
    )
    .agg([

        pl.len()
        .alias("plays"),

        pl.col("next_yardline_100")
        .mean()
        .alias("mean_next_yardline"),

        pl.col("seconds_to_next_state")
        .mean()
        .alias("mean_seconds"),

        pl.col("score_differential_change")
        .mean()
        .alias("mean_score_change"),

    ])
    .sort(
        "plays",
        descending=True,
    )
)


# ---------------------------------------------------------
# LIVE-BALL FAILURES
#
# Blocks + broken executions where opponent gets the ball
# without a score.
#
# Compare their field position with the ordinary missed-FG
# rule baseline.
# ---------------------------------------------------------

live_standard = (
    df
    .filter(
        pl.col("fg_outcome").is_in([
            "blocked",
            "broken",
        ])
        &
        pl.col("next_state_found")
        &
        (pl.col("score_differential_change") == 0)
        &
        pl.col("possession_flipped")
        &
        ~pl.col("crossed_halftime")
    )
    .with_columns(
        (
            pl.col("next_yardline_100")
            -
            pl.col("miss_rule_yardline")
        )
        .alias("rule_error")
    )
)


print("\nLIVE-BALL FAILURE FIELD POSITION")

print(
    live_standard
    .group_by("fg_outcome")
    .agg([

        pl.len()
        .alias("plays"),

        pl.col("yardline_100")
        .mean()
        .alias("mean_original_yardline"),

        pl.col("next_yardline_100")
        .mean()
        .alias("mean_next_yardline"),

        pl.col("next_yardline_100")
        .median()
        .alias("median_next_yardline"),

        pl.col("rule_error")
        .mean()
        .alias("mean_rule_error"),

        pl.col("rule_error")
        .abs()
        .mean()
        .alias("mean_absolute_rule_error"),

        pl.col("rule_error")
        .quantile(0.10)
        .alias("error_p10"),

        pl.col("rule_error")
        .median()
        .alias("error_median"),

        pl.col("rule_error")
        .quantile(0.90)
        .alias("error_p90"),

    ])
    .sort("fg_outcome")
)


# ---------------------------------------------------------
# Quarter-transition audit
# ---------------------------------------------------------

print("\nFG QUARTER TRANSITIONS")

print(
    df
    .filter(
        pl.col("next_state_found")
    )
    .group_by([
        "fg_outcome",
        "qtr",
        "next_qtr",
    ])
    .len()
    .sort([
        "fg_outcome",
        "qtr",
        "next_qtr",
    ])
)
