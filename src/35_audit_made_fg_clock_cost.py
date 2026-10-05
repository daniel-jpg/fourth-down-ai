import polars as pl


pl.Config.set_tbl_rows(100)
pl.Config.set_fmt_str_lengths(140)


df = pl.read_parquet(
    "data/fg_made_causal_kickoff_audit.parquet"
)


# ---------------------------------------------------------
# Clean made FGs for which the generated kickoff was
# actually observed before the next offensive state.
#
# These give us the post-FG game clock directly:
#
# FG start clock - kickoff start clock
#
# The game clock does NOT run between the score and kickoff,
# so this is the clock consumed by the FG play itself.
# ---------------------------------------------------------

clean = (
    df
    .filter(
        pl.col("causal_path")
        == "kickoff_before_state"
    )
    .with_columns([

        (
            pl.col("game_seconds_remaining")
            -
            pl.col(
                "kickoff_game_seconds_remaining"
            )
        )
        .alias("fg_clock_cost"),

        pl.when(
            pl.col("season") <= 2022
        )
        .then(pl.lit("train"))
        .otherwise(
            pl.lit("validation")
        )
        .alias("split"),

        (
            pl.col("yardline_100")
            + 18
        )
        .alias(
            "fg_distance_estimate"
        ),

    ])
)


# ---------------------------------------------------------
# Sanity checks.
# ---------------------------------------------------------

print(
    "\nCLEAN MADE-FG CLOCK SAMPLE"
)

print(
    clean
    .group_by("split")
    .len()
    .sort("split")
)


bad = clean.filter(
    pl.col("fg_clock_cost").is_null()
    |
    (pl.col("fg_clock_cost") < 0)
)


print(
    "\nINVALID CLOCK COSTS"
)

print(
    f"Rows: {bad.height}"
)


# ---------------------------------------------------------
# Overall train / validation distribution.
# ---------------------------------------------------------

print(
    "\nFG CLOCK COST BY SPLIT"
)

print(
    clean
    .group_by("split")
    .agg([

        pl.len()
        .alias("plays"),

        pl.col("fg_clock_cost")
        .mean()
        .alias("mean"),

        pl.col("fg_clock_cost")
        .median()
        .alias("median"),

        pl.col("fg_clock_cost")
        .quantile(0.10)
        .alias("p10"),

        pl.col("fg_clock_cost")
        .quantile(0.25)
        .alias("p25"),

        pl.col("fg_clock_cost")
        .quantile(0.75)
        .alias("p75"),

        pl.col("fg_clock_cost")
        .quantile(0.90)
        .alias("p90"),

        pl.col("fg_clock_cost")
        .quantile(0.95)
        .alias("p95"),

        pl.col("fg_clock_cost")
        .max()
        .alias("max"),

    ])
    .sort("split")
)


# ---------------------------------------------------------
# Era stability.
# ---------------------------------------------------------

print(
    "\nFG CLOCK COST BY ERA"
)

print(
    clean
    .group_by("era")
    .agg([

        pl.len()
        .alias("plays"),

        pl.col("fg_clock_cost")
        .mean()
        .alias("mean"),

        pl.col("fg_clock_cost")
        .median()
        .alias("median"),

        pl.col("fg_clock_cost")
        .quantile(0.10)
        .alias("p10"),

        pl.col("fg_clock_cost")
        .quantile(0.90)
        .alias("p90"),

        pl.col("fg_clock_cost")
        .quantile(0.95)
        .alias("p95"),

    ])
    .sort("era")
)


# ---------------------------------------------------------
# Does kick distance affect clock cost enough to justify
# conditioning the transition on distance?
# ---------------------------------------------------------

clean = clean.with_columns(

    pl.when(
        pl.col("fg_distance_estimate") <= 39
    )
    .then(pl.lit("<=39"))

    .when(
        pl.col("fg_distance_estimate") <= 49
    )
    .then(pl.lit("40-49"))

    .when(
        pl.col("fg_distance_estimate") <= 59
    )
    .then(pl.lit("50-59"))

    .otherwise(
        pl.lit("60+")
    )

    .alias("distance_bucket")

)


print(
    "\nFG CLOCK COST BY DISTANCE"
)

print(
    clean
    .group_by([
        "split",
        "distance_bucket",
    ])
    .agg([

        pl.len()
        .alias("plays"),

        pl.col("fg_clock_cost")
        .mean()
        .alias("mean"),

        pl.col("fg_clock_cost")
        .median()
        .alias("median"),

        pl.col("fg_clock_cost")
        .quantile(0.90)
        .alias("p90"),

    ])
    .sort([
        "split",
        "distance_bucket",
    ])
)


print(
    "\nTRAIN DISTANCE / CLOCK CORRELATION"
)

print(
    clean
    .filter(
        pl.col("split")
        == "train"
    )
    .select(
        pl.corr(
            "fg_distance_estimate",
            "fg_clock_cost",
        )
        .alias("correlation")
    )
)


# ---------------------------------------------------------
# Exact empirical clock-cost frequencies.
#
# If the distribution is stable, this can simply become
# our training sampler.
# ---------------------------------------------------------

print(
    "\nTRAIN FG CLOCK COST FREQUENCIES"
)

print(
    clean
    .filter(
        pl.col("split")
        == "train"
    )
    .group_by(
        "fg_clock_cost"
    )
    .len()
    .sort(
        "fg_clock_cost"
    )
)


# ---------------------------------------------------------
# Inspect large values. These may be data oddities,
# penalties, or period-boundary cases.
# ---------------------------------------------------------

print(
    "\nLARGE FG CLOCK COSTS"
)

print(
    clean
    .filter(
        pl.col("fg_clock_cost") > 12
    )
    .select([
        "season",
        "week",
        "game_id",
        "play_id",

        "qtr",
        "game_seconds_remaining",

        "fg_distance_estimate",
        "fg_clock_cost",

        "kickoff_play_id",
        "kickoff_qtr",
        "kickoff_game_seconds_remaining",

        "desc",
        "kickoff_desc",
    ])
    .sort(
        "fg_clock_cost",
        descending=True,
    )
    .head(30)
)


# ---------------------------------------------------------
# Save TRAINING clock samples only.
#
# We will freeze this after reviewing the audit.
# ---------------------------------------------------------

(
    clean
    .filter(
        pl.col("split")
        == "train"
    )
    .select([
        "game_id",
        "play_id",
        "season",
        "fg_distance_estimate",
        "fg_clock_cost",
    ])
    .write_parquet(
        "data/fg_made_clock_training_pool.parquet"
    )
)


print(
    "\nSaved:"
)

print(
    "data/fg_made_clock_training_pool.parquet"
)