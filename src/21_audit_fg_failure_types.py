import polars as pl


pl.Config.set_tbl_rows(100)


# ---------------------------------------------------------
# Load field-goal decisions
# ---------------------------------------------------------

df = pl.read_parquet(
    "data/fourth_downs_labeled.parquet"
)


fg = (
    df
    .filter(
        (pl.col("action") == "FIELD_GOAL")
        &
        (pl.col("season") <= 2024)
    )
    .with_columns([

        (
            pl.col("yardline_100") + 18
        ).alias("fg_distance_estimate"),

        pl.when(
            pl.col("execution_status") == "BROKEN"
        )
        .then(pl.lit("broken"))

        .otherwise(
            pl.col("field_goal_result")
            .fill_null("unknown")
            .str.to_lowercase()
        )
        .alias("fg_outcome"),

        pl.when(
            pl.col("season") <= 2022
        )
        .then(pl.lit("train"))

        .otherwise(
            pl.lit("validation")
        )
        .alias("split"),
    ])
)


# ---------------------------------------------------------
# Overall outcomes
# ---------------------------------------------------------

print("\nFG OUTCOMES BY SPLIT")

print(
    fg
    .group_by([
        "split",
        "fg_outcome",
    ])
    .len()
    .sort([
        "split",
        "fg_outcome",
    ])
)


# ---------------------------------------------------------
# Broken execution rate
# ---------------------------------------------------------

print("\nBROKEN RATE BY SPLIT")

print(
    fg
    .group_by("split")
    .agg([
        pl.len().alias("attempts"),

        (
            pl.col("execution_status")
            == "BROKEN"
        )
        .sum()
        .alias("broken"),

        (
            pl.col("execution_status")
            == "BROKEN"
        )
        .mean()
        .alias("broken_rate"),
    ])
)


print("\nBROKEN PLAYS BY SEASON")

print(
    fg
    .group_by("season")
    .agg([
        pl.len().alias("attempts"),

        (
            pl.col("execution_status")
            == "BROKEN"
        )
        .sum()
        .alias("broken"),

        (
            pl.col("execution_status")
            == "BROKEN"
        )
        .mean()
        .alias("broken_rate"),
    ])
    .sort("season")
)


# ---------------------------------------------------------
# Clean failed kicks only:
#
# missed versus blocked
# ---------------------------------------------------------

failures = (
    fg
    .filter(
        (pl.col("execution_status") == "NORMAL")
        &
        pl.col("fg_outcome").is_in([
            "missed",
            "blocked",
        ])
    )
)


print("\nCLEAN FAILURE COUNTS")

print(
    failures
    .group_by([
        "split",
        "fg_outcome",
    ])
    .len()
    .sort([
        "split",
        "fg_outcome",
    ])
)


# ---------------------------------------------------------
# Block rate among clean failures
# ---------------------------------------------------------

print("\nBLOCK RATE AMONG CLEAN FAILURES")

print(
    failures
    .group_by("split")
    .agg([
        pl.len().alias("failed_kicks"),

        (
            pl.col("fg_outcome")
            == "blocked"
        )
        .sum()
        .alias("blocked"),

        (
            pl.col("fg_outcome")
            == "blocked"
        )
        .mean()
        .alias("block_rate"),
    ])
)


# ---------------------------------------------------------
# Distance buckets
# ---------------------------------------------------------

failures = failures.with_columns(

    pl.when(
        pl.col("fg_distance_estimate") <= 29
    )
    .then(pl.lit("<=29"))

    .when(
        pl.col("fg_distance_estimate") <= 34
    )
    .then(pl.lit("30-34"))

    .when(
        pl.col("fg_distance_estimate") <= 39
    )
    .then(pl.lit("35-39"))

    .when(
        pl.col("fg_distance_estimate") <= 44
    )
    .then(pl.lit("40-44"))

    .when(
        pl.col("fg_distance_estimate") <= 49
    )
    .then(pl.lit("45-49"))

    .when(
        pl.col("fg_distance_estimate") <= 54
    )
    .then(pl.lit("50-54"))

    .when(
        pl.col("fg_distance_estimate") <= 59
    )
    .then(pl.lit("55-59"))

    .otherwise(
        pl.lit("60+")
    )
    .alias("distance_bucket")
)


print("\nBLOCK RATE BY DISTANCE")

print(
    failures
    .group_by([
        "split",
        "distance_bucket",
    ])
    .agg([
        pl.len().alias("failures"),

        (
            pl.col("fg_outcome")
            == "blocked"
        )
        .sum()
        .alias("blocked"),

        (
            pl.col("fg_outcome")
            == "blocked"
        )
        .mean()
        .alias("block_rate"),

        pl.col("fg_distance_estimate")
        .mean()
        .alias("avg_distance"),
    ])
    .sort([
        "split",
        "avg_distance",
    ])
)


# ---------------------------------------------------------
# Blocked vs missed distance summary
# ---------------------------------------------------------

print("\nFAILURE DISTANCE SUMMARY")

print(
    failures
    .group_by([
        "split",
        "fg_outcome",
    ])
    .agg([
        pl.len().alias("attempts"),

        pl.col("fg_distance_estimate")
        .mean()
        .alias("mean_distance"),

        pl.col("fg_distance_estimate")
        .median()
        .alias("median_distance"),
    ])
    .sort([
        "split",
        "fg_outcome",
    ])
)