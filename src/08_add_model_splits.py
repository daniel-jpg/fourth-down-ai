import polars as pl


pl.Config.set_tbl_rows(100)
pl.Config.set_fmt_str_lengths(120)


# ---------------------------------------------------------
# Load modeling dataset
# ---------------------------------------------------------

df = pl.read_parquet(
    "data/fourth_down_modeling.parquet"
)

print(f"Loaded {df.height:,} modeling plays.")


# ---------------------------------------------------------
# Time-based split
#
# TRAIN:      2014-2022
# VALIDATION: 2023-2024
# TEST:       2025
#
# We do NOT randomly split NFL plays.
# The test set should represent genuinely future football.
# ---------------------------------------------------------

df = df.with_columns(

    pl.when(
        pl.col("season") <= 2022
    )
    .then(pl.lit("train"))

    .when(
        pl.col("season").is_in([2023, 2024])
    )
    .then(pl.lit("validation"))

    .when(
        pl.col("season") == 2025
    )
    .then(pl.lit("test"))

    .otherwise(pl.lit("unknown"))

    .alias("split")
)


# ---------------------------------------------------------
# Make sure every season was assigned
# ---------------------------------------------------------

unknown_rows = df.filter(
    pl.col("split") == "unknown"
).height

print(f"Unknown split rows: {unknown_rows}")

assert unknown_rows == 0


# ---------------------------------------------------------
# Overall split sizes
# ---------------------------------------------------------

print("\nSPLIT COUNTS")

print(
    df
    .group_by("split")
    .len()
    .sort("split")
)


# ---------------------------------------------------------
# Make sure every action appears in each split
# ---------------------------------------------------------

print("\nACTION COUNTS BY SPLIT")

print(
    df
    .group_by(["split", "action"])
    .len()
    .sort(["split", "action"])
)


# ---------------------------------------------------------
# Season distribution
# ---------------------------------------------------------

print("\nSEASONS BY SPLIT")

print(
    df
    .group_by(["split", "season"])
    .len()
    .sort(["split", "season"])
)


# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

df.write_parquet(
    "data/fourth_down_modeling_split.parquet"
)

print("\nSaved:")
print("data/fourth_down_modeling_split.parquet")