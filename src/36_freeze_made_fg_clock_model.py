import json

import polars as pl


pl.Config.set_tbl_rows(100)


# ---------------------------------------------------------
# Training-only clock observations produced by Script 35.
# ---------------------------------------------------------

df = pl.read_parquet(
    "data/fg_made_clock_training_pool.parquet"
)


print(
    "\nRAW MADE-FG CLOCK POOL"
)

print(
    f"Rows: {df.height:,}"
)


# ---------------------------------------------------------
# Distance buckets.
#
# 60+ is far too sparse to stand alone, so combine all
# 50+ kicks.
# ---------------------------------------------------------

df = df.with_columns(

    pl.when(
        pl.col("fg_distance_estimate") <= 39
    )
    .then(
        pl.lit("<=39")
    )

    .when(
        pl.col("fg_distance_estimate") <= 49
    )
    .then(
        pl.lit("40-49")
    )

    .otherwise(
        pl.lit("50+")
    )

    .alias("distance_bucket")

)


# ---------------------------------------------------------
# Remove implausible / administrative clock artifacts.
#
# Validation had max = 6 sec.
# Training is overwhelmingly <= 6 sec.
#
# We use 8 sec as a conservative ceiling rather than
# trimming tightly to the validation set.
# ---------------------------------------------------------

clean = df.filter(
    pl.col("fg_clock_cost") <= 8
)


dropped = (
    df.height
    -
    clean.height
)


print(
    "\nCLOCK CLEANING"
)

print(
    f"Raw rows:     {df.height:,}"
)

print(
    f"Kept rows:    {clean.height:,}"
)

print(
    f"Dropped rows: {dropped:,}"
)

print(
    f"Dropped rate: {dropped / df.height:.6f}"
)


assert clean[
    "fg_clock_cost"
].max() <= 8


# ---------------------------------------------------------
# Final bucket summaries.
# ---------------------------------------------------------

print(
    "\nFINAL MADE-FG CLOCK MODEL BY DISTANCE"
)

print(
    clean
    .group_by(
        "distance_bucket"
    )
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
        .max()
        .alias("max"),

    ])
    .sort("distance_bucket")
)


# ---------------------------------------------------------
# Exact empirical distribution within each bucket.
#
# Later simulation can sample one observed clock cost from
# the appropriate bucket.
# ---------------------------------------------------------

print(
    "\nFINAL CLOCK FREQUENCIES"
)

print(
    clean
    .group_by([
        "distance_bucket",
        "fg_clock_cost",
    ])
    .len()
    .sort([
        "distance_bucket",
        "fg_clock_cost",
    ])
)


# ---------------------------------------------------------
# Save final empirical pool.
# ---------------------------------------------------------

clean.select([
    "game_id",
    "play_id",
    "season",
    "fg_distance_estimate",
    "distance_bucket",
    "fg_clock_cost",
]).write_parquet(
    "data/fg_made_clock_transition_pool.parquet"
)


# ---------------------------------------------------------
# Save final FG transition specification.
# ---------------------------------------------------------

spec = {

    "made": {

        "score_change":
            3,

        "clock_model":
            "empirical_by_distance_bucket",

        "clock_training_seasons":
            "2014-2022",

        "clock_max_training_seconds":
            8,

        "distance_buckets": [
            "<=39",
            "40-49",
            "50+",
        ],

        "next_state":
            "KICKOFF_PENDING",

    },

    "missed": {

        "score_change":
            0,

        "possession":
            "opponent",

        "down":
            1,

        "ydstogo":
            10,

        "yardline_rule":
            "min(80, 92 - original_yardline_100)",

        "clock_model":
            "borrow_made_fg_clock_model",

    },

    "blocked": {

        "transition_model":
            "data/fg_live_transition_model.json",

        "ordinary_transition_pool":
            "data/fg_block_transition_pool.parquet",

    },

    "broken": {

        "policy":
            "borrow_blocked_fg_transition_model",

    },

    "period_end": {

        "policy":
            "handled_by_terminal_or_halftime_state_logic",

    },

}


with open(
    "data/fg_transition_spec.json",
    "w",
) as f:

    json.dump(
        spec,
        f,
        indent=2,
    )


print(
    "\nSAVED"
)

print(
    "data/fg_made_clock_transition_pool.parquet"
)

print(
    "data/fg_transition_spec.json"
)