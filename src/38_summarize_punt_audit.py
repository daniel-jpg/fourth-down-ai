import polars as pl


pl.Config.set_tbl_rows(100)
pl.Config.set_fmt_str_lengths(100)


df = pl.read_parquet(
    "data/punt_outcome_audit.parquet"
)


print(
    f"\nLoaded {df.height:,} punt decisions."
)


# ---------------------------------------------------------
# Basic sample.
# ---------------------------------------------------------

print("\nPUNT SAMPLE BY SPLIT")

print(
    df
    .group_by("split")
    .len()
    .sort("split")
)


print("\nEXECUTION STATUS BY SPLIT")

print(
    df
    .group_by([
        "split",
        "execution_status",
    ])
    .len()
    .sort([
        "split",
        "execution_status",
    ])
)


# ---------------------------------------------------------
# Raw play type.
# ---------------------------------------------------------

if "pbp_play_type" in df.columns:

    print("\nRAW PLAY TYPE BY EXECUTION")

    print(
        df
        .group_by([
            "execution_status",
            "pbp_play_type",
        ])
        .len()
        .sort([
            "execution_status",
            "pbp_play_type",
        ])
    )


# ---------------------------------------------------------
# Fields actually available.
# ---------------------------------------------------------

print("\nAVAILABLE PUNT-SPECIFIC FIELDS")

for column in df.columns:

    if column.startswith("pbp_"):

        print(
            f"{column}: "
            f"{df.height - df[column].null_count():,} available"
        )


# ---------------------------------------------------------
# Work only with normal executions for ordinary punt model.
# ---------------------------------------------------------

normal = df.filter(
    pl.col("execution_status")
    == "NORMAL"
)


print(
    f"\nNORMAL PUNTS: {normal.height:,}"
)


# ---------------------------------------------------------
# Binary flags.
# ---------------------------------------------------------

binary_fields = [
    "pbp_punt_blocked",
    "pbp_punt_inside_twenty",
    "pbp_punt_in_endzone",
    "pbp_punt_out_of_bounds",
    "pbp_punt_downed",
    "pbp_punt_fair_catch",
    "pbp_touchback",
    "pbp_fumble",
    "pbp_fumble_lost",
    "pbp_penalty",
]


print("\nNORMAL PUNT OUTCOME FLAGS")

rows = []

for column in binary_fields:

    if column not in normal.columns:
        continue

    for split in [
        "train",
        "validation",
    ]:

        x = normal.filter(
            pl.col("split") == split
        )

        count = (
            x[column]
            .fill_null(0)
            .sum()
        )

        rows.append({
            "split":
                split,

            "field":
                column,

            "count":
                int(count),

            "rate":
                float(
                    count / x.height
                ),
        })


print(
    pl.DataFrame(rows)
    .sort([
        "field",
        "split",
    ])
)


# ---------------------------------------------------------
# Special teams result.
# ---------------------------------------------------------

if "pbp_special_teams_result" in normal.columns:

    print("\nNORMAL SPECIAL TEAMS RESULT")

    print(
        normal
        .group_by([
            "split",
            "pbp_special_teams_result",
        ])
        .len()
        .sort([
            "split",
            "pbp_special_teams_result",
        ])
    )


# ---------------------------------------------------------
# Punt distance.
# ---------------------------------------------------------

print("\nNORMAL PUNT DISTANCE")

if "pbp_kick_distance" in normal.columns:

    print(
        normal
        .group_by("split")
        .agg([

            pl.col("pbp_kick_distance")
            .count()
            .alias("n"),

            pl.col("pbp_kick_distance")
            .mean()
            .alias("mean"),

            pl.col("pbp_kick_distance")
            .median()
            .alias("median"),

            pl.col("pbp_kick_distance")
            .quantile(0.10)
            .alias("p10"),

            pl.col("pbp_kick_distance")
            .quantile(0.90)
            .alias("p90"),

        ])
        .sort("split")
    )


# ---------------------------------------------------------
# Returns.
# ---------------------------------------------------------

print("\nNORMAL PUNT RETURN YARDS")

if "pbp_return_yards" in normal.columns:

    print(
        normal
        .group_by("split")
        .agg([

            pl.col("pbp_return_yards")
            .count()
            .alias("n"),

            pl.col("pbp_return_yards")
            .mean()
            .alias("mean"),

            pl.col("pbp_return_yards")
            .median()
            .alias("median"),

            pl.col("pbp_return_yards")
            .quantile(0.10)
            .alias("p10"),

            pl.col("pbp_return_yards")
            .quantile(0.90)
            .alias("p90"),

            pl.col("pbp_return_yards")
            .max()
            .alias("max"),

        ])
        .sort("split")
    )


# ---------------------------------------------------------
# Punt outcome class.
#
# Keep this simple for now.
# ---------------------------------------------------------

expr = None

if "pbp_punt_blocked" in normal.columns:

    expr = (
        pl.when(
            pl.col("pbp_punt_blocked")
            .fill_null(0)
            == 1
        )
        .then(pl.lit("BLOCKED"))
    )


def add_when(current, condition, label):

    if current is None:

        return (
            pl.when(condition)
            .then(pl.lit(label))
        )

    return (
        current
        .when(condition)
        .then(pl.lit(label))
    )


if "pbp_touchback" in normal.columns:

    expr = add_when(
        expr,
        pl.col("pbp_touchback")
        .fill_null(0)
        == 1,
        "TOUCHBACK",
    )


if "pbp_punt_out_of_bounds" in normal.columns:

    expr = add_when(
        expr,
        pl.col("pbp_punt_out_of_bounds")
        .fill_null(0)
        == 1,
        "OUT_OF_BOUNDS",
    )


if "pbp_punt_downed" in normal.columns:

    expr = add_when(
        expr,
        pl.col("pbp_punt_downed")
        .fill_null(0)
        == 1,
        "DOWNED",
    )


if "pbp_punt_fair_catch" in normal.columns:

    expr = add_when(
        expr,
        pl.col("pbp_punt_fair_catch")
        .fill_null(0)
        == 1,
        "FAIR_CATCH",
    )


if "pbp_return_yards" in normal.columns:

    expr = add_when(
        expr,
        pl.col("pbp_return_yards")
        .is_not_null(),
        "RETURN",
    )


if expr is not None:

    normal = normal.with_columns(

        expr
        .otherwise(
            pl.lit("OTHER")
        )
        .alias("punt_outcome_class")

    )


    print("\nNORMAL PUNT OUTCOME CLASSES")

    print(
        normal
        .group_by([
            "split",
            "punt_outcome_class",
        ])
        .len()
        .sort([
            "split",
            "punt_outcome_class",
        ])
    )


# ---------------------------------------------------------
# Starting field position matters a LOT for punting.
# Bucket yardline_100.
# ---------------------------------------------------------

normal = normal.with_columns(

    pl.when(
        pl.col("yardline_100") >= 80
    )
    .then(pl.lit("own_1_20"))

    .when(
        pl.col("yardline_100") >= 60
    )
    .then(pl.lit("own_21_40"))

    .when(
        pl.col("yardline_100") >= 40
    )
    .then(pl.lit("midfield"))

    .when(
        pl.col("yardline_100") >= 20
    )
    .then(pl.lit("opp_21_40"))

    .otherwise(
        pl.lit("opp_1_20")
    )

    .alias("field_bucket")

)


print("\nPUNT DISTANCE BY STARTING FIELD POSITION")

if "pbp_kick_distance" in normal.columns:

    print(
        normal
        .group_by([
            "split",
            "field_bucket",
        ])
        .agg([

            pl.len()
            .alias("plays"),

            pl.col("pbp_kick_distance")
            .mean()
            .alias("mean_kick"),

            pl.col("pbp_kick_distance")
            .median()
            .alias("median_kick"),

        ])
        .sort([
            "split",
            "field_bucket",
        ])
    )


print("\nDONE")