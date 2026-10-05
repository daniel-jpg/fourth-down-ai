import polars as pl


pl.Config.set_tbl_rows(100)
pl.Config.set_fmt_str_lengths(120)


# ---------------------------------------------------------
# Load labeled fourth-down data
# ---------------------------------------------------------

df = pl.read_parquet(
    "data/fourth_downs_labeled.parquet"
)

fg = df.filter(
    pl.col("action") == "FIELD_GOAL"
)


print(
    f"Field-goal decisions: {fg.height:,}"
)


# ---------------------------------------------------------
# IMPORTANT:
# Do not inspect 2025 test data here.
#
# Train = 2014-2022
# Validation = 2023-2024
# ---------------------------------------------------------

normal_kicks = (
    fg
    .filter(
        (pl.col("execution_status") == "NORMAL")
        &
        pl.col("kick_distance").is_not_null()
    )
)


train = normal_kicks.filter(
    pl.col("season") <= 2022
)

validation = normal_kicks.filter(
    pl.col("season").is_between(
        2023,
        2024,
        closed="both",
    )
)


print("\nDISTANCE AUDIT SIZES")

print(
    f"Train:      {train.height:,}"
)

print(
    f"Validation: {validation.height:,}"
)


# ---------------------------------------------------------
# What offset does nflverse kick_distance imply?
#
# implied_offset =
# kick_distance - yardline_100
# ---------------------------------------------------------

def print_implied_offsets(data, label):

    offsets = (
        data
        .with_columns(
            (
                pl.col("kick_distance")
                - pl.col("yardline_100")
            )
            .alias("implied_offset")
        )
        .group_by(
            "implied_offset"
        )
        .len()
        .sort(
            "len",
            descending=True,
        )
    )

    print(
        f"\nIMPLIED DISTANCE OFFSETS — {label}"
    )

    print(
        offsets.head(20)
    )


print_implied_offsets(
    train,
    "TRAIN",
)

print_implied_offsets(
    validation,
    "VALIDATION",
)


# ---------------------------------------------------------
# Compare simple formulas:
#
# yardline_100 + 17
# yardline_100 + 18
# yardline_100 + 19
# yardline_100 + 20
#
# Choose based on TRAIN only.
# Validation is just a sanity check.
# ---------------------------------------------------------

def compare_offsets(data, label):

    rows = []

    for offset in [
        17,
        18,
        19,
        20,
    ]:

        temp = (
            data
            .with_columns(
                (
                    pl.col("yardline_100")
                    + offset
                )
                .alias("estimated_distance")
            )
            .with_columns(
                (
                    pl.col("kick_distance")
                    - pl.col("estimated_distance")
                )
                .alias("error")
            )
        )

        stats = (
            temp
            .select([
                pl.len()
                .alias("plays"),

                (
                    pl.col("error")
                    == 0
                )
                .mean()
                .alias("exact_match_rate"),

                pl.col("error")
                .abs()
                .mean()
                .alias("mean_absolute_error"),

                pl.col("error")
                .mean()
                .alias("mean_error"),

                pl.col("error")
                .median()
                .alias("median_error"),

                pl.col("error")
                .abs()
                .max()
                .alias("max_absolute_error"),
            ])
            .row(
                0,
                named=True,
            )
        )

        rows.append({
            "offset": offset,
            **stats,
        })


    result = pl.DataFrame(
        rows
    )


    print(
        f"\nOFFSET COMPARISON — {label}"
    )

    print(
        result
    )


compare_offsets(
    train,
    "TRAIN",
)

compare_offsets(
    validation,
    "VALIDATION",
)


# ---------------------------------------------------------
# Does the implied offset change over time?
# ---------------------------------------------------------

season_offset = (
    normal_kicks
    .filter(
        pl.col("season") <= 2024
    )
    .with_columns(
        (
            pl.col("kick_distance")
            - pl.col("yardline_100")
        )
        .alias("implied_offset")
    )
    .group_by("season")
    .agg([
        pl.len()
        .alias("kicks"),

        pl.col("implied_offset")
        .mean()
        .alias("mean_offset"),

        pl.col("implied_offset")
        .median()
        .alias("median_offset"),

        pl.col("implied_offset")
        .min()
        .alias("min_offset"),

        pl.col("implied_offset")
        .max()
        .alias("max_offset"),
    ])
    .sort("season")
)


print(
    "\nIMPLIED OFFSET BY SEASON"
)

print(
    season_offset
)


# ---------------------------------------------------------
# Weather missingness
#
# Again exclude 2025 test data.
# ---------------------------------------------------------

fg_pre_test = fg.filter(
    pl.col("season") <= 2024
)


weather_by_roof = (
    fg_pre_test
    .group_by("roof")
    .agg([
        pl.len()
        .alias("plays"),

        pl.col("temp")
        .is_null()
        .mean()
        .alias("temp_missing_rate"),

        pl.col("wind")
        .is_null()
        .mean()
        .alias("wind_missing_rate"),

        pl.col("weather")
        .is_null()
        .mean()
        .alias("weather_missing_rate"),
    ])
    .sort(
        "plays",
        descending=True,
    )
)


print(
    "\nWEATHER MISSINGNESS BY ROOF"
)

print(
    weather_by_roof
)


# ---------------------------------------------------------
# Outdoor weather coverage specifically
# ---------------------------------------------------------

outdoor = fg_pre_test.filter(
    pl.col("roof") == "outdoors"
)


print(
    "\nOUTDOOR WEATHER COVERAGE"
)

print(
    outdoor.select([
        pl.len()
        .alias("plays"),

        pl.col("temp")
        .null_count()
        .alias("temp_nulls"),

        pl.col("wind")
        .null_count()
        .alias("wind_nulls"),

        pl.col("weather")
        .null_count()
        .alias("weather_nulls"),
    ])
)


# ---------------------------------------------------------
# Kicker identity support
#
# How many validation kickers were never seen in training?
# ---------------------------------------------------------

train_kickers = set(
    train
    .get_column("kicker_player_id")
    .drop_nulls()
    .to_list()
)


validation_with_history = (
    validation
    .with_columns(
        pl.col("kicker_player_id")
        .is_in(
            list(train_kickers)
        )
        .alias("seen_in_train")
    )
)


print(
    "\nVALIDATION KICKER HISTORY"
)

print(
    validation_with_history
    .group_by("seen_in_train")
    .agg([
        pl.len()
        .alias("attempts"),

        pl.col("kicker_player_id")
        .n_unique()
        .alias("unique_kickers"),
    ])
    .sort("seen_in_train")
)