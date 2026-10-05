import polars as pl


pl.Config.set_tbl_rows(100)
pl.Config.set_fmt_str_lengths(120)


# ---------------------------------------------------------
# Load full labeled data because Script 07 intentionally
# keeps only a subset of the original nflverse columns.
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
# Candidate pre-play features we may want later
# ---------------------------------------------------------

candidates = [
    "season",
    "week",
    "game_date",

    "posteam",
    "defteam",

    "kicker_player_name",
    "kicker_player_id",

    "roof",
    "surface",
    "stadium",

    "weather",
    "temp",
    "wind",

    "yardline_100",
    "kick_distance",

    "qtr",
    "game_seconds_remaining",
    "score_differential",

    "spread_line",
    "total_line",
]


print("\nFEATURE AVAILABILITY")

for col in candidates:

    if col in fg.columns:

        nulls = (
            fg
            .select(
                pl.col(col)
                .null_count()
            )
            .item()
        )

        print(
            f"{col:25s} "
            f"YES   "
            f"nulls={nulls:,} "
            f"({nulls / fg.height:.1%})"
        )

    else:

        print(
            f"{col:25s} NO"
        )


# ---------------------------------------------------------
# Kicker coverage
# ---------------------------------------------------------

if "kicker_player_name" in fg.columns:

    print("\nKICKER COVERAGE")

    print(
        fg
        .group_by("execution_status")
        .agg([
            pl.len().alias("plays"),

            pl.col(
                "kicker_player_name"
            )
            .null_count()
            .alias("null_kicker_names"),

            pl.col(
                "kicker_player_name"
            )
            .n_unique()
            .alias("unique_kickers"),
        ])
        .sort("execution_status")
    )


    print("\nMOST COMMON KICKERS")

    print(
        fg
        .filter(
            pl.col(
                "kicker_player_name"
            ).is_not_null()
        )
        .group_by(
            "kicker_player_name"
        )
        .len()
        .sort(
            "len",
            descending=True,
        )
        .head(30)
    )


# ---------------------------------------------------------
# Environment values
# ---------------------------------------------------------

for col in [
    "roof",
    "surface",
]:

    if col in fg.columns:

        print(
            f"\n{col.upper()} COUNTS"
        )

        print(
            fg
            .group_by(col)
            .len()
            .sort(
                "len",
                descending=True,
            )
        )


# ---------------------------------------------------------
# Numeric weather summaries
# ---------------------------------------------------------

for col in [
    "temp",
    "wind",
]:

    if col in fg.columns:

        print(
            f"\n{col.upper()} SUMMARY"
        )

        print(
            fg.select([
                pl.col(col)
                .min()
                .alias("min"),

                pl.col(col)
                .quantile(0.25)
                .alias("q25"),

                pl.col(col)
                .median()
                .alias("median"),

                pl.col(col)
                .quantile(0.75)
                .alias("q75"),

                pl.col(col)
                .max()
                .alias("max"),

                pl.col(col)
                .null_count()
                .alias("nulls"),
            ])
        )


# ---------------------------------------------------------
# Validate our pre-play FG distance approximation
#
# kick_distance itself is outcome-recorded, so we do NOT
# want to depend on it as the recommender input.
# ---------------------------------------------------------

if "kick_distance" in fg.columns:

    distance_check = (
        fg
        .filter(
            pl.col(
                "kick_distance"
            ).is_not_null()
        )
        .with_columns(
            (
                pl.col("yardline_100")
                + 17
            ).alias(
                "estimated_distance"
            )
        )
        .with_columns(
            (
                pl.col("kick_distance")
                - pl.col(
                    "estimated_distance"
                )
            )
            .alias(
                "distance_difference"
            )
        )
    )


    print("\nKICK DISTANCE CHECK")

    print(
        distance_check
        .select([
            pl.len()
            .alias("plays"),

            (
                pl.col(
                    "distance_difference"
                )
                == 0
            )
            .mean()
            .alias(
                "exact_match_rate"
            ),

            pl.col(
                "distance_difference"
            )
            .mean()
            .alias(
                "mean_difference"
            ),

            pl.col(
                "distance_difference"
            )
            .abs()
            .max()
            .alias(
                "max_abs_difference"
            ),
        ])
    )


    print(
        "\nLARGEST DISTANCE MISMATCHES"
    )

    print(
        distance_check
        .filter(
            pl.col(
                "distance_difference"
            ) != 0
        )
        .select([
            "season",
            "week",
            "game_id",
            "play_id",
            "yardline_100",
            "estimated_distance",
            "kick_distance",
            "distance_difference",
            "desc",
        ])
        .sort(
            pl.col(
                "distance_difference"
            ).abs(),
            descending=True,
        )
        .head(20)
    )