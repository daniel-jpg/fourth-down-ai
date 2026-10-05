import polars as pl
import nflreadpy as nfl


pl.Config.set_tbl_rows(100)
pl.Config.set_fmt_str_lengths(120)


# ---------------------------------------------------------
# Load PBP through 2024 only.
# 2025 remains untouched.
# ---------------------------------------------------------

print("Loading PBP...")

pbp = (
    nfl.load_pbp(
        list(range(2014, 2025))
    )
    .sort([
        "game_id",
        "play_id",
    ])
)


print(
    f"Loaded {pbp.height:,} plays."
)


# ---------------------------------------------------------
# Load the transition audit built in Script 24.
# ---------------------------------------------------------

audit = (
    pl.read_parquet(
        "data/fg_post_state_audit.parquet"
    )
    .sort([
        "game_id",
        "play_id",
    ])
)


print(
    f"Loaded {audit.height:,} FG decisions."
)


# ---------------------------------------------------------
# STRICT scrimmage definition:
# exactly the same eligibility rule used in Script 24.
# ---------------------------------------------------------

strict_mask = (
    pl.col("posteam").is_not_null()
    &
    pl.col("down").is_not_null()
    &
    pl.col("play_type").is_in([
        "run",
        "pass",
        "qb_kneel",
        "qb_spike",
    ])
)


strict = (
    pbp
    .filter(
        strict_mask
    )
    .select([
        "game_id",
        "play_id",
        "qtr",
        "game_seconds_remaining",
        "posteam",
        "down",
        "ydstogo",
        "yardline_100",
        "play_type",
        "desc",
    ])
    .rename({
        "play_id":
            "strict_play_id",

        "qtr":
            "strict_qtr",

        "game_seconds_remaining":
            "strict_game_seconds_remaining",

        "posteam":
            "strict_posteam",

        "down":
            "strict_down",

        "ydstogo":
            "strict_ydstogo",

        "yardline_100":
            "strict_yardline_100",

        "play_type":
            "strict_play_type",

        "desc":
            "strict_desc",
    })
    .sort([
        "game_id",
        "strict_play_id",
    ])
)


# ---------------------------------------------------------
# RELAXED scrimmage definition:
#
# Same play types, but DO NOT require posteam/down.
#
# If this finds an earlier row, one of those metadata
# requirements is causing us to skip a real play.
# ---------------------------------------------------------

relaxed_mask = (
    pl.col("play_type").is_in([
        "run",
        "pass",
        "qb_kneel",
        "qb_spike",
    ])
)


relaxed = (
    pbp
    .filter(
        relaxed_mask
    )
    .select([
        "game_id",
        "play_id",
        "qtr",
        "game_seconds_remaining",
        "posteam",
        "down",
        "ydstogo",
        "yardline_100",
        "play_type",
        "desc",
    ])
    .rename({
        "play_id":
            "relaxed_play_id",

        "qtr":
            "relaxed_qtr",

        "game_seconds_remaining":
            "relaxed_game_seconds_remaining",

        "posteam":
            "relaxed_posteam",

        "down":
            "relaxed_down",

        "ydstogo":
            "relaxed_ydstogo",

        "yardline_100":
            "relaxed_yardline_100",

        "play_type":
            "relaxed_play_type",

        "desc":
            "relaxed_desc",
    })
    .sort([
        "game_id",
        "relaxed_play_id",
    ])
)


# ---------------------------------------------------------
# Independent next-play method:
#
# Use a forward ASOF join instead of backward_fill.
#
# Add a tiny epsilon to play_id so the FG row itself
# can NEVER match.
# ---------------------------------------------------------

left = (
    audit
    .with_columns(
        (
            pl.col("play_id")
            + 0.000001
        )
        .alias("search_play_id")
    )
    .sort([
        "game_id",
        "search_play_id",
    ])
)


check = (
    left
    .join_asof(
        strict,
        left_on="search_play_id",
        right_on="strict_play_id",
        by="game_id",
        strategy="forward",
    )
    .join_asof(
        relaxed,
        left_on="search_play_id",
        right_on="relaxed_play_id",
        by="game_id",
        strategy="forward",
    )
)


# ---------------------------------------------------------
# Compare Script 24 window result to independent ASOF.
# ---------------------------------------------------------

check = check.with_columns([

    (
        pl.col("next_play_id")
        == pl.col("strict_play_id")
    )
    .fill_null(False)
    .alias(
        "window_matches_strict"
    ),

    (
        pl.col("strict_play_id")
        == pl.col("relaxed_play_id")
    )
    .fill_null(False)
    .alias(
        "strict_matches_relaxed"
    ),

])


print(
    "\nWINDOW METHOD VS STRICT ASOF"
)

print(
    check.select([

        pl.len()
        .alias("fg_plays"),

        (
            ~pl.col(
                "window_matches_strict"
            )
        )
        .sum()
        .alias("mismatches"),

        pl.col(
            "window_matches_strict"
        )
        .mean()
        .alias("match_rate"),

    ])
)


print(
    "\nSTRICT VS RELAXED SCRIMMAGE MASK"
)

print(
    check.select([

        pl.len()
        .alias("fg_plays"),

        (
            ~pl.col(
                "strict_matches_relaxed"
            )
        )
        .sum()
        .alias("mismatches"),

        pl.col(
            "strict_matches_relaxed"
        )
        .mean()
        .alias("match_rate"),

    ])
)


# ---------------------------------------------------------
# Show window-vs-ASOF mismatches.
# ---------------------------------------------------------

print(
    "\nWINDOW / ASOF MISMATCH EXAMPLES"
)

print(
    check
    .filter(
        ~pl.col(
            "window_matches_strict"
        )
    )
    .select([
        "season",
        "week",
        "game_id",
        "play_id",
        "fg_outcome",

        "next_play_id",
        "next_qtr",
        "next_game_seconds_remaining",
        "next_yardline_100",
        "next_desc",

        "strict_play_id",
        "strict_qtr",
        "strict_game_seconds_remaining",
        "strict_yardline_100",
        "strict_desc",
    ])
    .head(30)
)


# ---------------------------------------------------------
# Show cases where requiring posteam/down caused us to
# skip an earlier run/pass/kneel/spike row.
# ---------------------------------------------------------

print(
    "\nSTRICT / RELAXED MASK MISMATCH EXAMPLES"
)

print(
    check
    .filter(
        ~pl.col(
            "strict_matches_relaxed"
        )
    )
    .select([
        "season",
        "week",
        "game_id",
        "play_id",
        "fg_outcome",

        "strict_play_id",
        "strict_posteam",
        "strict_down",
        "strict_play_type",
        "strict_desc",

        "relaxed_play_id",
        "relaxed_posteam",
        "relaxed_down",
        "relaxed_play_type",
        "relaxed_desc",
    ])
    .head(30)
)


# ---------------------------------------------------------
# Re-evaluate missed-FG rule using independent ASOF target.
# ---------------------------------------------------------

check = check.with_columns([

    pl.min_horizontal(
        pl.lit(80.0),
        (
            pl.lit(92.0)
            - pl.col("yardline_100")
        ),
    )
    .alias("miss_rule_yardline"),

    (
        (
            pl.col("qtr") == 2
        )
        &
        (
            pl.col("strict_qtr") == 3
        )
    )
    .fill_null(False)
    .alias(
        "strict_crossed_halftime"
    ),

    (
        pl.col(
            "game_seconds_remaining"
        )
        -
        pl.col(
            "strict_game_seconds_remaining"
        )
    )
    .alias(
        "strict_seconds_to_next"
    ),

])


check = check.with_columns(
    (
        pl.col(
            "strict_yardline_100"
        )
        -
        pl.col(
            "miss_rule_yardline"
        )
    )
    .alias(
        "strict_rule_error"
    )
)


standard_miss = check.filter(
    (pl.col("fg_outcome") == "missed")
    &
    pl.col("strict_play_id").is_not_null()
    &
    ~pl.col("strict_crossed_halftime")
    &
    (
        pl.col("strict_posteam")
        ==
        pl.col("defteam")
    )
)


print(
    "\nASOF STANDARD MISSED-FG RULE PERFORMANCE"
)

print(
    standard_miss
    .with_columns(
        pl.when(
            pl.col("season") <= 2022
        )
        .then(pl.lit("train"))
        .otherwise(
            pl.lit("validation")
        )
        .alias("split")
    )
    .group_by("split")
    .agg([

        pl.len()
        .alias("misses"),

        (
            pl.col(
                "strict_rule_error"
            )
            .abs()
            < 0.5
        )
        .mean()
        .alias(
            "exact_match_rate"
        ),

        pl.col(
            "strict_rule_error"
        )
        .abs()
        .mean()
        .alias(
            "mean_absolute_error"
        ),

        pl.col(
            "strict_rule_error"
        )
        .mean()
        .alias(
            "mean_error"
        ),

        pl.col(
            "strict_rule_error"
        )
        .abs()
        .quantile(0.95)
        .alias(
            "p95_absolute_error"
        ),

    ])
    .sort("split")
)


# ---------------------------------------------------------
# Largest remaining rule errors using ASOF next state.
# ---------------------------------------------------------

largest = (
    standard_miss
    .sort(
        pl.col(
            "strict_rule_error"
        )
        .abs(),
        descending=True,
    )
    .head(10)
)


print(
    "\nLARGEST ASOF STANDARD MISS ERRORS"
)

print(
    largest.select([
        "season",
        "week",
        "game_id",
        "play_id",

        "qtr",
        "game_seconds_remaining",

        "yardline_100",
        "miss_rule_yardline",

        "strict_play_id",
        "strict_qtr",
        "strict_game_seconds_remaining",
        "strict_seconds_to_next",

        "strict_yardline_100",
        "strict_rule_error",

        "desc",
        "strict_desc",
    ])
)


# ---------------------------------------------------------
# For the five worst cases, print EVERY PBP row between
# the FG and the selected ASOF scrimmage.
#
# This tells us exactly what was skipped.
# ---------------------------------------------------------

print(
    "\nINTERMEDIATE PBP FOR WORST CASES"
)


for row in (
    largest
    .head(5)
    .iter_rows(
        named=True
    )
):

    game_id = row[
        "game_id"
    ]

    fg_play_id = row[
        "play_id"
    ]

    next_play_id = row[
        "strict_play_id"
    ]


    print(
        "\n"
        + "=" * 80
    )

    print(
        f"{game_id} | "
        f"FG play {fg_play_id} | "
        f"ASOF next {next_play_id}"
    )

    print(
        "=" * 80
    )


    segment = (
        pbp
        .filter(
            (pl.col("game_id") == game_id)
            &
            (
                pl.col("play_id")
                > fg_play_id
            )
            &
            (
                pl.col("play_id")
                <= next_play_id
            )
        )
        .select([
            "play_id",
            "qtr",
            "game_seconds_remaining",
            "posteam",
            "down",
            "ydstogo",
            "yardline_100",
            "play_type",
            "desc",
        ])
    )


    print(
        segment
    )