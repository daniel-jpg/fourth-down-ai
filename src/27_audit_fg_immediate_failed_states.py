import polars as pl
import nflreadpy as nfl


pl.Config.set_tbl_rows(100)
pl.Config.set_fmt_str_lengths(120)


# ---------------------------------------------------------
# Load development seasons only.
# 2025 stays untouched.
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


audit = pl.read_parquet(
    "data/fg_post_state_audit.parquet"
)


print(
    f"Loaded {pbp.height:,} PBP rows."
)

print(
    f"Loaded {audit.height:,} FG decisions."
)


# ---------------------------------------------------------
# FAILED FG outcomes only.
# ---------------------------------------------------------

failed = (
    audit
    .filter(
        pl.col("fg_outcome").is_in([
            "missed",
            "blocked",
            "broken",
        ])
    )
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


# ---------------------------------------------------------
# Immediate state row.
#
# IMPORTANT:
# Do NOT require run/pass.
#
# A no_play row can represent the first snap following the
# FG and its PRE-PLAY state is exactly what we want.
#
# Requiring down/posteam/yardline excludes kickoffs,
# conversion attempts, administrative rows, etc.
# ---------------------------------------------------------

state_rows = (
    pbp
    .filter(
        pl.col("posteam").is_not_null()
        &
        pl.col("down").is_not_null()
        &
        pl.col("yardline_100").is_not_null()
    )
    .select([
        "game_id",
        "play_id",

        "qtr",
        "game_seconds_remaining",

        "posteam",
        "defteam",

        "down",
        "ydstogo",
        "yardline_100",

        "posteam_score",
        "defteam_score",

        "play_type",
        "desc",
    ])
    .rename({

        "play_id":
            "state_play_id",

        "qtr":
            "state_qtr",

        "game_seconds_remaining":
            "state_game_seconds_remaining",

        "posteam":
            "state_posteam",

        "defteam":
            "state_defteam",

        "down":
            "state_down",

        "ydstogo":
            "state_ydstogo",

        "yardline_100":
            "state_yardline_100",

        "posteam_score":
            "state_posteam_score",

        "defteam_score":
            "state_defteam_score",

        "play_type":
            "state_play_type",

        "desc":
            "state_desc",
    })
    .sort([
        "game_id",
        "state_play_id",
    ])
)


result = (
    failed
    .join_asof(
        state_rows,
        left_on="search_play_id",
        right_on="state_play_id",
        by="game_id",
        strategy="forward",
    )
)


# ---------------------------------------------------------
# Sanity check.
# ---------------------------------------------------------

bad = result.filter(
    pl.col("state_play_id").is_not_null()
    &
    (
        pl.col("state_play_id")
        <= pl.col("play_id")
    )
)


print("\nSTRICT FUTURE CHECK")

print(
    f"Bad rows: {bad.height}"
)

assert bad.height == 0


# ---------------------------------------------------------
# Timing / halftime.
# ---------------------------------------------------------

result = result.with_columns([

    (
        (pl.col("qtr") == 2)
        &
        (pl.col("state_qtr") == 3)
    )
    .fill_null(False)
    .alias("crossed_halftime"),

    (
        pl.col("game_seconds_remaining")
        -
        pl.col("state_game_seconds_remaining")
    )
    .alias("seconds_to_state"),

])


# ---------------------------------------------------------
# Score differential from ORIGINAL kicking team's
# perspective.
# ---------------------------------------------------------

result = result.with_columns(

    pl.when(
        pl.col("state_posteam")
        ==
        pl.col("posteam")
    )
    .then(
        pl.col("state_posteam_score")
        -
        pl.col("state_defteam_score")
    )

    .when(
        pl.col("state_posteam")
        ==
        pl.col("defteam")
    )
    .then(
        pl.col("state_defteam_score")
        -
        pl.col("state_posteam_score")
    )

    .otherwise(None)

    .alias(
        "state_original_score_differential"
    )

)


result = result.with_columns(

    (
        pl.col(
            "state_original_score_differential"
        )
        -
        pl.col(
            "score_differential"
        )
    )
    .alias(
        "score_change"
    )

)


# ---------------------------------------------------------
# Missed-FG placement rule.
# ---------------------------------------------------------

result = result.with_columns(

    pl.min_horizontal(
        pl.lit(80.0),
        (
            pl.lit(92.0)
            -
            pl.col("yardline_100")
        ),
    )
    .alias(
        "miss_rule_yardline"
    )

)


result = result.with_columns(

    (
        pl.col("state_yardline_100")
        -
        pl.col("miss_rule_yardline")
    )
    .alias(
        "miss_rule_error"
    )

)


# ---------------------------------------------------------
# What type of row is now selected?
#
# no_play is GOOD here: its pre-play state is the state
# immediately before the penalty snap occurred.
# ---------------------------------------------------------

print("\nFIRST STATEFUL ROW TYPE")

print(
    result
    .group_by([
        "fg_outcome",
        "state_play_type",
    ])
    .len()
    .sort([
        "fg_outcome",
        "len",
    ])
)


# ---------------------------------------------------------
# Ordinary missed FGs:
#
# opponent possession
# no score
# no halftime
# ---------------------------------------------------------

normal_miss = result.filter(
    (pl.col("fg_outcome") == "missed")
    &
    (pl.col("state_posteam") == pl.col("defteam"))
    &
    (pl.col("score_change") == 0)
    &
    ~pl.col("crossed_halftime")
)


normal_miss = normal_miss.with_columns(

    pl.when(
        pl.col("season") <= 2022
    )
    .then(pl.lit("train"))
    .otherwise(pl.lit("validation"))
    .alias("split")

)


print("\nIMMEDIATE MISSED-FG RULE PERFORMANCE")

print(
    normal_miss
    .group_by("split")
    .agg([

        pl.len()
        .alias("misses"),

        (
            pl.col("miss_rule_error")
            .abs()
            < 0.5
        )
        .mean()
        .alias("exact_match_rate"),

        pl.col("miss_rule_error")
        .abs()
        .mean()
        .alias("mean_absolute_error"),

        pl.col("miss_rule_error")
        .mean()
        .alias("mean_error"),

        pl.col("miss_rule_error")
        .abs()
        .quantile(0.90)
        .alias("p90_absolute_error"),

        pl.col("miss_rule_error")
        .abs()
        .quantile(0.95)
        .alias("p95_absolute_error"),

    ])
    .sort("split")
)


print("\nLARGEST IMMEDIATE MISSED-FG ERRORS")

print(
    normal_miss
    .sort(
        pl.col("miss_rule_error")
        .abs(),
        descending=True,
    )
    .select([
        "season",
        "week",
        "game_id",
        "play_id",

        "yardline_100",
        "miss_rule_yardline",

        "state_play_id",
        "state_play_type",
        "state_yardline_100",

        "miss_rule_error",
        "seconds_to_state",

        "desc",
        "state_desc",
    ])
    .head(20)
)


# ---------------------------------------------------------
# Blocks / broken executions:
# scoreless opponent-possession cases.
#
# These are the cases for which we need a genuine
# transition distribution.
# ---------------------------------------------------------

live_failure = result.filter(
    pl.col("fg_outcome").is_in([
        "blocked",
        "broken",
    ])
    &
    (pl.col("state_posteam") == pl.col("defteam"))
    &
    (pl.col("score_change") == 0)
    &
    ~pl.col("crossed_halftime")
)


print("\nLIVE-BALL IMMEDIATE FIELD POSITION")

print(
    live_failure
    .group_by("fg_outcome")
    .agg([

        pl.len()
        .alias("plays"),

        pl.col("state_yardline_100")
        .mean()
        .alias("mean_yardline"),

        pl.col("state_yardline_100")
        .median()
        .alias("median_yardline"),

        pl.col("seconds_to_state")
        .mean()
        .alias("mean_seconds"),

        pl.col("miss_rule_error")
        .mean()
        .alias("mean_rule_error"),

        pl.col("miss_rule_error")
        .abs()
        .mean()
        .alias("mean_absolute_rule_error"),

        pl.col("miss_rule_error")
        .quantile(0.10)
        .alias("error_p10"),

        pl.col("miss_rule_error")
        .median()
        .alias("error_median"),

        pl.col("miss_rule_error")
        .quantile(0.90)
        .alias("error_p90"),

    ])
    .sort("fg_outcome")
)


# ---------------------------------------------------------
# All live-ball transition classes.
# ---------------------------------------------------------

live_all = result.filter(
    pl.col("fg_outcome").is_in([
        "blocked",
        "broken",
    ])
)


live_all = live_all.with_columns(

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


print("\nLIVE-BALL IMMEDIATE TRANSITION CLASSES")

print(
    live_all
    .group_by([
        "fg_outcome",
        "transition_class",
    ])
    .len()
    .sort([
        "fg_outcome",
        "transition_class",
    ])
)


# ---------------------------------------------------------
# Save cleaned immediate-state audit.
# ---------------------------------------------------------

result.write_parquet(
    "data/fg_failed_immediate_state_audit.parquet"
)


print("\nSaved:")

print(
    "data/fg_failed_immediate_state_audit.parquet"
)