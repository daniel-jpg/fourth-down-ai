import polars as pl
import nflreadpy as nfl


pl.Config.set_tbl_rows(100)
pl.Config.set_fmt_str_lengths(120)


# ---------------------------------------------------------
# Load development seasons only.
#
# 2025 remains untouched.
# ---------------------------------------------------------

seasons = list(
    range(
        2014,
        2025,
    )
)


print(
    "Loading full play-by-play..."
)

pbp = nfl.load_pbp(
    seasons
)


pbp = pbp.sort([
    "game_id",
    "play_id",
])


print(
    f"Loaded {pbp.height:,} plays."
)


# ---------------------------------------------------------
# Define an actual offensive scrimmage play.
#
# We intentionally skip kickoffs, PATs, punts, FGs,
# timeouts, and no-play penalties.
#
# If a penalty occurs before the next real snap, the
# following snap's pre-play state should already reflect it.
# ---------------------------------------------------------

is_scrimmage = (
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


# ---------------------------------------------------------
# For every PBP row, attach the first subsequent offensive
# scrimmage state in that same game.
#
# Some broken FG plays are encoded as run/pass.
# Shift first so only a strictly later scrimmage row
# can become the next state.
# ---------------------------------------------------------

def next_scrimmage_expr(
    column,
    alias,
):

    return (
        pl.when(
            is_scrimmage
        )
        .then(
            pl.col(column)
        )
        .otherwise(None)

        # Shift first so the CURRENT row can never
        # qualify as its own "next" scrimmage play.
        .shift(-1)

        # Then find the first subsequent scrimmage row.
        .backward_fill()
        .over("game_id")
        .alias(alias)
    )


pbp = pbp.with_columns([

    next_scrimmage_expr(
        "play_id",
        "next_play_id",
    ),

    next_scrimmage_expr(
        "qtr",
        "next_qtr",
    ),

    next_scrimmage_expr(
        "game_seconds_remaining",
        "next_game_seconds_remaining",
    ),

    next_scrimmage_expr(
        "posteam",
        "next_posteam",
    ),

    next_scrimmage_expr(
        "defteam",
        "next_defteam",
    ),

    next_scrimmage_expr(
        "down",
        "next_down",
    ),

    next_scrimmage_expr(
        "ydstogo",
        "next_ydstogo",
    ),

    next_scrimmage_expr(
        "yardline_100",
        "next_yardline_100",
    ),

    next_scrimmage_expr(
        "posteam_score",
        "next_posteam_score",
    ),

    next_scrimmage_expr(
        "defteam_score",
        "next_defteam_score",
    ),

    next_scrimmage_expr(
        "desc",
        "next_desc",
    ),

])


# ---------------------------------------------------------
# Load our reviewed FG decisions.
# ---------------------------------------------------------

labeled = pl.read_parquet(
    "data/fourth_downs_labeled.parquet"
)


fg = (
    labeled
    .filter(
        (pl.col("action") == "FIELD_GOAL")
        &
        (pl.col("season") <= 2024)
    )
    .with_columns(

        pl.when(
            pl.col("execution_status")
            == "BROKEN"
        )
        .then(
            pl.lit("broken")
        )

        .otherwise(
            pl.col("field_goal_result")
            .fill_null("unknown")
            .str.to_lowercase()
        )

        .alias("fg_outcome")

    )
    .select([
        "season",
        "week",
        "game_id",
        "play_id",

        "posteam",
        "defteam",

        "qtr",
        "game_seconds_remaining",
        "yardline_100",
        "score_differential",

        "execution_status",
        "fg_outcome",
        "desc",
    ])
)


# ---------------------------------------------------------
# Attach next-scrimmage state.
# ---------------------------------------------------------

next_columns = [
    "game_id",
    "play_id",

    "next_play_id",
    "next_qtr",
    "next_game_seconds_remaining",

    "next_posteam",
    "next_defteam",

    "next_down",
    "next_ydstogo",
    "next_yardline_100",

    "next_posteam_score",
    "next_defteam_score",

    "next_desc",
]


audit = fg.join(
    pbp.select(
        next_columns
    ),
    on=[
        "game_id",
        "play_id",
    ],
    how="left",
)

# ---------------------------------------------------------
# Sanity check:
# every attached "next" play must actually occur later.
# ---------------------------------------------------------

bad_future_rows = audit.filter(
    pl.col("next_play_id").is_not_null()
    &
    (
        pl.col("next_play_id")
        <= pl.col("play_id")
    )
)


print(
    "\nSTRICT FUTURE CHECK"
)

print(
    f"Rows whose next play is not actually later: "
    f"{bad_future_rows.height}"
)


assert (
    bad_future_rows.height == 0
), (
    "Found a next-scrimmage state that is not strictly "
    "after the field-goal play."
)


# ---------------------------------------------------------
# Derived transition information
# ---------------------------------------------------------

audit = audit.with_columns([

    pl.col(
        "next_play_id"
    )
    .is_not_null()
    .alias(
        "next_state_found"
    ),

    (
        pl.col("next_posteam")
        ==
        pl.col("defteam")
    )
    .alias(
        "possession_flipped"
    ),

    (
        pl.col("next_posteam")
        ==
        pl.col("posteam")
    )
    .alias(
        "offense_kept_ball"
    ),

    (
        pl.col(
            "game_seconds_remaining"
        )
        -
        pl.col(
            "next_game_seconds_remaining"
        )
    )
    .alias(
        "seconds_to_next_state"
    ),

])


# ---------------------------------------------------------
# Convert next state's score differential back into the
# ORIGINAL fourth-down offense's perspective.
# ---------------------------------------------------------

audit = audit.with_columns(

    pl.when(
        pl.col("next_posteam")
        ==
        pl.col("posteam")
    )
    .then(
        pl.col("next_posteam_score")
        -
        pl.col("next_defteam_score")
    )

    .when(
        pl.col("next_posteam")
        ==
        pl.col("defteam")
    )
    .then(
        pl.col("next_defteam_score")
        -
        pl.col("next_posteam_score")
    )

    .otherwise(None)

    .alias(
        "next_original_score_differential"
    )

)


audit = audit.with_columns(

    (
        pl.col(
            "next_original_score_differential"
        )
        -
        pl.col(
            "score_differential"
        )
    )
    .alias(
        "score_differential_change"
    )

)


# ---------------------------------------------------------
# Coverage
# ---------------------------------------------------------

print(
    "\nNEXT SCRIMMAGE COVERAGE"
)

print(
    audit
    .group_by(
        "fg_outcome"
    )
    .agg([

        pl.len()
        .alias("plays"),

        pl.col(
            "next_state_found"
        )
        .sum()
        .alias("found"),

        pl.col(
            "next_state_found"
        )
        .mean()
        .alias("found_rate"),

    ])
    .sort(
        "fg_outcome"
    )
)


# ---------------------------------------------------------
# Possession transition
# ---------------------------------------------------------

found = audit.filter(
    pl.col(
        "next_state_found"
    )
)


print(
    "\nPOSSESSION TRANSITION"
)

print(
    found
    .group_by(
        "fg_outcome"
    )
    .agg([

        pl.len()
        .alias("plays"),

        pl.col(
            "possession_flipped"
        )
        .mean()
        .alias(
            "flipped_rate"
        ),

        pl.col(
            "offense_kept_ball"
        )
        .mean()
        .alias(
            "kept_rate"
        ),

    ])
    .sort(
        "fg_outcome"
    )
)


# ---------------------------------------------------------
# State-transition summary
# ---------------------------------------------------------

print(
    "\nTRANSITION SUMMARY"
)

print(
    found
    .group_by(
        "fg_outcome"
    )
    .agg([

        pl.len()
        .alias("plays"),

        pl.col(
            "seconds_to_next_state"
        )
        .mean()
        .alias(
            "avg_seconds"
        ),

        pl.col(
            "seconds_to_next_state"
        )
        .median()
        .alias(
            "median_seconds"
        ),

        pl.col(
            "score_differential_change"
        )
        .mean()
        .alias(
            "avg_score_change"
        ),

        pl.col(
            "next_yardline_100"
        )
        .mean()
        .alias(
            "avg_next_yardline"
        ),

        pl.col(
            "next_yardline_100"
        )
        .median()
        .alias(
            "median_next_yardline"
        ),

    ])
    .sort(
        "fg_outcome"
    )
)


# ---------------------------------------------------------
# Made FGs should normally change original offense score
# differential by +3.
# ---------------------------------------------------------

print(
    "\nMADE FG SCORE CHANGE"
)

print(
    found
    .filter(
        pl.col(
            "fg_outcome"
        )
        == "made"
    )
    .group_by(
        "score_differential_change"
    )
    .len()
    .sort(
        "len",
        descending=True,
    )
)


# ---------------------------------------------------------
# Missed FG transition by kick distance.
# ---------------------------------------------------------

missed = (
    found
    .filter(
        pl.col(
            "fg_outcome"
        )
        == "missed"
    )
    .with_columns(
        (
            pl.col(
                "yardline_100"
            )
            + 18
        )
        .alias(
            "fg_distance_estimate"
        )
    )
)


print(
    "\nMISSED FG NEXT FIELD POSITION"
)

print(
    missed
    .with_columns(

        pl.when(
            pl.col(
                "fg_distance_estimate"
            )
            <= 39
        )
        .then(
            pl.lit("<=39")
        )

        .when(
            pl.col(
                "fg_distance_estimate"
            )
            <= 49
        )
        .then(
            pl.lit("40-49")
        )

        .when(
            pl.col(
                "fg_distance_estimate"
            )
            <= 59
        )
        .then(
            pl.lit("50-59")
        )

        .otherwise(
            pl.lit("60+")
        )

        .alias(
            "distance_bucket"
        )

    )
    .group_by(
        "distance_bucket"
    )
    .agg([

        pl.len()
        .alias("misses"),

        pl.col(
            "next_yardline_100"
        )
        .mean()
        .alias(
            "avg_next_yardline"
        ),

        pl.col(
            "next_yardline_100"
        )
        .median()
        .alias(
            "median_next_yardline"
        ),

        pl.col(
            "seconds_to_next_state"
        )
        .mean()
        .alias(
            "avg_seconds"
        ),

    ])
    .sort(
        "distance_bucket"
    )
)


# ---------------------------------------------------------
# Broken FG plays are tiny enough to inspect individually.
# ---------------------------------------------------------

print(
    "\nBROKEN FIELD GOAL TRANSITIONS"
)

print(
    audit
    .filter(
        pl.col(
            "fg_outcome"
        )
        == "broken"
    )
    .select([
        "season",
        "week",
        "game_id",
        "play_id",

        "posteam",
        "defteam",

        "yardline_100",

        "next_posteam",
        "next_down",
        "next_ydstogo",
        "next_yardline_100",

        "score_differential_change",
        "seconds_to_next_state",

        "desc",
        "next_desc",
    ])
    .sort([
        "season",
        "week",
    ])
)

# ---------------------------------------------------------
# MISSED FG RULE-BASED FIELD POSITION
#
# Approximate kick spot:
#
#   yardline_100 + 8 yards behind LOS
#
# After a missed FG:
#
#   new offense yardline_100 ≈ 92 - old yardline_100
#
# with the own-20 placement acting as a cap:
#
#   next_yardline_100 <= 80
#
# Deviations can occur because actual holder depth varies,
# kicks can be returned, penalties occur, etc.
# ---------------------------------------------------------

miss_rule = (
    found
    .filter(
        pl.col("fg_outcome") == "missed"
    )
    .with_columns(
        pl.min_horizontal(
            pl.lit(80.0),
            (
                pl.lit(92.0)
                - pl.col("yardline_100")
            ),
        )
        .alias(
            "rule_next_yardline_100"
        )
    )
    .with_columns(
        (
            pl.col("next_yardline_100")
            -
            pl.col(
                "rule_next_yardline_100"
            )
        )
        .alias(
            "yardline_error"
        )
    )
)


print(
    "\nMISSED FG RULE BASELINE"
)

print(
    miss_rule
    .select([

        pl.len()
        .alias("misses"),

        (
            pl.col("yardline_error")
            .abs()
            < 0.5
        )
        .mean()
        .alias("exact_match_rate"),

        pl.col("yardline_error")
        .abs()
        .mean()
        .alias("mean_absolute_error"),

        pl.col("yardline_error")
        .mean()
        .alias("mean_error"),

        pl.col("yardline_error")
        .abs()
        .quantile(0.95)
        .alias("p95_absolute_error"),

        pl.col("yardline_error")
        .abs()
        .max()
        .alias("max_absolute_error"),

    ])
)


print(
    "\nLARGEST MISSED-FG RULE DEVIATIONS"
)

print(
    miss_rule
    .select([
        "season",
        "week",
        "game_id",
        "play_id",

        "yardline_100",
        "rule_next_yardline_100",
        "next_yardline_100",
        "yardline_error",

        "seconds_to_next_state",

        "desc",
        "next_desc",
    ])
    .sort(
        pl.col(
            "yardline_error"
        )
        .abs(),
        descending=True,
    )
    .head(20)
)


# ---------------------------------------------------------
# MADE FG -> kickoff transition by season
#
# This is important because kickoff/touchback rules have
# changed over time. We should NOT pool all seasons without
# checking for structural rule changes.
# ---------------------------------------------------------

print(
    "\nMADE FG NEXT FIELD POSITION BY SEASON"
)

print(
    found
    .filter(
        pl.col("fg_outcome") == "made"
    )
    .group_by("season")
    .agg([

        pl.len()
        .alias("made_fgs"),

        pl.col("next_yardline_100")
        .mean()
        .alias("mean_next_yardline"),

        pl.col("next_yardline_100")
        .median()
        .alias("median_next_yardline"),

        (
            pl.col("next_yardline_100")
            == 80
        )
        .mean()
        .alias("own_20_rate"),

        (
            pl.col("next_yardline_100")
            == 75
        )
        .mean()
        .alias("own_25_rate"),

        (
            pl.col("next_yardline_100")
            == 70
        )
        .mean()
        .alias("own_30_rate"),

        (
            pl.col(
                "score_differential_change"
            )
            == 3
        )
        .mean()
        .alias("exact_plus_3_rate"),

    ])
    .sort("season")
)


# ---------------------------------------------------------
# LIVE-BALL FG FAILURES
#
# Blocked and broken plays can create returns, fumbles,
# touchdowns, recoveries, etc.
#
# Classify SCORE CHANGE FIRST. Otherwise a defensive TD
# followed by a kickoff can misleadingly look like the
# kicking team "kept" possession at the next scrimmage.
# ---------------------------------------------------------

live_failure = (
    found
    .filter(
        pl.col("fg_outcome").is_in([
            "blocked",
            "broken",
        ])
    )
    .with_columns(

        pl.when(
            pl.col(
                "score_differential_change"
            ) < 0
        )
        .then(
            pl.lit("opponent_scored")
        )

        .when(
            pl.col(
                "score_differential_change"
            ) > 0
        )
        .then(
            pl.lit("original_offense_scored")
        )

        .when(
            pl.col("next_posteam")
            == pl.col("defteam")
        )
        .then(
            pl.lit("opponent_ball_no_score")
        )

        .when(
            pl.col("next_posteam")
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
)


print(
    "\nLIVE-BALL FAILURE TRANSITION CLASSES"
)

print(
    live_failure
    .group_by([
        "fg_outcome",
        "transition_class",
    ])
    .agg([

        pl.len()
        .alias("plays"),

        pl.col(
            "score_differential_change"
        )
        .mean()
        .alias("avg_score_change"),

        pl.col(
            "next_yardline_100"
        )
        .mean()
        .alias("avg_next_yardline"),

        pl.col(
            "seconds_to_next_state"
        )
        .mean()
        .alias("avg_seconds"),

    ])
    .sort([
        "fg_outcome",
        "transition_class",
    ])
)


# ---------------------------------------------------------
# Scoreless live-ball transitions specifically
# ---------------------------------------------------------

print(
    "\nSCORELESS LIVE-BALL FIELD POSITION"
)

print(
    live_failure
    .filter(
        pl.col(
            "score_differential_change"
        )
        == 0
    )
    .group_by([
        "fg_outcome",
        "transition_class",
    ])
    .agg([

        pl.len()
        .alias("plays"),

        pl.col(
            "next_yardline_100"
        )
        .mean()
        .alias("mean_next_yardline"),

        pl.col(
            "next_yardline_100"
        )
        .median()
        .alias("median_next_yardline"),

        pl.col(
            "seconds_to_next_state"
        )
        .mean()
        .alias("mean_seconds"),

    ])
    .sort([
        "fg_outcome",
        "transition_class",
    ])
)


# ---------------------------------------------------------
# Save this audit so future scripts do not need to reload
# all 531k PBP rows again.
# ---------------------------------------------------------

audit.write_parquet(
    "data/fg_post_state_audit.parquet"
)


print(
    "\nSaved:"
)

print(
    "data/fg_post_state_audit.parquet"
)