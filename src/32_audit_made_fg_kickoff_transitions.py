import polars as pl
import nflreadpy as nfl


pl.Config.set_tbl_rows(100)
pl.Config.set_fmt_str_lengths(140)


# ---------------------------------------------------------
# Development data only.
#
# DO NOT load 2025.
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
    f"Loaded {pbp.height:,} PBP rows."
)


# ---------------------------------------------------------
# Start from the FG audit we already built.
# ---------------------------------------------------------

fg = (
    pl.read_parquet(
        "data/fg_post_state_audit.parquet"
    )
    .filter(
        pl.col("fg_outcome")
        == "made"
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

print(
    f"Loaded {fg.height:,} made FGs."
)


# ---------------------------------------------------------
# FIRST subsequent offensive state.
#
# no_play rows are intentionally allowed:
# their PRE-PLAY state is exactly what we want.
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


# ---------------------------------------------------------
# FIRST kickoff after the FG.
# ---------------------------------------------------------

kickoff_columns = [
    "game_id",
    "play_id",
    "qtr",
    "game_seconds_remaining",
    "posteam",
    "defteam",
    "desc",
]


# Add useful kickoff fields only if nflverse provides them.
for column in [
    "touchback",
    "return_yards",
    "fumble_lost",
    "touchdown",
    "td_team",
]:
    if column in pbp.columns:
        kickoff_columns.append(
            column
        )


kickoffs = (
    pbp
    .filter(
        pl.col("play_type")
        == "kickoff"
    )
    .select(
        kickoff_columns
    )
)


rename_map = {
    column:
        f"kickoff_{column}"
    for column in kickoff_columns
    if column != "game_id"
}


kickoffs = (
    kickoffs
    .rename(
        rename_map
    )
    .sort([
        "game_id",
        "kickoff_play_id",
    ])
)


# ---------------------------------------------------------
# Independent forward-ASOF joins.
# ---------------------------------------------------------

audit = (
    fg
    .join_asof(
        state_rows,
        left_on="search_play_id",
        right_on="state_play_id",
        by="game_id",
        strategy="forward",
    )
    .join_asof(
        kickoffs,
        left_on="search_play_id",
        right_on="kickoff_play_id",
        by="game_id",
        strategy="forward",
    )
)


# ---------------------------------------------------------
# Strict future checks.
# ---------------------------------------------------------

bad_state = audit.filter(
    pl.col("state_play_id").is_not_null()
    &
    (
        pl.col("state_play_id")
        <= pl.col("play_id")
    )
)


bad_kickoff = audit.filter(
    pl.col("kickoff_play_id").is_not_null()
    &
    (
        pl.col("kickoff_play_id")
        <= pl.col("play_id")
    )
)


print(
    "\nSTRICT FUTURE CHECK"
)

print(
    f"Bad state rows:   {bad_state.height}"
)

print(
    f"Bad kickoff rows: {bad_kickoff.height}"
)

assert bad_state.height == 0
assert bad_kickoff.height == 0


# ---------------------------------------------------------
# State score differential from ORIGINAL kicking team's
# perspective.
# ---------------------------------------------------------

audit = audit.with_columns(

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


audit = audit.with_columns([

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
    ),

    (
        pl.col(
            "game_seconds_remaining"
        )
        -
        pl.col(
            "state_game_seconds_remaining"
        )
    )
    .alias(
        "seconds_to_state"
    ),

])


# ---------------------------------------------------------
# Period-transition flags.
# ---------------------------------------------------------

audit = audit.with_columns([

    (
        (pl.col("qtr") == 2)
        &
        (pl.col("state_qtr") == 3)
    )
    .fill_null(False)
    .alias(
        "crossed_halftime"
    ),

    (
        (pl.col("qtr") == 4)
        &
        (
            pl.col("state_qtr")
            >= 5
        )
    )
    .fill_null(False)
    .alias(
        "entered_overtime"
    ),

])


# ---------------------------------------------------------
# Was the ASOF kickoff actually the kickoff generated by
# the made FG?
#
# A Q3 opening kickoff after a Q2 FG is NOT.
# An overtime kickoff after an end-Q4 FG is NOT.
# ---------------------------------------------------------

audit = audit.with_columns(

    (
        pl.col(
            "kickoff_play_id"
        )
        .is_not_null()

        &

        ~(
            (pl.col("qtr") == 2)
            &
            (
                pl.col("kickoff_qtr")
                == 3
            )
        )

        &

        ~(
            (pl.col("qtr") == 4)
            &
            (
                pl.col("kickoff_qtr")
                >= 5
            )
        )
    )
    .alias(
        "fg_generated_kickoff_found"
    )

)


# ---------------------------------------------------------
# Onside indicator from description.
# ---------------------------------------------------------

audit = audit.with_columns(

    pl.col(
        "kickoff_desc"
    )
    .fill_null("")
    .str.contains(
        "(?i)onside"
    )
    .alias(
        "onside_kick"
    )

)


# ---------------------------------------------------------
# Rule / data eras.
# ---------------------------------------------------------

audit = audit.with_columns(

    pl.when(
        pl.col("season")
        <= 2015
    )
    .then(
        pl.lit("2014-2015")
    )

    .when(
        pl.col("season")
        <= 2022
    )
    .then(
        pl.lit("2016-2022")
    )

    .when(
        pl.col("season")
        == 2023
    )
    .then(
        pl.lit("2023")
    )

    .otherwise(
        pl.lit("2024")
    )

    .alias("era")

)


# ---------------------------------------------------------
# Transition class.
#
# +3 and opponent ball:
#   ordinary post-kickoff receiving state
#
# +3 and original team ball:
#   onside / kickoff recovery branch
#
# < +3:
#   opponent scored after the made FG
#
# > +3:
#   original kicking team scored again before next state
# ---------------------------------------------------------

audit = audit.with_columns(

    pl.when(
        pl.col("state_play_id")
        .is_null()
    )
    .then(
        pl.lit("no_later_state")
    )

    .when(
        pl.col("crossed_halftime")
    )
    .then(
        pl.lit("crossed_halftime")
    )

    .when(
        pl.col("entered_overtime")
    )
    .then(
        pl.lit("entered_overtime")
    )

    .when(
        (pl.col("score_change") == 3)
        &
        (
            pl.col("state_posteam")
            ==
            pl.col("defteam")
        )
    )
    .then(
        pl.lit(
            "receiver_ball_no_extra_score"
        )
    )

    .when(
        (pl.col("score_change") == 3)
        &
        (
            pl.col("state_posteam")
            ==
            pl.col("posteam")
        )
    )
    .then(
        pl.lit(
            "kicking_team_ball_no_extra_score"
        )
    )

    .when(
        pl.col("score_change")
        < 3
    )
    .then(
        pl.lit(
            "opponent_scored_after_fg"
        )
    )

    .when(
        pl.col("score_change")
        > 3
    )
    .then(
        pl.lit(
            "original_team_extra_score"
        )
    )

    .otherwise(
        pl.lit("other")
    )

    .alias(
        "transition_class"
    )

)


# ---------------------------------------------------------
# Overall transition classes.
# ---------------------------------------------------------

print(
    "\nMADE FG TRANSITION CLASSES"
)

print(
    audit
    .group_by([
        "era",
        "transition_class",
    ])
    .len()
    .sort([
        "era",
        "transition_class",
    ])
)


# ---------------------------------------------------------
# Ordinary receiving states.
# ---------------------------------------------------------

ordinary = audit.filter(
    pl.col("transition_class")
    ==
    "receiver_ball_no_extra_score"
)


print(
    "\nORDINARY MADE-FG RECEIVING STATE BY ERA"
)

print(
    ordinary
    .group_by("era")
    .agg([

        pl.len()
        .alias("plays"),

        pl.col(
            "state_yardline_100"
        )
        .mean()
        .alias(
            "mean_yardline"
        ),

        pl.col(
            "state_yardline_100"
        )
        .median()
        .alias(
            "median_yardline"
        ),

        (
            pl.col(
                "state_yardline_100"
            )
            == 80
        )
        .mean()
        .alias(
            "own_20_rate"
        ),

        (
            pl.col(
                "state_yardline_100"
            )
            == 75
        )
        .mean()
        .alias(
            "own_25_rate"
        ),

        (
            pl.col(
                "state_yardline_100"
            )
            == 70
        )
        .mean()
        .alias(
            "own_30_rate"
        ),

        pl.col(
            "seconds_to_state"
        )
        .mean()
        .alias(
            "mean_seconds"
        ),

        pl.col(
            "seconds_to_state"
        )
        .median()
        .alias(
            "median_seconds"
        ),

    ])
    .sort("era")
)


# ---------------------------------------------------------
# Kickoff / onside frequency.
# ---------------------------------------------------------

print(
    "\nMADE-FG KICKOFF TYPE BY ERA"
)

print(
    audit
    .filter(
        pl.col(
            "fg_generated_kickoff_found"
        )
    )
    .group_by("era")
    .agg([

        pl.len()
        .alias("kickoffs"),

        pl.col(
            "onside_kick"
        )
        .sum()
        .alias(
            "onside_kicks"
        ),

        pl.col(
            "onside_kick"
        )
        .mean()
        .alias(
            "onside_rate"
        ),

    ])
    .sort("era")
)


# ---------------------------------------------------------
# Made FG followed by kicking team retaining possession.
# ---------------------------------------------------------

print(
    "\nKICKING TEAM RETAINS BALL AFTER MADE FG"
)

print(
    audit
    .filter(
        pl.col("transition_class")
        ==
        "kicking_team_ball_no_extra_score"
    )
    .select([
        "season",
        "week",
        "game_id",
        "play_id",

        "qtr",
        "game_seconds_remaining",

        "score_differential",
        "score_change",

        "kickoff_play_id",
        "onside_kick",
        "kickoff_desc",

        "state_play_id",
        "state_posteam",
        "state_down",
        "state_ydstogo",
        "state_yardline_100",

        "desc",
        "state_desc",
    ])
    .sort([
        "season",
        "week",
    ])
)


# ---------------------------------------------------------
# Scoring events after the made FG and before the next
# ordinary offensive state.
# ---------------------------------------------------------

print(
    "\nSCORING AFTER MADE FG"
)

print(
    audit
    .filter(
        pl.col("transition_class")
        .is_in([
            "opponent_scored_after_fg",
            "original_team_extra_score",
        ])
    )
    .group_by([
        "era",
        "transition_class",
        "score_change",
    ])
    .len()
    .sort([
        "era",
        "transition_class",
        "score_change",
    ])
)


# ---------------------------------------------------------
# Halftime / terminal / OT cases.
# ---------------------------------------------------------

print(
    "\nMADE FG PERIOD-END CASES"
)

print(
    audit
    .filter(
        pl.col("transition_class")
        .is_in([
            "crossed_halftime",
            "entered_overtime",
            "no_later_state",
        ])
    )
    .group_by([
        "era",
        "transition_class",
    ])
    .len()
    .sort([
        "era",
        "transition_class",
    ])
)


# ---------------------------------------------------------
# Save audit.
# ---------------------------------------------------------

audit.write_parquet(
    "data/fg_made_kickoff_audit.parquet"
)


print(
    "\nSaved:"
)

print(
    "data/fg_made_kickoff_audit.parquet"
)