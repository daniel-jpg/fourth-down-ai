import polars as pl


pl.Config.set_tbl_rows(100)
pl.Config.set_fmt_str_lengths(160)


df = pl.read_parquet(
    "data/fg_made_causal_kickoff_audit.parquet"
)


# ---------------------------------------------------------
# Anything that is NOT:
#
# - a clean kickoff after the made FG
# - halftime
# - overtime
# - terminal / no later state
#
# is an administrative / continuation exception.
# ---------------------------------------------------------

exceptions = df.filter(
    ~pl.col("causal_path").is_in([
        "kickoff_before_state",
        "crossed_halftime",
        "entered_overtime",
        "no_later_state",
    ])
)


print(
    "\nMADE-FG CONTINUATION EXCEPTIONS"
)

print(
    exceptions
    .group_by([
        "era",
        "causal_path",
    ])
    .len()
    .sort([
        "era",
        "causal_path",
    ])
)


# ---------------------------------------------------------
# Basic diagnostics.
# ---------------------------------------------------------

exceptions = exceptions.with_columns([

    (
        pl.col("state_posteam")
        ==
        pl.col("posteam")
    )
    .fill_null(False)
    .alias(
        "original_team_still_has_ball"
    ),

    (
        pl.col("game_seconds_remaining")
        -
        pl.col("state_game_seconds_remaining")
    )
    .alias(
        "clock_elapsed_to_state"
    ),

])


print(
    "\nEXCEPTION SCORE CHANGES"
)

print(
    exceptions
    .group_by([
        "causal_path",
        "score_change",
    ])
    .len()
    .sort([
        "causal_path",
        "score_change",
    ])
)


print(
    "\nEXCEPTION POSSESSION"
)

print(
    exceptions
    .group_by([
        "causal_path",
        "original_team_still_has_ball",
    ])
    .len()
    .sort([
        "causal_path",
        "original_team_still_has_ball",
    ])
)


print(
    "\nEXCEPTION PENALTY FLAGS"
)

print(
    exceptions
    .group_by([
        "fg_desc_has_penalty",
        "fg_desc_no_play",
    ])
    .len()
    .sort(
        "len",
        descending=True,
    )
)


# ---------------------------------------------------------
# Full details.
# ---------------------------------------------------------

print(
    "\nEXCEPTION DETAILS"
)

print(
    exceptions
    .select([
        "season",
        "week",
        "game_id",
        "play_id",

        "qtr",
        "game_seconds_remaining",

        "posteam",
        "defteam",

        "score_differential",
        "score_change",

        "fg_desc_has_penalty",
        "fg_desc_no_play",

        "causal_path",

        "state_play_id",
        "state_qtr",
        "state_game_seconds_remaining",
        "clock_elapsed_to_state",

        "state_posteam",
        "state_down",
        "state_ydstogo",
        "state_yardline_100",

        "kickoff_play_id",
        "kickoff_qtr",
        "kickoff_game_seconds_remaining",

        "desc",
        "state_desc",
        "kickoff_desc",
    ])
    .sort([
        "season",
        "week",
    ])
)