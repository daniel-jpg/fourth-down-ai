import polars as pl


pl.Config.set_tbl_rows(100)
pl.Config.set_fmt_str_lengths(250)


df = pl.read_parquet(
    "data/fourth_downs_labeled.parquet"
)


broken = (
    df
    .filter(
        pl.col("action")
        == "BROKEN_SPECIAL_TEAMS_PLAY"
    )
)


print(
    f"Broken special-teams plays: "
    f"{broken.height}"
)


# ---------------------------------------------------------
# Useful text signals
# ---------------------------------------------------------

desc = pl.col("desc").fill_null("")


broken = broken.with_columns([

    desc.str.contains(
        "(?i)punt formation|fake punt|lined up to punt"
    )
    .alias("punt_formation_text"),

    desc.str.contains(
        "(?i)field goal formation|field-goal formation|fake field goal|fake fg"
    )
    .alias("fg_formation_text"),

    desc.str.contains(
        "(?i)bad snap|fumbled snap|mishandled snap"
    )
    .alias("botched_snap_text"),

])


# ---------------------------------------------------------
# Select whatever useful columns actually exist
# ---------------------------------------------------------

wanted_columns = [
    "season",
    "week",
    "game_id",
    "play_id",

    "posteam",
    "defteam",

    "qtr",
    "game_seconds_remaining",

    "yardline_100",
    "ydstogo",

    "play_type",

    "punt_attempt",
    "field_goal_attempt",

    "aborted_play",
    "safety",

    "rush_attempt",
    "pass_attempt",
    "qb_scramble",

    "first_down",
    "yards_gained",
    "touchdown",

    "fumble_lost",

    "kick_distance",
    "field_goal_result",

    "punter_player_name",
    "kicker_player_name",

    "punt_formation_text",
    "fg_formation_text",
    "botched_snap_text",

    "desc",
]


available_columns = [
    col
    for col in wanted_columns
    if col in broken.columns
]


audit = (
    broken
    .select(available_columns)
    .sort([
        "season",
        "week",
        "game_id",
        "play_id",
    ])
)


# ---------------------------------------------------------
# Save audit file
# ---------------------------------------------------------

audit.write_csv(
    "data/broken_special_teams_audit.csv"
)


print("\nSaved:")
print(
    "data/broken_special_teams_audit.csv"
)


# ---------------------------------------------------------
# Print each play individually
# ---------------------------------------------------------

for i, row in enumerate(
    audit.iter_rows(named=True),
    start=1,
):

    print("\n" + "=" * 80)

    print(
        f"BROKEN PLAY {i} / "
        f"{audit.height}"
    )

    print("=" * 80)

    for key, value in row.items():

        if key != "desc":
            print(
                f"{key}: {value}"
            )

    print("\nDESCRIPTION:")
    print(
        row.get("desc")
    )
