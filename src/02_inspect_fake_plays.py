import polars as pl


# Make terminal output easier to read
pl.Config.set_tbl_rows(100)
pl.Config.set_fmt_str_lengths(200)


# We already saved this in script 01, so no downloading needed
fourth = pl.read_parquet("data/fourth_downs_raw.parquet")

print(f"Loaded {fourth.height:,} fourth-down plays.")


# ---------------------------------------------------------
# First look only at actual run/pass attempts on 4th down
# ---------------------------------------------------------

go_plays = fourth.filter(
    pl.col("play_type").is_in(["run", "pass"])
)

print(f"Run/pass fourth-down attempts: {go_plays.height:,}")


# ---------------------------------------------------------
# See every play_type / play_type_nfl combination
# ---------------------------------------------------------

print("\nALL RUN/PASS vs NFL PLAY TYPE COMBINATIONS")

print(
    go_plays
    .group_by(["play_type", "play_type_nfl"])
    .len()
    .sort("len", descending=True)
)


# ---------------------------------------------------------
# High-confidence fake punt / fake FG candidates
# ---------------------------------------------------------

desc = pl.col("desc").fill_null("")

punt_formation_signal = desc.str.contains(
    "(?i)punt formation|fake punt"
)

fg_formation_signal = desc.str.contains(
    "(?i)field goal formation|field-goal formation|fake field goal|fake fg"
)

fake_candidates = (
    go_plays
    .filter(
        punt_formation_signal |
        fg_formation_signal
    )
    .with_columns(
        pl.when(punt_formation_signal)
        .then(pl.lit("PUNT_FORMATION"))
        .when(fg_formation_signal)
        .then(pl.lit("FIELD_GOAL_FORMATION"))
        .otherwise(pl.lit("UNKNOWN"))
        .alias("formation")
    )
)


# ---------------------------------------------------------
# Keep columns useful for manual inspection
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

    "ydstogo",
    "yardline_100",

    "formation",
    "play_type",
    "play_type_nfl",

    "special_teams_play",
    "aborted_play",

    "punt_attempt",
    "field_goal_attempt",

    "yards_gained",
    "first_down",
    "touchdown",

    "desc",
]

wanted_columns = [
    col for col in wanted_columns
    if col in fake_candidates.columns
]

fake_candidates = fake_candidates.select(wanted_columns)


print("\nFAKE / SPECIAL-TEAMS CANDIDATES")
print(f"Candidates: {fake_candidates.height}")


print("\nCOUNTS BY FORMATION AND PLAY TYPE")
print(
    fake_candidates
    .group_by(["formation", "play_type"])
    .len()
    .sort(["formation", "play_type"])
)


print("\nSPECIAL-TEAMS FLAG COMBINATIONS")
print(
    fake_candidates
    .group_by([
        "formation",
        "play_type",
        "special_teams_play",
        "punt_attempt",
        "field_goal_attempt",
        "aborted_play",
    ])
    .len()
    .sort("len", descending=True)
)


# Don't print the full table to the terminal
print("\nFirst 10 candidate plays:")
print(fake_candidates.head(10))


# Save these separately so we can inspect them easily
fake_candidates.write_csv("data/fake_candidates.csv")

print("\nSaved:")
print("data/fake_candidates.csv")