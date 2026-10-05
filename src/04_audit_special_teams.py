import polars as pl


pl.Config.set_tbl_rows(50)
pl.Config.set_fmt_str_lengths(180)


# ---------------------------------------------------------
# Load labeled data
# ---------------------------------------------------------

df = pl.read_parquet("data/fourth_downs_labeled.parquet")

print(f"Loaded {df.height:,} labeled fourth-down plays.")


fake_actions = [
    "FAKE_PUNT_RUN",
    "FAKE_PUNT_PASS",
    "FAKE_FG_RUN",
    "FAKE_FG_PASS",
]


# ---------------------------------------------------------
# Basic integrity checks
# ---------------------------------------------------------

print("\nNULL ACTION LABELS")
print(df["action"].null_count())

print("\nTOTAL ACTION COUNTS")
print(
    df.group_by("action")
    .len()
    .sort("len", descending=True)
)


# ---------------------------------------------------------
# Look for suspicious fake plays
#
# We do NOT automatically relabel these yet.
# We want to inspect them first.
# ---------------------------------------------------------

desc = pl.col("desc").fill_null("")

if "safety" in df.columns:
    safety_signal = (
        pl.col("safety").fill_null(0) == 1
    )
else:
    safety_signal = pl.lit(False)

suspicious_text = desc.str.contains(
    "(?i)safety|aborted|fumble|bad snap|sacked"
)

large_loss = (
    pl.col("yards_gained").fill_null(0) <= -10
)


suspicious_fake = df.filter(
    pl.col("action").is_in(fake_actions)
    &
    (
        safety_signal
        | suspicious_text
        | large_loss
    )
)


print("\nSUSPICIOUS FAKE PLAYS")
print(f"Rows: {suspicious_fake.height}")


wanted = [
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
    "action",
    "yards_gained",
    "first_down",
    "safety",
    "aborted_play",
    "desc",
]

wanted = [
    col for col in wanted
    if col in suspicious_fake.columns
]


print(
    suspicious_fake
    .select(wanted)
    .head(50)
)


# ---------------------------------------------------------
# Inspect all broken special-teams plays
# ---------------------------------------------------------

broken = df.filter(
    pl.col("action") == "BROKEN_SPECIAL_TEAMS_PLAY"
)

print("\nBROKEN SPECIAL-TEAMS PLAYS")
print(f"Rows: {broken.height}")

print(
    broken
    .group_by("play_type")
    .len()
    .sort("len", descending=True)
)


# ---------------------------------------------------------
# Inspect UNKNOWN rows
# ---------------------------------------------------------

unknown = df.filter(
    pl.col("action") == "UNKNOWN"
)

print("\nUNKNOWN PLAYS")
print(f"Rows: {unknown.height}")

print(
    unknown
    .group_by("play_type")
    .len()
    .sort("len", descending=True)
)


# ---------------------------------------------------------
# Save audit files
# ---------------------------------------------------------

suspicious_fake.select(wanted).write_csv(
    "data/suspicious_fake_plays.csv"
)

broken.select([
    col for col in wanted
    if col in broken.columns
]).write_csv(
    "data/broken_special_teams.csv"
)

unknown.write_csv(
    "data/unknown_fourth_downs.csv"
)


print("\nSaved:")
print("data/suspicious_fake_plays.csv")
print("data/broken_special_teams.csv")
print("data/unknown_fourth_downs.csv")