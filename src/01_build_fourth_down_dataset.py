import nflreadpy as nfl
import polars as pl


SEASONS = list(range(2014, 2026))

print("Loading NFL play-by-play data...")
print(f"Seasons: {SEASONS[0]}-{SEASONS[-1]}")

pbp = nfl.load_pbp(SEASONS)

print("\nLoaded successfully!")
print(f"Rows: {pbp.height:,}")
print(f"Columns: {pbp.width}")

print("\nSample plays:")
print(
    pbp.select([
        "season",
        "week",
        "game_id",
        "play_id",
        "down",
        "ydstogo",
        "yardline_100",
        "play_type",
    ]).head(10)
)

# Extract fourth-down plays from regulation and overtime.
fourth = pbp.filter(
    (pl.col("down") == 4)
    &
    pl.col("qtr").is_not_null()
)

print("\nFOURTH DOWNS")
print(f"Rows: {fourth.height:,}")

print("\nFOURTH-DOWN PLAY TYPES")
print(
    fourth
    .group_by("play_type")
    .len()
    .sort("len", descending=True)
)

# ---------------------------------------------------------
# Inspect special-teams / fake-related columns
# ---------------------------------------------------------

keywords = [
    "fake",
    "punt",
    "field_goal",
    "special",
    "aborted",
    "play_type_nfl",
]

interesting_columns = [
    col
    for col in fourth.columns
    if any(keyword in col.lower() for keyword in keywords)
]

print("\nPOTENTIALLY RELEVANT COLUMNS")
for col in interesting_columns:
    print(col)


# ---------------------------------------------------------
# Look for plays whose description explicitly says "fake"
# ---------------------------------------------------------

fake_by_description = fourth.filter(
    pl.col("desc")
    .fill_null("")
    .str.contains("(?i)fake")
)

print("\nPLAYS CONTAINING 'FAKE' IN DESCRIPTION")
print(f"Rows: {fake_by_description.height}")

fake_display_columns = [
    "season",
    "week",
    "game_id",
    "play_id",
    "posteam",
    "defteam",
    "qtr",
    "ydstogo",
    "yardline_100",
    "play_type",
    "desc",
]

print(
    fake_by_description
    .select([
        col for col in fake_display_columns
        if col in fake_by_description.columns
    ])
)


# ---------------------------------------------------------
# Inspect NFL's more detailed play-type classification
# ---------------------------------------------------------

if "play_type_nfl" in fourth.columns:
    print("\nPLAY TYPE vs NFL PLAY TYPE")
    print(
        fourth
        .group_by(["play_type", "play_type_nfl"])
        .len()
        .sort("len", descending=True)
        .head(50)
    )
else:
    print("\nNo play_type_nfl column found.")


fourth.write_parquet("data/fourth_downs_raw.parquet")

print("\nSaved raw fourth-down data to:")
print("data/fourth_downs_raw.parquet")