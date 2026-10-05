import polars as pl
import nflreadpy as nfl


pl.Config.set_tbl_rows(120)
pl.Config.set_fmt_str_lengths(150)


# ---------------------------------------------------------
# Development data only.
#
# 2025 remains untouched.
# ---------------------------------------------------------

model = (
    pl.read_parquet(
        "data/fourth_down_modeling_split.parquet"
    )
    .filter(
        (pl.col("action") == "PUNT")
        &
        (pl.col("season") <= 2024)
    )
)


print("\nPUNT SAMPLE BY SPLIT")

print(
    model
    .group_by("split")
    .len()
    .sort("split")
)


print("\nPUNT EXECUTION STATUS BY SPLIT")

print(
    model
    .group_by([
        "split",
        "execution_status",
    ])
    .len()
    .sort([
        "split",
        "execution_status",
    ])
)


# ---------------------------------------------------------
# Load raw PBP so we can recover punt-specific fields that
# were intentionally not included in the generic modeling
# dataset.
# ---------------------------------------------------------

print("\nLoading 2014-2024 PBP...")

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
# Punt-specific fields we want to inspect.
#
# Use only fields actually available in this nflverse
# version so the audit does not crash if a column name is
# absent.
# ---------------------------------------------------------

candidate_fields = [

    "play_type",

    "punt_attempt",
    "punt_blocked",

    "punter_player_id",
    "punter_player_name",

    "kick_distance",
    "return_yards",

    "punt_inside_twenty",
    "punt_in_endzone",
    "punt_out_of_bounds",
    "punt_downed",
    "punt_fair_catch",

    "touchback",

    "fumble",
    "fumble_lost",

    "td_team",

    "special_teams_play",
    "special_teams_result",

    "penalty",
    "penalty_team",
    "penalty_yards",
]


existing = [
    column
    for column in candidate_fields
    if column in pbp.columns
]


missing = [
    column
    for column in candidate_fields
    if column not in pbp.columns
]


print("\nAVAILABLE PUNT FIELDS")

for column in existing:
    print(column)


print("\nMISSING CANDIDATE FIELDS")

for column in missing:
    print(column)


# ---------------------------------------------------------
# Prefix raw fields to avoid collisions with generic
# modeling columns.
# ---------------------------------------------------------

raw = pbp.select(
    [
        "game_id",
        "play_id",
    ]
    +
    existing
)


rename_map = {
    column: f"pbp_{column}"
    for column in existing
}


raw = raw.rename(
    rename_map
)


punts = model.join(
    raw,
    on=[
        "game_id",
        "play_id",
    ],
    how="left",
)


assert punts.height == model.height


# ---------------------------------------------------------
# Join sanity check.
# ---------------------------------------------------------

if "pbp_play_type" in punts.columns:

    missing_join = punts.filter(
        pl.col("pbp_play_type").is_null()
    )

    print("\nPBP JOIN CHECK")

    print(
        f"Punt rows:      {punts.height:,}"
    )

    print(
        f"No play_type:   {missing_join.height:,}"
    )


# ---------------------------------------------------------
# Raw play type vs our intent/execution labels.
#
# This should expose bad-snap / broken punts whose raw PBP
# result looks like a run or pass.
# ---------------------------------------------------------

if "pbp_play_type" in punts.columns:

    print("\nRAW PLAY TYPE BY EXECUTION STATUS")

    print(
        punts
        .group_by([
            "execution_status",
            "pbp_play_type",
        ])
        .len()
        .sort([
            "execution_status",
            "pbp_play_type",
        ])
    )


# ---------------------------------------------------------
# Null / availability audit.
# ---------------------------------------------------------

punt_fields = [
    column
    for column in punts.columns
    if column.startswith("pbp_")
]


print("\nPUNT FIELD NULL COUNTS")

null_rows = []

for column in punt_fields:

    null_rows.append({

        "field":
            column,

        "nulls":
            punts[column]
            .null_count(),

        "available":
            punts.height
            -
            punts[column]
            .null_count(),

    })


print(
    pl.DataFrame(
        null_rows
    )
    .sort("field")
)


# ---------------------------------------------------------
# Binary punt flags.
# ---------------------------------------------------------

binary_fields = [

    "pbp_punt_attempt",
    "pbp_punt_blocked",

    "pbp_punt_inside_twenty",
    "pbp_punt_in_endzone",
    "pbp_punt_out_of_bounds",
    "pbp_punt_downed",
    "pbp_punt_fair_catch",

    "pbp_touchback",

    "pbp_fumble",
    "pbp_fumble_lost",

    "pbp_special_teams_play",

    "pbp_penalty",
]


binary_fields = [
    column
    for column in binary_fields
    if column in punts.columns
]


print("\nPUNT BINARY FLAGS BY SPLIT")

for column in binary_fields:

    print(f"\n{column}")

    print(
        punts
        .group_by("split")
        .agg(
            pl.col(column)
            .fill_null(0)
            .sum()
            .alias("count")
        )
        .sort("split")
    )


# ---------------------------------------------------------
# Special-teams result categories.
# ---------------------------------------------------------

if "pbp_special_teams_result" in punts.columns:

    print("\nSPECIAL TEAMS RESULT")

    print(
        punts
        .group_by([
            "split",
            "pbp_special_teams_result",
        ])
        .len()
        .sort([
            "split",
            "pbp_special_teams_result",
        ])
    )


# ---------------------------------------------------------
# Punt distance and return-yard distributions.
# ---------------------------------------------------------

numeric_fields = [
    column
    for column in [
        "pbp_kick_distance",
        "pbp_return_yards",
        "pbp_penalty_yards",
    ]
    if column in punts.columns
]


print("\nPUNT NUMERIC SUMMARY")

for column in numeric_fields:

    print(f"\n{column}")

    print(
        punts
        .group_by("split")
        .agg([

            pl.col(column)
            .count()
            .alias("non_null"),

            pl.col(column)
            .mean()
            .alias("mean"),

            pl.col(column)
            .median()
            .alias("median"),

            pl.col(column)
            .quantile(0.10)
            .alias("p10"),

            pl.col(column)
            .quantile(0.90)
            .alias("p90"),

            pl.col(column)
            .min()
            .alias("min"),

            pl.col(column)
            .max()
            .alias("max"),

        ])
        .sort("split")
    )


# ---------------------------------------------------------
# Touchdowns involving PUNT-intent plays.
#
# We need to distinguish:
#   - return TD
#   - kicking-team TD
#   - broken-punt TD
#   - data oddities
# ---------------------------------------------------------

print("\nPUNT TOUCHDOWNS")

td_columns = [

    "season",
    "week",
    "split",

    "game_id",
    "play_id",

    "posteam",
    "defteam",

    "yardline_100",
    "game_seconds_remaining",

    "execution_status",
    "touchdown",
]


for column in [
    "pbp_play_type",
    "pbp_punt_blocked",
    "pbp_kick_distance",
    "pbp_return_yards",
    "pbp_fumble_lost",
    "pbp_td_team",
    "pbp_special_teams_result",
]:

    if column in punts.columns:
        td_columns.append(column)


td_columns.append("desc")


print(
    punts
    .filter(
        pl.col("touchdown") == 1
    )
    .select(
        td_columns
    )
    .sort([
        "season",
        "week",
    ])
)


# ---------------------------------------------------------
# Fumble-lost punt plays.
# ---------------------------------------------------------

if "pbp_fumble_lost" in punts.columns:

    print("\nPUNT FUMBLE-LOST PLAYS")

    fumble_columns = [

        "season",
        "week",
        "split",

        "game_id",
        "play_id",

        "posteam",
        "defteam",

        "execution_status",
    ]


    for column in [
        "pbp_play_type",
        "pbp_punt_blocked",
        "pbp_kick_distance",
        "pbp_return_yards",
        "pbp_td_team",
        "pbp_special_teams_result",
    ]:

        if column in punts.columns:
            fumble_columns.append(
                column
            )


    fumble_columns.append(
        "desc"
    )


    print(
        punts
        .filter(
            pl.col("pbp_fumble_lost")
            .fill_null(0)
            == 1
        )
        .select(
            fumble_columns
        )
        .sort([
            "season",
            "week",
        ])
    )


# ---------------------------------------------------------
# Our 25 BROKEN punt executions.
#
# These must NOT silently contaminate the ordinary punt
# distance/return model.
# ---------------------------------------------------------

print("\nBROKEN PUNT DETAILS")

broken_columns = [

    "season",
    "week",
    "split",

    "game_id",
    "play_id",

    "qtr",
    "game_seconds_remaining",
    "yardline_100",

    "execution_status",
]


for column in [
    "pbp_play_type",
    "pbp_punt_attempt",
    "pbp_punt_blocked",
    "pbp_kick_distance",
    "pbp_return_yards",
    "pbp_fumble_lost",
    "pbp_td_team",
    "pbp_special_teams_result",
]:

    if column in punts.columns:
        broken_columns.append(
            column
        )


broken_columns.append(
    "desc"
)


print(
    punts
    .filter(
        pl.col("execution_status")
        == "BROKEN"
    )
    .select(
        broken_columns
    )
    .sort([
        "season",
        "week",
    ])
)


# ---------------------------------------------------------
# Save the joined development audit so we don't need to
# reload the full nflverse PBP for every punt script.
# ---------------------------------------------------------

punts.write_parquet(
    "data/punt_outcome_audit.parquet"
)


print("\nSaved:")

print(
    "data/punt_outcome_audit.parquet"
)
