import polars as pl


pl.Config.set_tbl_rows(100)
pl.Config.set_fmt_str_lengths(180)


df = pl.read_parquet(
    "data/fourth_downs_labeled.parquet"
)

go = df.filter(
    pl.col("action").is_in([
        "NORMAL_GO_RUN",
        "NORMAL_GO_PASS",
    ])
)

print(f"Normal go plays: {go.height:,}")


# ---------------------------------------------------------
# See which nflverse play-type helper columns are available
# ---------------------------------------------------------

check_columns = [
    "qb_scramble",
    "qb_dropback",
    "pass_attempt",
    "rush_attempt",
    "sack",
    "complete_pass",
    "incomplete_pass",
    "aborted_play",
]

print("\nAVAILABLE COLUMNS")

for col in check_columns:
    print(
        f"{col}: "
        f"{col in go.columns}"
    )


# ---------------------------------------------------------
# Action x QB scramble
# ---------------------------------------------------------

if "qb_scramble" in go.columns:

    print("\nQB SCRAMBLES BY CURRENT ACTION")

    print(
        go
        .group_by([
            "action",
            "qb_scramble",
        ])
        .len()
        .sort([
            "action",
            "qb_scramble",
        ])
    )


# ---------------------------------------------------------
# Action x QB dropback
#
# qb_dropback is especially important:
# pass attempts, sacks, and scrambles commonly come from
# intended passing plays.
# ---------------------------------------------------------

if "qb_dropback" in go.columns:

    print("\nQB DROPBACK BY CURRENT ACTION")

    print(
        go
        .group_by([
            "action",
            "qb_dropback",
        ])
        .len()
        .sort([
            "action",
            "qb_dropback",
        ])
    )


# ---------------------------------------------------------
# Action x rush/pass attempts
# ---------------------------------------------------------

if (
    "rush_attempt" in go.columns
    and
    "pass_attempt" in go.columns
):

    print("\nRUSH / PASS ATTEMPT FLAGS")

    print(
        go
        .group_by([
            "action",
            "rush_attempt",
            "pass_attempt",
        ])
        .len()
        .sort([
            "action",
            "rush_attempt",
            "pass_attempt",
        ])
    )


# ---------------------------------------------------------
# Description-based scramble check
# ---------------------------------------------------------

scramble_desc = go.filter(
    pl.col("desc")
    .fill_null("")
    .str.contains(
        "(?i)scramble|scrambles"
    )
)

print("\nDESCRIPTION SCRAMBLES")

print(
    f"Rows with scramble in description: "
    f"{scramble_desc.height}"
)

print(
    scramble_desc
    .group_by("action")
    .len()
    .sort("action")
)


# ---------------------------------------------------------
# Suspicious RUN rows
#
# These are currently labeled as RUN but look like they
# may have originated from a pass/dropback.
# ---------------------------------------------------------

conditions = []

if "qb_scramble" in go.columns:
    conditions.append(
        pl.col("qb_scramble").fill_null(0) == 1
    )

if "qb_dropback" in go.columns:
    conditions.append(
        pl.col("qb_dropback").fill_null(0) == 1
    )


if conditions:

    suspicious_condition = conditions[0]

    for condition in conditions[1:]:
        suspicious_condition = (
            suspicious_condition
            |
            condition
        )

    suspicious_runs = go.filter(
        (pl.col("action") == "NORMAL_GO_RUN")
        &
        suspicious_condition
    )

    print("\nCURRENT RUNS THAT LOOK LIKE DROPBACK/SCRAMBLE PLAYS")

    print(
        f"Suspicious run rows: "
        f"{suspicious_runs.height}"
    )

    wanted = [
        "season",
        "week",
        "game_id",
        "play_id",
        "posteam",
        "defteam",
        "ydstogo",
        "yardline_100",
        "action",
        "play_type",
        "qb_scramble",
        "qb_dropback",
        "rush_attempt",
        "pass_attempt",
        "sack",
        "yards_gained",
        "first_down",
        "desc",
    ]

    wanted = [
        c for c in wanted
        if c in suspicious_runs.columns
    ]

    print(
        suspicious_runs
        .select(wanted)
        .head(50)
    )


# ---------------------------------------------------------
# Empirical run/pass counts by yards to go
#
# Propensity scores can make overlap look better than the
# actual observed action counts.
# ---------------------------------------------------------

print("\nACTUAL ACTION COUNTS BY YARDS TO GO")

print(
    go
    .filter(
        pl.col("ydstogo") <= 10
    )
    .group_by([
        "ydstogo",
        "action",
    ])
    .agg([
        pl.len().alias("plays"),
        pl.col("converted")
        .mean()
        .alias("conversion_rate"),
    ])
    .sort([
        "ydstogo",
        "action",
    ])
)