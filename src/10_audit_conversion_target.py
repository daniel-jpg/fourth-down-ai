import polars as pl


pl.Config.set_tbl_rows(50)
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
# Offensive touchdowns only
# ---------------------------------------------------------

offensive_td = (
    (pl.col("touchdown").fill_null(0) == 1)
    &
    (pl.col("td_team") == pl.col("posteam"))
)

defensive_td = (
    (pl.col("touchdown").fill_null(0) == 1)
    &
    (pl.col("td_team") == pl.col("defteam"))
)


offensive_touchdowns = go.filter(
    offensive_td
)

offensive_td_not_converted = go.filter(
    offensive_td
    &
    (pl.col("converted") == 0)
)

defensive_touchdowns = go.filter(
    defensive_td
)


print("\nTOUCHDOWN CHECK")

print(
    f"Offensive touchdowns: "
    f"{offensive_touchdowns.height}"
)

print(
    f"Offensive touchdowns marked NOT converted: "
    f"{offensive_td_not_converted.height}"
)

print(
    f"Defensive touchdowns: "
    f"{defensive_touchdowns.height}"
)


# ---------------------------------------------------------
# Yardage disagreements
# ---------------------------------------------------------

enough_yards_not_converted = go.filter(
    (pl.col("yards_gained") >= pl.col("ydstogo"))
    &
    (pl.col("converted") == 0)
)

converted_without_enough_yards = go.filter(
    (pl.col("converted") == 1)
    &
    (pl.col("yards_gained") < pl.col("ydstogo"))
)


print("\nYARDAGE CHECK")

print(
    "Enough yards but NOT converted: "
    f"{enough_yards_not_converted.height}"
)

print(
    "Converted without enough recorded yards: "
    f"{converted_without_enough_yards.height}"
)


# ---------------------------------------------------------
# Show any offensive TD problem
# ---------------------------------------------------------

wanted = [
    "season",
    "week",
    "game_id",
    "play_id",
    "posteam",
    "defteam",
    "td_team",
    "action",
    "ydstogo",
    "yards_gained",
    "first_down",
    "converted",
    "touchdown",
    "desc",
]

wanted = [
    col for col in wanted
    if col in go.columns
]


if offensive_td_not_converted.height > 0:

    print("\nOFFENSIVE TDs MARKED NOT CONVERTED")

    print(
        offensive_td_not_converted
        .select(wanted)
    )


print("\nCONVERTED WITHOUT ENOUGH YARDS")

print(
    converted_without_enough_yards
    .select(wanted)
)