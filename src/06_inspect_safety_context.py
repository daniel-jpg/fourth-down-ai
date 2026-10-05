import polars as pl


pl.Config.set_tbl_rows(50)
pl.Config.set_tbl_width_chars(220)
pl.Config.set_fmt_str_lengths(120)


df = pl.read_parquet("data/fourth_downs_labeled.parquet")

desc = pl.col("desc").fill_null("")


# ---------------------------------------------------------
# Punt-formation plays currently labeled as fake runs
# that resulted in safeties
# ---------------------------------------------------------

safety_candidates = (
    df
    .filter(
        (pl.col("action") == "FAKE_PUNT_RUN")
        &
        (pl.col("safety").fill_null(0) == 1)
    )
    .with_columns(
        desc.str.slice(0, 100).alias("desc_short")
    )
)


wanted_columns = [
    "season",
    "week",
    "game_id",
    "play_id",
    "posteam",
    "defteam",
    "qtr",
    "game_seconds_remaining",
    "score_differential",
    "posteam_score",
    "defteam_score",
    "ydstogo",
    "yardline_100",
    "yards_gained",
]

wanted_columns = [
    col
    for col in wanted_columns
    if col in safety_candidates.columns
]


print("\nSAFETY CONTEXT")
print(f"Rows: {safety_candidates.height}")

print("\nSAFETY GAME CONTEXT")

context = (
    safety_candidates
    .select(wanted_columns)
    .sort(["season", "week"])
)

for row in context.iter_rows(named=True):
    print(
        f"{row['season']} W{row['week']} | "
        f"{row['game_id']} | "
        f"Q{row['qtr']} | "
        f"{row['game_seconds_remaining']} sec left | "
        f"score diff {row['score_differential']} | "
        f"score {row['posteam_score']}-{row['defteam_score']} | "
        f"4th & {row['ydstogo']} | "
        f"yardline_100={row['yardline_100']} | "
        f"yards={row['yards_gained']}"
    )


safety_candidates.select(
    wanted_columns
).write_csv(
    "data/safety_context.csv"
)

print("\nSaved:")
print("data/safety_context.csv")