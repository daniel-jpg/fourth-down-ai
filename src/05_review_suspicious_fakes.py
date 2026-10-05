import polars as pl


pl.Config.set_tbl_rows(100)
pl.Config.set_fmt_str_lengths(220)


df = pl.read_parquet("data/fourth_downs_labeled.parquet")

fake_actions = [
    "FAKE_PUNT_RUN",
    "FAKE_PUNT_PASS",
    "FAKE_FG_RUN",
    "FAKE_FG_PASS",
]

desc = pl.col("desc").fill_null("")


# ---------------------------------------------------------
# Signals
# ---------------------------------------------------------

explicit_fake = desc.str.contains(
    "(?i)fake punt|fake field goal|fake fg|it was a fake"
)

botched_snap = desc.str.contains(
    "(?i)fumbled snap|bad snap|mishandled snap|aborted"
)

intentional_safety = (
    (pl.col("safety").fill_null(0) == 1)
    &
    desc.str.contains(
        "(?i)ran ob in end zone|pushed ob in end zone|tackled in end zone"
    )
)

sacked = desc.str.contains("(?i)sacked")

fumble = desc.str.contains("(?i)fumble")


# ---------------------------------------------------------
# Only inspect current fake labels
# ---------------------------------------------------------

review = (
    df
    .filter(
        pl.col("action").is_in(fake_actions)
        &
        (
            explicit_fake
            | botched_snap
            | intentional_safety
            | sacked
            | fumble
        )
    )
    .with_columns(

        pl.when(explicit_fake)
        .then(pl.lit("EXPLICIT_FAKE"))

        .when(intentional_safety)
        .then(pl.lit("LIKELY_INTENTIONAL_SAFETY"))

        .when(botched_snap)
        .then(pl.lit("LIKELY_BOTCHED_SNAP"))

        .when(sacked)
        .then(pl.lit("FAKE_RESULTED_IN_SACK"))

        .when(fumble)
        .then(pl.lit("FAKE_RESULTED_IN_FUMBLE"))

        .otherwise(pl.lit("REVIEW"))

        .alias("review_category")
    )
)


print("\nREVIEW CATEGORY COUNTS")

print(
    review
    .group_by(["review_category", "action"])
    .len()
    .sort("len", descending=True)
)


print("\nLIKELY INTENTIONAL SAFETIES")

print(
    review
    .filter(
        pl.col("review_category") == "LIKELY_INTENTIONAL_SAFETY"
    )
    .select([
        "season",
        "week",
        "game_id",
        "play_id",
        "qtr",
        "game_seconds_remaining",
        "yardline_100",
        "ydstogo",
        "action",
        "yards_gained",
        "desc",
    ])
)


print("\nLIKELY BOTCHED SNAPS")

print(
    review
    .filter(
        pl.col("review_category") == "LIKELY_BOTCHED_SNAP"
    )
    .select([
        "season",
        "week",
        "game_id",
        "play_id",
        "action",
        "yards_gained",
        "desc",
    ])
)


review.write_csv(
    "data/suspicious_fake_review.csv"
)

print("\nSaved:")
print("data/suspicious_fake_review.csv")
