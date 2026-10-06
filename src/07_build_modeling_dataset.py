import polars as pl


pl.Config.set_tbl_rows(50)
pl.Config.set_fmt_str_lengths(160)


# ---------------------------------------------------------
# Load cleaned fourth-down data
# ---------------------------------------------------------

df = pl.read_parquet("data/fourth_downs_labeled.parquet")

print(f"Loaded {df.height:,} labeled fourth-down plays.")


# ---------------------------------------------------------
# Actions that represent actual fourth-down decisions
# ---------------------------------------------------------

model_actions = [
    "NORMAL_GO_RUN",
    "NORMAL_GO_PASS",
    "PUNT",
    "FIELD_GOAL",
    "FAKE_PUNT_RUN",
    "FAKE_PUNT_PASS",
    "FAKE_FG_RUN",
    "FAKE_FG_PASS",
]

model_df = df.filter(
    pl.col("action").is_in(model_actions)
)

print(f"Usable decision plays: {model_df.height:,}")


# ---------------------------------------------------------
# Pre-play state features
#
# IMPORTANT:
# These must describe what was known BEFORE the play.
# We do not use outcome information such as yards_gained
# as predictive features.
# ---------------------------------------------------------

if "posteam_type" in model_df.columns:
    is_home_expr = (
        pl.col("posteam_type") == "home"
    ).cast(pl.Int8)
elif (
    "posteam" in model_df.columns
    and "home_team" in model_df.columns
):
    is_home_expr = (
        pl.col("posteam") == pl.col("home_team")
    ).cast(pl.Int8)
else:
    is_home_expr = pl.lit(None).cast(pl.Int8)


if "location" not in model_df.columns:
    raise RuntimeError(
        "Expected nflverse venue column 'location' "
        "was not found in labeled fourth-down data."
    )


is_neutral_site_expr = (
    pl.col("location")
    .eq("Neutral")
    .fill_null(False)
    .cast(pl.Int8)
)


site_advantage_expr = (
    pl.when(
        pl.col("location") == "Neutral"
    )
    .then(pl.lit(0))
    .when(
        is_home_expr == 1
    )
    .then(pl.lit(1))
    .otherwise(pl.lit(-1))
    .cast(pl.Int8)
)


model_df = model_df.with_columns([

    is_home_expr.alias("is_home"),

    is_neutral_site_expr.alias(
        "is_neutral_site"
    ),

    site_advantage_expr.alias(
        "site_advantage"
    ),

    (
        pl.col("yardline_100") + 18
    ).alias("fg_distance_estimate"),

    (
        pl.col("ydstogo") <= 1
    ).cast(pl.Int8).alias("short_yardage"),

    (
        pl.col("yardline_100") <= 10
    ).cast(pl.Int8).alias("inside_10"),

    (
        pl.col("yardline_100") <= 20
    ).cast(pl.Int8).alias("inside_20"),

    (
        pl.col("game_seconds_remaining") <= 120
    ).cast(pl.Int8).alias("final_two_minutes"),

    (
        pl.col("score_differential") < 0
    ).cast(pl.Int8).alias("trailing"),

    (
        pl.col("score_differential") > 0
    ).cast(pl.Int8).alias("leading"),

])


# ---------------------------------------------------------
# Keep modeling columns
# ---------------------------------------------------------

candidate_columns = [
    # IDs
    "season",
    "week",
    "game_id",
    "play_id",

    # Teams
    "posteam",
    "defteam",
    "home_team",
    "away_team",
    "location",
    "is_home",
    "is_neutral_site",
    "site_advantage",

    # Core state
    "qtr",
    "game_seconds_remaining",
    "half_seconds_remaining",
    "yardline_100",
    "ydstogo",
    "goal_to_go",

    # Score
    "score_differential",
    "posteam_score",
    "defteam_score",

    # Timeouts
    "posteam_timeouts_remaining",
    "defteam_timeouts_remaining",

    # Pregame team-strength information
    "spread_line",
    "total_line",

    # Useful derived state
    "fg_distance_estimate",
    "short_yardage",
    "inside_10",
    "inside_20",
    "final_two_minutes",
    "trailing",
    "leading",

    # Action / execution metadata -- NOT predictive features
    "action_initial",
    "action",
    "execution_status",

    # Outcomes -- NOT predictive features
    "converted",
    "yards_gained",
    "touchdown",
    "field_goal_result",
    "safety",

    # Auditing only
    "desc",
]

keep_columns = [
    col
    for col in candidate_columns
    if col in model_df.columns
]

model_df = model_df.select(keep_columns)


# ---------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------

print("\nACTION COUNTS")
print(
    model_df
    .group_by("action")
    .len()
    .sort("len", descending=True)
)

print("\nACTION / EXECUTION STATUS COUNTS")

print(
    model_df
    .group_by([
        "action",
        "execution_status",
    ])
    .len()
    .sort([
        "action",
        "execution_status",
    ])
)

print("\nCOLUMN NULL COUNTS")

null_summary = (
    model_df
    .null_count()
    .transpose(
        include_header=True,
        header_name="column",
        column_names=["nulls"],
    )
    .sort("nulls", descending=True)
)

print(null_summary)


print("\nMODELING COLUMNS")

for col in model_df.columns:
    print(col)


# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

model_df.write_parquet(
    "data/fourth_down_modeling.parquet"
)

print("\nSaved:")
print("data/fourth_down_modeling.parquet")
