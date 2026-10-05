import polars as pl


pl.Config.set_tbl_rows(50)
pl.Config.set_fmt_str_lengths(160)


# ---------------------------------------------------------
# Load raw fourth downs
# ---------------------------------------------------------

fourth = pl.read_parquet("data/fourth_downs_raw.parquet")

print(f"Loaded {fourth.height:,} fourth-down plays.")


# ---------------------------------------------------------
# Text signals for fake punt / fake field goal
# ---------------------------------------------------------

desc = pl.col("desc").fill_null("")

punt_formation = desc.str.contains(
    "(?i)punt formation|fake punt|lined up to punt.*fake"
)

fg_formation = desc.str.contains(
    "(?i)field goal formation|field-goal formation|fake field goal|fake fg"
)

# nflverse classifies QB scrambles as rush attempts / run play_type.
# For our recommender, however, we care about the intended play call.
#
# A QB scramble is treated as a PASS/DROPBACK decision that happened
# to end with the QB running.

is_run_result = (
    pl.col("play_type") == "run"
)

is_pass_result = (
    pl.col("play_type") == "pass"
)

is_scramble = (
    pl.col("qb_scramble")
    .fill_null(0)
    == 1
)

is_pass_intent = (
    is_pass_result
    |
    (
        is_run_result
        & is_scramble
    )
)

is_run_intent = (
    is_run_result
    &
    ~is_scramble
)

is_aborted = pl.col("aborted_play").fill_null(0) == 1

is_safety = pl.col("safety").fill_null(0) == 1

botched_snap = desc.str.contains(
    "(?i)fumbled snap|bad snap|mishandled snap"
)


# Intentional safeties generally happen very late,
# with the offense protecting a lead.
intentional_safety = (
    punt_formation
    & is_run_intent
    & is_safety
    & (pl.col("qtr") == 4)
    & (pl.col("game_seconds_remaining") <= 180)
    & (pl.col("score_differential") > 2)
)

non_intentional_punt_safety = (
    punt_formation
    & is_run_intent
    & is_safety
    & ~intentional_safety
)


# ---------------------------------------------------------
# Assign action labels
#
# IMPORTANT:
# Put special-teams fake rules BEFORE normal run/pass rules.
# Otherwise fake punts would get labeled NORMAL_GO_RUN/PASS.
# ---------------------------------------------------------

fourth_labeled = fourth.with_columns(

    pl.when(
    intentional_safety
)
.then(pl.lit("INTENTIONAL_SAFETY"))


.when(
    (
        (punt_formation | fg_formation)
        & (is_aborted | botched_snap)
    )
    |
    non_intentional_punt_safety
)
.then(pl.lit("BROKEN_SPECIAL_TEAMS_PLAY"))


.when(
    punt_formation
    & is_run_intent
)
.then(pl.lit("FAKE_PUNT_RUN"))

    .when(
        punt_formation
        & is_pass_intent
    )
    .then(pl.lit("FAKE_PUNT_PASS"))
.when(
    fg_formation
    & is_run_intent
)
.then(pl.lit("FAKE_FG_RUN"))
    .when(
    fg_formation
    & is_pass_intent
)
.then(pl.lit("FAKE_FG_PASS"))

.when(
    is_run_intent
)
.then(pl.lit("NORMAL_GO_RUN"))

.when(
    is_pass_intent
)
.then(pl.lit("NORMAL_GO_PASS"))

.when(
    pl.col("play_type") == "punt"
)
.then(pl.lit("PUNT"))

    .when(
        pl.col("play_type") == "field_goal"
    )
    .then(pl.lit("FIELD_GOAL"))

    .when(
        pl.col("play_type") == "no_play"
    )
    .then(pl.lit("NO_PLAY"))

    .when(
        pl.col("play_type") == "qb_kneel"
    )
    .then(pl.lit("QB_KNEEL"))

    .otherwise(pl.lit("UNKNOWN"))

    .alias("action_initial")
)


# ---------------------------------------------------------
# Mark whether the offense converted
# ---------------------------------------------------------

fourth_labeled = fourth_labeled.with_columns(

    (
        pl.col("first_down")
        .fill_null(0)
        == 1
    )
    .cast(pl.Int8)
    .alias("converted")
)

# ---------------------------------------------------------
# Manual review overrides for ambiguous/broken
# special-teams plays
#
# These specific plays were manually reviewed using
# play descriptions and contemporaneous game reports.
# ---------------------------------------------------------

def reviewed_play(game_id, play_id):
    return (
        (pl.col("game_id") == game_id)
        &
        (pl.col("play_id") == float(play_id))
    )


# Designed fake FG pass with a bad snap.
reviewed_fake_fg_pass = (
    reviewed_play(
        "2014_08_OAK_CLE",
        243,
    )
)


# Designed fake punt runs whose execution was mishandled.
reviewed_fake_punt_run = (
    reviewed_play(
        "2018_07_CIN_KC",
        1099,
    )
    |
    reviewed_play(
        "2021_15_NYJ_MIA",
        3095,
    )
    |
    reviewed_play(
        "2022_15_CIN_TB",
        1907,
    )
    |
    reviewed_play(
        "2023_17_LAC_DEN",
        1738,
    )
    |
    reviewed_play(
        "2025_14_MIA_NYJ",
        3117,
    )
)


# Strategic safety rather than an accidental punt failure.
reviewed_intentional_safety = (
    reviewed_play(
        "2019_07_HOU_IND",
        4185,
    )
)


# ---------------------------------------------------------
# Convert execution artifact labels into intended actions
# ---------------------------------------------------------

fourth_labeled = fourth_labeled.with_columns([

    pl.when(
        reviewed_fake_fg_pass
    )
    .then(
        pl.lit("FAKE_FG_PASS")
    )

    .when(
        reviewed_fake_punt_run
    )
    .then(
        pl.lit("FAKE_PUNT_RUN")
    )

    .when(
        reviewed_intentional_safety
    )
    .then(
        pl.lit("INTENTIONAL_SAFETY")
    )

    # A broken FG snap is still a FIELD_GOAL decision.
    .when(
        (pl.col("action_initial")
         == "BROKEN_SPECIAL_TEAMS_PLAY")
        &
        fg_formation
    )
    .then(
        pl.lit("FIELD_GOAL")
    )

    # A broken punt snap is still a PUNT decision.
    .when(
        (pl.col("action_initial")
         == "BROKEN_SPECIAL_TEAMS_PLAY")
        &
        punt_formation
    )
    .then(
        pl.lit("PUNT")
    )

    .otherwise(
        pl.col("action_initial")
    )
    .alias("action"),


    # ---------------------------------------------
    # Execution quality is separate from decision.
    # ---------------------------------------------

    pl.when(
        reviewed_intentional_safety
        |
        (
            pl.col("action_initial")
            == "INTENTIONAL_SAFETY"
        )
    )
    .then(
        pl.lit("INTENTIONAL_SAFETY")
    )

    .when(
        pl.col("action_initial")
        == "BROKEN_SPECIAL_TEAMS_PLAY"
    )
    .then(
        pl.lit("BROKEN")
    )

    .otherwise(
        pl.lit("NORMAL")
    )
    .alias("execution_status"),
])

# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------

print("\nACTION COUNTS")

action_counts = (
    fourth_labeled
    .group_by("action")
    .len()
    .sort("len", descending=True)
)

print(action_counts)


print("\nFAKE PLAY CONVERSION RATES")

fake_summary = (
    fourth_labeled
    .filter(
        pl.col("action").is_in([
            "FAKE_PUNT_RUN",
            "FAKE_PUNT_PASS",
            "FAKE_FG_RUN",
            "FAKE_FG_PASS",
        ])
    )
    .group_by("action")
    .agg([
        pl.len().alias("attempts"),
        pl.col("converted").mean().alias("conversion_rate"),
        pl.col("yards_gained").mean().alias("avg_yards"),
    ])
    .sort("attempts", descending=True)
)

print(fake_summary)

print("\nEXECUTION STATUS COUNTS")

print(
    fourth_labeled
    .group_by("execution_status")
    .len()
    .sort("len", descending=True)
)

# ---------------------------------------------------------
# Save labeled dataset
# ---------------------------------------------------------

fourth_labeled.write_parquet(
    "data/fourth_downs_labeled.parquet"
)

print("\nSaved:")
print("data/fourth_downs_labeled.parquet")