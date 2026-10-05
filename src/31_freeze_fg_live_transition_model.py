import json

import polars as pl


pl.Config.set_tbl_rows(100)


# ---------------------------------------------------------
# Load cleaned FG immediate-state audit.
# ---------------------------------------------------------

df = pl.read_parquet(
    "data/fg_failed_immediate_state_audit.parquet"
)


# ---------------------------------------------------------
# Training blocked FGs only.
#
# Broken FG executions will BORROW this model because
# their sample size is far too small.
# ---------------------------------------------------------

blocks = (
    df
    .filter(
        (pl.col("season") <= 2022)
        &
        (pl.col("fg_outcome") == "blocked")
    )
)


# ---------------------------------------------------------
# Reconstruct transition class defensively.
# ---------------------------------------------------------

blocks = blocks.with_columns(

    pl.when(
        pl.col("score_change") < 0
    )
    .then(
        pl.lit("opponent_scored")
    )

    .when(
        pl.col("score_change") > 0
    )
    .then(
        pl.lit("original_offense_scored")
    )

    .when(
        pl.col("state_posteam")
        == pl.col("defteam")
    )
    .then(
        pl.lit("opponent_ball_no_score")
    )

    .when(
        pl.col("state_posteam")
        == pl.col("posteam")
    )
    .then(
        pl.lit("original_ball_no_score")
    )

    .otherwise(
        pl.lit("other")
    )

    .alias("transition_class")

)


# ---------------------------------------------------------
# Halftime transitions are NOT part of the blocked-FG
# live-ball transition distribution.
#
# They will be handled separately by the game-clock /
# halftime state logic.
# ---------------------------------------------------------

model_rows = blocks.filter(
    ~pl.col("crossed_halftime")
    &
    pl.col("state_play_id").is_not_null()
)


print(
    "\nBLOCKED FG MODEL ROWS"
)

print(
    f"All training blocks: {blocks.height}"
)

print(
    f"Usable same-half blocks: {model_rows.height}"
)


# ---------------------------------------------------------
# Transition counts.
# ---------------------------------------------------------

counts = (
    model_rows
    .group_by(
        "transition_class"
    )
    .agg(
        pl.len()
        .alias("plays")
    )
    .sort(
        "plays",
        descending=True,
    )
)


total = counts[
    "plays"
].sum()


rates = counts.with_columns(
    (
        pl.col("plays")
        /
        total
    )
    .alias("probability")
)


print(
    "\nFINAL BLOCKED-FG TRANSITION RATES"
)

print(
    rates
)


# ---------------------------------------------------------
# We expect exactly two genuine branches:
#
# 1. opponent ball, no score
# 2. opponent scored
#
# Original-team possession in the earlier audit was caused
# by crossing halftime.
# ---------------------------------------------------------

rate_dict = {
    row["transition_class"]:
        row["plays"]
    for row in counts.iter_rows(
        named=True
    )
}


assert (
    rate_dict.get(
        "opponent_ball_no_score",
        0,
    )
    == 149
)


assert (
    rate_dict.get(
        "opponent_scored",
        0,
    )
    == 5
)


assert (
    rate_dict.get(
        "original_ball_no_score",
        0,
    )
    == 0
)


assert (
    rate_dict.get(
        "other",
        0,
    )
    == 0
)


assert total == 154


# ---------------------------------------------------------
# Ordinary blocked-FG transition pool.
#
# Preserve residual + elapsed time JOINTLY.
#
# Later simulation should sample an entire row, not sample
# yardline and time independently.
# ---------------------------------------------------------

ordinary = (
    model_rows
    .filter(
        pl.col("transition_class")
        == "opponent_ball_no_score"
    )
    .with_columns(
        (
            pl.col("state_yardline_100")
            -
            pl.col("miss_rule_yardline")
        )
        .alias("yardline_residual")
    )
)


print(
    "\nORDINARY BLOCK POOL"
)

print(
    ordinary.select([

        pl.len()
        .alias("plays"),

        pl.col("yardline_residual")
        .mean()
        .alias("mean_residual"),

        pl.col("yardline_residual")
        .median()
        .alias("median_residual"),

        pl.col("seconds_to_state")
        .mean()
        .alias("mean_seconds"),

        pl.col("seconds_to_state")
        .median()
        .alias("median_seconds"),

    ])
)


ordinary.select([
    "game_id",
    "play_id",
    "season",

    "yardline_100",
    "miss_rule_yardline",
    "yardline_residual",

    "seconds_to_state",

    "state_yardline_100",
]).write_parquet(
    "data/fg_block_transition_pool.parquet"
)


# ---------------------------------------------------------
# Scoring branch.
#
# Freeze only the score change here.
#
# Do NOT use seconds_to_state as touchdown-return duration:
# it can include PAT / kickoff / later administration.
# ---------------------------------------------------------

scored = model_rows.filter(
    pl.col("transition_class")
    == "opponent_scored"
)


print(
    "\nBLOCKED-FG SCORING BRANCH"
)

print(
    scored
    .group_by(
        "score_change"
    )
    .len()
    .sort(
        "score_change"
    )
)


assert scored.height == 5

assert (
    scored[
        "score_change"
    ]
    .unique()
    .to_list()
    == [-7.0]
)


# ---------------------------------------------------------
# Save transition rates.
# ---------------------------------------------------------

rates.write_parquet(
    "data/fg_live_transition_rates.parquet"
)


# ---------------------------------------------------------
# Save simple metadata / model specification.
# ---------------------------------------------------------

metadata = {

    "training_seasons":
        "2014-2022",

    "blocked_training_plays":
        159,

    "same_half_model_plays":
        154,

    "opponent_ball_no_score_count":
        149,

    "opponent_scored_count":
        5,

    "opponent_ball_no_score_probability":
        149 / 154,

    "opponent_scored_probability":
        5 / 154,

    "broken_fg_policy":
        "borrow_blocked_fg_transition_model",

    "ordinary_transition_policy":
        "sample_joint_yardline_residual_and_elapsed_time",

    "scoring_policy":
        "opponent_td_branch_then_generic_score_kickoff_handler",

    "halftime_policy":
        "handled_outside_live_failure_model",

}


with open(
    "data/fg_live_transition_model.json",
    "w",
) as f:

    json.dump(
        metadata,
        f,
        indent=2,
    )


print(
    "\nSAVED"
)

print(
    "data/fg_live_transition_rates.parquet"
)

print(
    "data/fg_block_transition_pool.parquet"
)

print(
    "data/fg_live_transition_model.json"
)