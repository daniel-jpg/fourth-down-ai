from pathlib import Path

import nflreadpy as nfl
import polars as pl


# =========================================================
# Configuration
# =========================================================

SEASONS = list(
    range(
        2014,
        2026,
    )
)

OUTPUT_PATH = (
    "data/overtime_states.parquet"
)

Path("data").mkdir(
    exist_ok=True
)


# =========================================================
# Load PBP
# =========================================================

print(
    "\nLoading 2014-2025 PBP..."
)

pbp = (
    nfl.load_pbp(
        SEASONS
    )
    .sort([
        "game_id",
        "play_id",
    ])
)

print(
    f"Loaded {pbp.height:,} plays."
)


# =========================================================
# Resolve column names
# =========================================================

if (
    "fixed_drive"
    in pbp.columns
):

    drive_col = (
        "fixed_drive"
    )

elif (
    "drive"
    in pbp.columns
):

    drive_col = (
        "drive"
    )

else:

    raise RuntimeError(
        "Could not find fixed_drive or drive."
    )


if (
    "season_type"
    in pbp.columns
):

    game_type_col = (
        "season_type"
    )

elif (
    "game_type"
    in pbp.columns
):

    game_type_col = (
        "game_type"
    )

else:

    raise RuntimeError(
        "Could not find season_type or game_type."
    )


required = [
    "season",
    "game_id",
    "play_id",
    "qtr",
    "posteam",
    "home_team",
    "away_team",
    "posteam_score",
    "defteam_score",
    "posteam_timeouts_remaining",
    "defteam_timeouts_remaining",
    "quarter_seconds_remaining",
    "yardline_100",
    "ydstogo",
    "goal_to_go",
    "down",
    "play_type",
    "result",
    drive_col,
    game_type_col,
]

missing = [
    col
    for col in required
    if col not in pbp.columns
]

if missing:

    raise RuntimeError(
        "Missing required columns: "
        +
        ", ".join(
            missing
        )
    )


# =========================================================
# Final game outcome
# =========================================================

game_outcomes = (
    pbp
    .select([
        "game_id",
        "result",
    ])
    .drop_nulls()
    .group_by(
        "game_id"
    )
    .agg(
        pl.col("result")
        .last()
        .alias(
            "final_home_margin"
        )
    )
    .with_columns(

        pl.when(
            pl.col(
                "final_home_margin"
            ) > 0
        )
        .then(
            pl.lit(2)
        )

        .when(
            pl.col(
                "final_home_margin"
            ) < 0
        )
        .then(
            pl.lit(0)
        )

        .otherwise(
            pl.lit(1)
        )

        .cast(
            pl.Int8
        )
        .alias(
            "outcome_class"
        )
    )
)


# =========================================================
# Overtime rows
# =========================================================

ot = (
    pbp
    .filter(
        (
            pl.col("qtr")
            >= 5
        )
        &
        pl.col(
            "posteam"
        )
        .is_not_null()
    )
)


# =========================================================
# Identify offensive possessions
#
# Use scrimmage downs to avoid treating kickoff rows as
# separate offensive possessions.
# =========================================================

ot_drives = (
    ot
    .filter(
        pl.col(
            drive_col
        )
        .is_not_null()
        &
        pl.col(
            "down"
        )
        .is_not_null()
    )
    .select([
        "game_id",
        drive_col,
        "posteam",
    ])
    .unique()
    .sort([
        "game_id",
        drive_col,
    ])
    .with_columns(

        pl.col(
            drive_col
        )
        .rank(
            method="dense"
        )
        .over(
            "game_id"
        )
        .cast(
            pl.Int16
        )
        .alias(
            "ot_possession_index"
        )
    )
)


first_ot_team = (
    ot_drives
    .filter(
        pl.col(
            "ot_possession_index"
        ) == 1
    )
    .select([
        "game_id",

        pl.col(
            "posteam"
        )
        .alias(
            "ot_first_posteam"
        ),
    ])
)


ot_drives = (
    ot_drives
    .join(
        first_ot_team,
        on="game_id",
        how="left",
    )
)


# =========================================================
# Keep model-relevant OT states
# =========================================================

STATE_PLAY_TYPES = [
    "run",
    "pass",
    "punt",
    "field_goal",
    "qb_kneel",
    "qb_spike",
    "no_play",
]


states = (
    ot
    .filter(
        pl.col(
            "play_type"
        )
        .is_in(
            STATE_PLAY_TYPES
        )
    )
    .join(
        ot_drives,
        on=[
            "game_id",
            drive_col,
            "posteam",
        ],
        how="left",
    )
    .join(
        game_outcomes,
        on="game_id",
        how="inner",
    )
)


# =========================================================
# Rule regime
# =========================================================

is_postseason_expr = (
    pl.col(
        game_type_col
    )
    !=
    pl.lit(
        "REG"
    )
)


states = (
    states
    .with_columns([

        is_postseason_expr
        .cast(
            pl.Int8
        )
        .alias(
            "is_postseason"
        ),

        (
            (
                (~is_postseason_expr)
                &
                (
                    pl.col("season")
                    >= 2025
                )
            )
            |
            (
                is_postseason_expr
                &
                (
                    pl.col("season")
                    >= 2022
                )
            )
        )
        .cast(
            pl.Int8
        )
        .alias(
            "both_teams_guaranteed"
        ),

    ])
)


# =========================================================
# Possession phase
# =========================================================

states = (
    states
    .with_columns([

        pl.when(
            pl.col(
                "ot_possession_index"
            ) == 1
        )
        .then(
            pl.lit(
                "OPENING_POSSESSION"
            )
        )

        .when(
            pl.col(
                "ot_possession_index"
            ) == 2
        )
        .then(
            pl.lit(
                "SECOND_POSSESSION"
            )
        )

        .otherwise(
            pl.lit(
                "SUDDEN_DEATH"
            )
        )
        .alias(
            "ot_phase"
        ),

        pl.when(
            pl.col(
                "ot_possession_index"
            ) <= 2
        )
        .then(
            pl.col(
                "ot_possession_index"
            )
        )
        .otherwise(
            pl.lit(3)
        )
        .cast(
            pl.Int8
        )
        .alias(
            "ot_phase_code"
        ),

        (
            pl.col(
                "posteam"
            )
            ==
            pl.col(
                "ot_first_posteam"
            )
        )
        .cast(
            pl.Int8
        )
        .alias(
            "is_first_ot_team"
        ),

        pl.col(
            "quarter_seconds_remaining"
        )
        .alias(
            "ot_seconds_remaining"
        ),

    ])
)


# =========================================================
# Historical OT period length
#
# Regular season:
#   2014-2016 -> 15 minutes
#   2017+     -> 10 minutes
#
# Postseason:
#   15-minute periods
# =========================================================

states = (
    states
    .with_columns(

        pl.when(
            pl.col(
                "is_postseason"
            ) == 1
        )
        .then(
            pl.lit(900)
        )

        .when(
            pl.col("season")
            >= 2017
        )
        .then(
            pl.lit(600)
        )

        .otherwise(
            pl.lit(900)
        )

        .cast(
            pl.Int16
        )
        .alias(
            "ot_period_length_seconds"
        )
    )
)


# =========================================================
# Home-team perspective
# =========================================================

states = (
    states
    .with_columns(

        (
            pl.col(
                "posteam"
            )
            ==
            pl.col(
                "home_team"
            )
        )
        .cast(
            pl.Int8
        )
        .alias(
            "is_home_posteam"
        )
    )
)


states = (
    states
    .with_columns([

        pl.when(
            pl.col(
                "is_home_posteam"
            ) == 1
        )
        .then(
            pl.col(
                "posteam_score"
            )
            -
            pl.col(
                "defteam_score"
            )
        )
        .otherwise(
            pl.col(
                "defteam_score"
            )
            -
            pl.col(
                "posteam_score"
            )
        )
        .alias(
            "home_score_differential"
        ),

        pl.when(
            pl.col(
                "is_home_posteam"
            ) == 1
        )
        .then(
            pl.col(
                "posteam_timeouts_remaining"
            )
        )
        .otherwise(
            pl.col(
                "defteam_timeouts_remaining"
            )
        )
        .alias(
            "home_timeouts_remaining"
        ),

        pl.when(
            pl.col(
                "is_home_posteam"
            ) == 1
        )
        .then(
            pl.col(
                "defteam_timeouts_remaining"
            )
        )
        .otherwise(
            pl.col(
                "posteam_timeouts_remaining"
            )
        )
        .alias(
            "away_timeouts_remaining"
        ),

    ])
)


# =========================================================
# Remove states whose possession could not be identified
# =========================================================

unresolved = (
    states
    .filter(
        pl.col(
            "ot_possession_index"
        )
        .is_null()
    )
    .height
)

states = (
    states
    .filter(
        pl.col(
            "ot_possession_index"
        )
        .is_not_null()
    )
)


# =========================================================
# Save
# =========================================================

states.write_parquet(
    OUTPUT_PATH
)


# =========================================================
# Diagnostics
# =========================================================

print(
    "\nOVERTIME STATE DATASET"
)

print(
    f"Rows:  {states.height:,}"
)

print(
    f"Games: "
    f"{states['game_id'].n_unique():,}"
)

print(
    f"Unresolved rows dropped: "
    f"{unresolved:,}"
)


print(
    "\nOT PHASE"
)

print(
    states
    .group_by(
        "ot_phase"
    )
    .len()
    .sort(
        "ot_phase"
    )
)


print(
    "\nRULE REGIME"
)

print(
    states
    .group_by([
        "both_teams_guaranteed",
        "is_postseason",
    ])
    .len()
    .sort([
        "both_teams_guaranteed",
        "is_postseason",
    ])
)


print(
    "\nPHASE BY RULE REGIME"
)

print(
    states
    .group_by([
        "both_teams_guaranteed",
        "ot_phase",
    ])
    .len()
    .sort([
        "both_teams_guaranteed",
        "ot_phase",
    ])
)


print(
    "\nSEASON / GAME TYPE"
)

print(
    states
    .group_by([
        "season",
        game_type_col,
    ])
    .len()
    .sort([
        "season",
        game_type_col,
    ])
)


print(
    "\nSAVED"
)

print(
    OUTPUT_PATH
)
