from pathlib import Path

import nflreadpy as nfl
import polars as pl


# =========================================================
# Configuration
# =========================================================

SEASONS = list(range(2013, 2026))

# 2013 is used only to provide history for early 2014.
OUTPUT_START_SEASON = 2014

ROLLING_GAMES = 8

OUTPUT_PATH = Path(
    "data/team_strength_pregame.parquet"
)


# =========================================================
# Helpers
# =========================================================

def canonical_team_expr(column):
    """
    Keep franchise history together across relocations.
    The original team code is still preserved separately.
    """

    return (
        pl.when(pl.col(column) == "OAK")
        .then(pl.lit("LV"))
        .when(pl.col(column) == "SD")
        .then(pl.lit("LAC"))
        .when(pl.col(column) == "STL")
        .then(pl.lit("LA"))
        .otherwise(pl.col(column))
    )


def safe_ratio(
    numerator,
    denominator,
    output_name,
):
    return (
        pl.when(
            pl.col(denominator) > 0
        )
        .then(
            pl.col(numerator)
            /
            pl.col(denominator)
        )
        .otherwise(None)
        .alias(output_name)
    )


# =========================================================
# Load play-by-play
# =========================================================

print(
    "Loading 2013-2025 PBP..."
)

pbp = nfl.load_pbp(
    SEASONS
)

print(
    f"Loaded {len(pbp):,} PBP rows."
)


# =========================================================
# Keep ordinary offensive plays
# =========================================================

plays = pbp.filter(
    pl.col("posteam").is_not_null()
    &
    pl.col("defteam").is_not_null()
    &
    pl.col("epa").is_not_null()
    &
    pl.col("play_type").is_in(
        ["run", "pass"]
    )
)


if "qb_kneel" in plays.columns:

    plays = plays.filter(
        pl.col("qb_kneel")
        .cast(
            pl.Int8,
            strict=False,
        )
        .fill_null(0)
        ==
        0
    )


if "qb_spike" in plays.columns:

    plays = plays.filter(
        pl.col("qb_spike")
        .cast(
            pl.Int8,
            strict=False,
        )
        .fill_null(0)
        ==
        0
    )


plays = plays.with_columns([

    canonical_team_expr(
        "posteam"
    ).alias(
        "posteam_key"
    ),

    canonical_team_expr(
        "defteam"
    ).alias(
        "defteam_key"
    ),

    (
        pl.col("epa") > 0
    )
    .cast(pl.Int8)
    .alias(
        "success"
    ),

])


print(
    f"Eligible offensive plays: "
    f"{len(plays):,}"
)


# =========================================================
# Team-game offense
# =========================================================

offense = (
    plays
    .group_by([
        "season",
        "week",
        "game_id",
        "posteam",
        "posteam_key",
    ])
    .agg([

        pl.len()
        .alias(
            "off_plays"
        ),

        pl.col("epa")
        .sum()
        .alias(
            "off_epa_sum"
        ),

        pl.col("success")
        .sum()
        .alias(
            "off_successes"
        ),

        pl.col("epa")
        .filter(
            pl.col("play_type")
            ==
            "pass"
        )
        .sum()
        .alias(
            "off_pass_epa_sum"
        ),

        pl.when(
            pl.col("play_type")
            ==
            "pass"
        )
        .then(1)
        .otherwise(0)
        .sum()
        .alias(
            "off_pass_plays"
        ),

        pl.col("epa")
        .filter(
            pl.col("play_type")
            ==
            "run"
        )
        .sum()
        .alias(
            "off_rush_epa_sum"
        ),

        pl.when(
            pl.col("play_type")
            ==
            "run"
        )
        .then(1)
        .otherwise(0)
        .sum()
        .alias(
            "off_rush_plays"
        ),

    ])
    .rename({
        "posteam":
            "team",

        "posteam_key":
            "team_key",
    })
)


# =========================================================
# Team-game defense
# =========================================================

defense = (
    plays
    .group_by([
        "season",
        "week",
        "game_id",
        "defteam",
        "defteam_key",
    ])
    .agg([

        pl.len()
        .alias(
            "def_plays"
        ),

        pl.col("epa")
        .sum()
        .alias(
            "def_epa_allowed_sum"
        ),

        pl.col("success")
        .sum()
        .alias(
            "def_successes_allowed"
        ),

        pl.col("epa")
        .filter(
            pl.col("play_type")
            ==
            "pass"
        )
        .sum()
        .alias(
            "def_pass_epa_allowed_sum"
        ),

        pl.when(
            pl.col("play_type")
            ==
            "pass"
        )
        .then(1)
        .otherwise(0)
        .sum()
        .alias(
            "def_pass_plays"
        ),

        pl.col("epa")
        .filter(
            pl.col("play_type")
            ==
            "run"
        )
        .sum()
        .alias(
            "def_rush_epa_allowed_sum"
        ),

        pl.when(
            pl.col("play_type")
            ==
            "run"
        )
        .then(1)
        .otherwise(0)
        .sum()
        .alias(
            "def_rush_plays"
        ),

    ])
    .rename({

        "defteam":
            "team_def",

        "defteam_key":
            "team_key",

    })
)


# =========================================================
# One row per team-game
# =========================================================

team_games = (
    offense
    .join(
        defense,
        on=[
            "season",
            "week",
            "game_id",
            "team_key",
        ],
        how="inner",
    )
    .drop(
        "team_def"
    )
    .sort([
        "team_key",
        "season",
        "week",
        "game_id",
    ])
)


print(
    f"Team-game rows: "
    f"{len(team_games):,}"
)


# =========================================================
# Rolling PRE-GAME totals
#
# shift(1) is critical:
# the current game's stats cannot enter its own features.
# =========================================================

ROLL_COLUMNS = [

    "off_plays",
    "off_epa_sum",
    "off_successes",

    "off_pass_epa_sum",
    "off_pass_plays",

    "off_rush_epa_sum",
    "off_rush_plays",

    "def_plays",
    "def_epa_allowed_sum",
    "def_successes_allowed",

    "def_pass_epa_allowed_sum",
    "def_pass_plays",

    "def_rush_epa_allowed_sum",
    "def_rush_plays",

]


rolling_expressions = []

for column in ROLL_COLUMNS:

    rolling_expressions.append(

        pl.col(column)
        .shift(1)
        .rolling_sum(
            window_size=
                ROLLING_GAMES,
            min_samples=1,
        )
        .over(
            "team_key"
        )
        .alias(
            f"prior8_{column}"
        )

    )


strength = (
    team_games
    .with_columns(
        rolling_expressions
    )
    .with_columns(

        (
            pl.col("game_id")
            .cum_count()
            .over(
                "team_key"
            )
            -
            1
        )
        .alias(
            "prior_games"
        )

    )
)


# =========================================================
# Convert totals to football-strength features
# =========================================================

strength = strength.with_columns([

    safe_ratio(
        "prior8_off_epa_sum",
        "prior8_off_plays",
        "off_epa_per_play_8",
    ),

    safe_ratio(
        "prior8_off_successes",
        "prior8_off_plays",
        "off_success_rate_8",
    ),

    safe_ratio(
        "prior8_off_pass_epa_sum",
        "prior8_off_pass_plays",
        "off_pass_epa_per_play_8",
    ),

    safe_ratio(
        "prior8_off_rush_epa_sum",
        "prior8_off_rush_plays",
        "off_rush_epa_per_play_8",
    ),

    safe_ratio(
        "prior8_def_epa_allowed_sum",
        "prior8_def_plays",
        "def_epa_allowed_per_play_8",
    ),

    safe_ratio(
        "prior8_def_successes_allowed",
        "prior8_def_plays",
        "def_success_allowed_rate_8",
    ),

    safe_ratio(
        "prior8_def_pass_epa_allowed_sum",
        "prior8_def_pass_plays",
        "def_pass_epa_allowed_per_play_8",
    ),

    safe_ratio(
        "prior8_def_rush_epa_allowed_sum",
        "prior8_def_rush_plays",
        "def_rush_epa_allowed_per_play_8",
    ),

])


# =========================================================
# Final table
# =========================================================

FEATURES = [

    "off_epa_per_play_8",
    "off_success_rate_8",

    "off_pass_epa_per_play_8",
    "off_rush_epa_per_play_8",

    "def_epa_allowed_per_play_8",
    "def_success_allowed_rate_8",

    "def_pass_epa_allowed_per_play_8",
    "def_rush_epa_allowed_per_play_8",

]


output = (
    strength
    .filter(
        pl.col("season")
        >=
        OUTPUT_START_SEASON
    )
    .select([

        "season",
        "week",
        "game_id",

        "team",
        "team_key",

        "prior_games",

        "prior8_off_plays",
        "prior8_def_plays",

        *FEATURES,

    ])
    .sort([
        "season",
        "week",
        "game_id",
        "team",
    ])
)


# =========================================================
# Sanity checks
# =========================================================

duplicate_rows = (
    output
    .group_by([
        "game_id",
        "team",
    ])
    .len()
    .filter(
        pl.col("len") != 1
    )
)


if len(duplicate_rows) != 0:

    raise RuntimeError(
        "Duplicate team-game rows found."
    )


print(
    "\nOUTPUT"
)

print(
    f"Rows:  {len(output):,}"
)

print(
    f"Games: "
    f"{output['game_id'].n_unique():,}"
)


print(
    "\nFEATURE NULL COUNTS"
)

print(
    output.select([
        pl.col(feature)
        .null_count()
        .alias(feature)
        for feature
        in FEATURES
    ])
)


print(
    "\nFEATURE MEANS"
)

print(
    output.select([
        pl.col(feature)
        .mean()
        .alias(feature)
        for feature
        in FEATURES
    ])
)


print(
    "\nPRIOR-GAME HISTORY"
)

print(
    output.select([

        pl.col(
            "prior_games"
        )
        .min()
        .alias(
            "min_prior_games"
        ),

        pl.col(
            "prior_games"
        )
        .median()
        .alias(
            "median_prior_games"
        ),

        pl.col(
            "prior_games"
        )
        .max()
        .alias(
            "max_prior_games"
        ),

    ])
)


# =========================================================
# Save
# =========================================================

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

output.write_parquet(
    OUTPUT_PATH
)


print(
    "\nSAVED"
)

print(
    OUTPUT_PATH
)