import json
from pathlib import Path

import joblib
import nflreadpy as nfl
import numpy as np
import polars as pl

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)


pl.Config.set_tbl_rows(100)
pl.Config.set_fmt_str_lengths(120)

Path("models").mkdir(
    exist_ok=True
)


# =========================================================
# 1. Load punt audit from Scripts 37-38.
# =========================================================

punts = (
    pl.read_parquet(
        "data/punt_outcome_audit.parquet"
    )
    .sort([
        "game_id",
        "play_id",
    ])
)

print(
    f"Loaded {punts.height:,} punt decisions."
)


# =========================================================
# 2. Load development PBP ONLY.
#
# 2025 remains untouched.
# =========================================================

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


# =========================================================
# 3. Attach exact SAME-PLAY punt metadata.
#
# Do not infer score / possession from a later play.
#
# play_id is not reliably chronological across every game,
# so the punt play itself is the source of truth for:
#
# - TD / safety scoring
# - lost fumbles
# - multi-fumble recovery sequence
# =========================================================

punt_meta = (
    pbp
    .filter(
        pl.col("punt_attempt")
        .fill_null(0)
        ==
        1
    )
    .select([
        "game_id",
        "play_id",

        "fixed_drive",

        "posteam_score",
        "defteam_score",
        "posteam_score_post",
        "defteam_score_post",

        "fumble_lost",

        "fumbled_1_team",
        "fumble_recovery_1_team",

        "fumbled_2_team",
        "fumble_recovery_2_team",

        "touchdown",
        "td_team",
        "safety",
    ])
    .rename({
        "fixed_drive":
            "pbp_fixed_drive",

        "posteam_score":
            "same_play_posteam_score_before",

        "defteam_score":
            "same_play_defteam_score_before",

        "posteam_score_post":
            "same_play_posteam_score_after",

        "defteam_score_post":
            "same_play_defteam_score_after",

        "fumble_lost":
            "same_play_fumble_lost",

        "fumbled_1_team":
            "same_play_fumbled_1_team",

        "fumble_recovery_1_team":
            "same_play_fumble_recovery_1_team",

        "fumbled_2_team":
            "same_play_fumbled_2_team",

        "fumble_recovery_2_team":
            "same_play_fumble_recovery_2_team",

        "touchdown":
            "same_play_touchdown",

        "td_team":
            "same_play_td_team",

        "safety":
            "same_play_safety",
    })
)


before_meta_join = punts.height

punts = punts.join(
    punt_meta,
    on=[
        "game_id",
        "play_id",
    ],
    how="left",
)

assert (
    punts.height
    ==
    before_meta_join
)


# =========================================================
# 4. Reconstruct next offensive state by FIXED DRIVE.
#
# fixed_drive + 1 is far safer than assuming a larger
# play_id means a later football state.
#
# We still validate the resulting state before allowing it
# into the field-position / clock model.
# =========================================================

drive_states = (
    pbp
    .filter(
        pl.col("fixed_drive")
        .is_not_null()
        &
        pl.col("posteam")
        .is_not_null()
        &
        pl.col("down")
        .is_not_null()
        &
        pl.col("yardline_100")
        .is_not_null()
    )
    .select([
        "game_id",
        "fixed_drive",

        "play_id",
        "qtr",
        "game_seconds_remaining",

        "posteam",
        "defteam",

        "down",
        "ydstogo",
        "yardline_100",

        "play_type",
        "desc",
    ])
    .sort(
        [
            "game_id",
            "fixed_drive",
            "qtr",
            "game_seconds_remaining",
            "play_id",
        ],
        descending=[
            False,
            False,
            False,
            True,
            False,
        ],
    )
    .group_by(
        [
            "game_id",
            "fixed_drive",
        ],
        maintain_order=True,
    )
    .agg([
        pl.col("play_id")
        .first()
        .alias("state_play_id"),

        pl.col("qtr")
        .first()
        .alias("state_qtr"),

        pl.col(
            "game_seconds_remaining"
        )
        .first()
        .alias(
            "state_game_seconds_remaining"
        ),

        pl.col("posteam")
        .first()
        .alias("state_posteam"),

        pl.col("defteam")
        .first()
        .alias("state_defteam"),

        pl.col("down")
        .first()
        .alias("state_down"),

        pl.col("ydstogo")
        .first()
        .alias("state_ydstogo"),

        pl.col("yardline_100")
        .first()
        .alias("state_yardline_100"),

        pl.col("play_type")
        .first()
        .alias("state_play_type"),

        pl.col("desc")
        .first()
        .alias("state_desc"),
    ])
    .rename({
        "fixed_drive":
            "target_fixed_drive"
    })
)


punts = (
    punts
    .with_columns(
        (
            pl.col("pbp_fixed_drive")
            + 1.0
        )
        .alias(
            "target_fixed_drive"
        )
    )
    .join(
        drive_states,
        on=[
            "game_id",
            "target_fixed_drive",
        ],
        how="left",
    )
)


# =========================================================
# 5. Same-play score and final loose-ball recovery.
# =========================================================

punts = punts.with_columns([

    (
        (
            pl.col(
                "same_play_posteam_score_after"
            )
            -
            pl.col(
                "same_play_defteam_score_after"
            )
        )
        -
        (
            pl.col(
                "same_play_posteam_score_before"
            )
            -
            pl.col(
                "same_play_defteam_score_before"
            )
        )
    )
    .alias(
        "same_play_score_change"
    ),

    pl.when(
        pl.col(
            "same_play_fumble_lost"
        )
        .fill_null(0)
        ==
        1
    )
    .then(
        pl.coalesce([
            pl.col(
                "same_play_fumble_recovery_2_team"
            ),

            # A second fumble that goes out of bounds stays
            # with the second fumbler.
            pl.col(
                "same_play_fumbled_2_team"
            ),

            pl.col(
                "same_play_fumble_recovery_1_team"
            ),
        ])
    )
    .otherwise(
        pl.lit(
            None,
            dtype=pl.String,
        )
    )
    .alias(
        "final_recovery_team"
    ),

    (
        pl.col(
            "game_seconds_remaining"
        )
        -
        pl.col(
            "state_game_seconds_remaining"
        )
    )
    .alias(
        "seconds_to_state"
    ),

])


# score_change now means score ON THE PUNT PLAY itself.
punts = punts.with_columns(
    pl.col(
        "same_play_score_change"
    )
    .alias(
        "score_change"
    )
)


# =========================================================
# 6. Period / state validation.
# =========================================================

punts = punts.with_columns([

    (
        (pl.col("qtr") == 2)
        &
        (pl.col("state_qtr") == 3)
    )
    .fill_null(False)
    .alias(
        "crossed_halftime"
    ),

    (
        (pl.col("qtr") == 4)
        &
        (pl.col("state_qtr") >= 5)
    )
    .fill_null(False)
    .alias(
        "entered_overtime"
    ),

    (
        (
            pl.col("state_qtr")
            ==
            pl.col("qtr")
        )
        |
        (
            (pl.col("qtr") == 1)
            &
            (pl.col("state_qtr") == 2)
        )
        |
        (
            (pl.col("qtr") == 3)
            &
            (pl.col("state_qtr") == 4)
        )
    )
    .fill_null(False)
    .alias(
        "valid_period_transition"
    ),

])


punts = punts.with_columns(
    (
        pl.col("state_play_id")
        .is_not_null()
        &
        pl.col(
            "valid_period_transition"
        )
        &
        pl.col(
            "seconds_to_state"
        )
        .is_not_null()
        &
        (
            pl.col(
                "seconds_to_state"
            )
            >=
            0
        )
        &
        (
            pl.col(
                "seconds_to_state"
            )
            <=
            30
        )
    )
    .fill_null(False)
    .alias(
        "state_chronological"
    )
)


# =========================================================
# 7. Physical punt branch.
# =========================================================

punts = punts.with_columns(

    pl.when(
        pl.col("execution_status")
        ==
        "BROKEN"
    )
    .then(
        pl.lit("BROKEN")
    )

    .when(
        pl.col("pbp_punt_blocked")
        .fill_null(0)
        ==
        1
    )
    .then(
        pl.lit("BLOCKED")
    )

    .otherwise(
        pl.lit("REGULAR")
    )

    .alias(
        "punt_branch"
    )

)


# =========================================================
# 8. Transition class.
#
# Scoring comes from the punt play itself.
#
# For an ordinary punt:
# - absent a lost return-team fumble, the receiving team
#   gets the ball;
# - if the return team loses the ball, use the FINAL
#   recovery on the play.
#
# Blocked / broken punts may not populate normal fumble
# fields, so only use the next-drive state when that state
# passes the chronology guard.
# =========================================================

punts = punts.with_columns(

    pl.when(
        pl.col("score_change")
        <
        0
    )
    .then(
        pl.lit(
            "opponent_scored"
        )
    )

    .when(
        pl.col("score_change")
        >
        0
    )
    .then(
        pl.lit(
            "kicking_team_scored"
        )
    )

    .when(
        pl.col("state_play_id")
        .is_null()
    )
    .then(
        pl.lit(
            "no_later_state"
        )
    )

    .when(
        pl.col(
            "crossed_halftime"
        )
    )
    .then(
        pl.lit(
            "crossed_halftime"
        )
    )

    .when(
        pl.col(
            "entered_overtime"
        )
    )
    .then(
        pl.lit(
            "entered_overtime"
        )
    )

    # Ordinary punt where receiving team lost the ball
    # and the kicking team finished with it.
    .when(
        (
            pl.col("punt_branch")
            ==
            "REGULAR"
        )
        &
        (
            pl.col(
                "same_play_fumble_lost"
            )
            .fill_null(0)
            ==
            1
        )
        &
        (
            pl.col(
                "final_recovery_team"
            )
            ==
            pl.col("posteam")
        )
    )
    .then(
        pl.lit(
            "kicking_team_ball_no_score"
        )
    )

    # All other non-scoring regular punts belong to the
    # receiving team.
    .when(
        pl.col("punt_branch")
        ==
        "REGULAR"
    )
    .then(
        pl.lit(
            "opponent_ball_no_score"
        )
    )

    # Blocked / broken punt: trust next-drive possession
    # only if the state passed the chronology guard.
    .when(
        pl.col(
            "state_chronological"
        )
        &
        (
            pl.col("state_posteam")
            ==
            pl.col("defteam")
        )
    )
    .then(
        pl.lit(
            "opponent_ball_no_score"
        )
    )

    .when(
        pl.col(
            "state_chronological"
        )
        &
        (
            pl.col("state_posteam")
            ==
            pl.col("posteam")
        )
    )
    .then(
        pl.lit(
            "kicking_team_ball_no_score"
        )
    )

    .otherwise(
        pl.lit("other")
    )

    .alias(
        "transition_class"
    )

)


# =========================================================
# 8B. Is the reconstructed state safe to use?
# =========================================================

punts = punts.with_columns(

    pl.when(
        pl.col(
            "transition_class"
        )
        ==
        "opponent_ball_no_score"
    )
    .then(
        pl.col(
            "state_chronological"
        )
        &
        (
            pl.col("state_posteam")
            ==
            pl.col("defteam")
        )
    )

    .when(
        pl.col(
            "transition_class"
        )
        ==
        "kicking_team_ball_no_score"
    )
    .then(
        pl.col(
            "state_chronological"
        )
        &
        (
            pl.col("state_posteam")
            ==
            pl.col("posteam")
        )
    )

    .when(
        pl.col(
            "transition_class"
        )
        .is_in([
            "opponent_scored",
            "kicking_team_scored",
        ])
    )
    .then(
        pl.col(
            "state_chronological"
        )
    )

    .otherwise(
        pl.lit(False)
    )

    .fill_null(False)
    .alias(
        "state_usable"
    )

)




# =========================================================
# Validation-only punt distribution audit.
#
# Development split:
#   train      = 2014-2022
#   validation = 2023-2024
#   test       = 2025
#
# 2025 is intentionally untouched.
#
# Audit the production ordinary-punt procedure:
#
#   point prediction
#   +
#   sampled OOF residual from similar starting-field bucket
#
# Rare branches are audited separately against their frozen
# development probabilities.
# =========================================================

import importlib.util
import numpy as np
import pandas as pd


SIMULATIONS_PER_ROW = 500
RANDOM_SEED = 510051


# ---------------------------------------------------------
# Load the actual production engine so residual selection
# and field clipping exactly match production.
# ---------------------------------------------------------

engine_spec = (
    importlib.util.spec_from_file_location(
        "decision_engine",
        "src/42_build_decision_engine.py",
    )
)

engine = (
    importlib.util.module_from_spec(
        engine_spec
    )
)

engine_spec.loader.exec_module(
    engine
)


validation = punts.filter(
    pl.col("split")
    ==
    "validation"
)


ordinary_val = (
    validation
    .filter(
        (
            pl.col("punt_branch")
            ==
            "REGULAR"
        )
        &
        (
            pl.col("transition_class")
            ==
            "opponent_ball_no_score"
        )
        &
        pl.col("state_usable")
    )
)


print(
    "\nPUNT VALIDATION DISTRIBUTION AUDIT"
)

print(
    f"Validation punt decisions: "
    f"{validation.height:,}"
)

print(
    f"Ordinary usable validation punts: "
    f"{ordinary_val.height:,}"
)

print(
    f"Simulations per ordinary punt: "
    f"{SIMULATIONS_PER_ROW:,}"
)


# ---------------------------------------------------------
# Production point prediction.
# ---------------------------------------------------------

FEATURES = list(
    engine.PUNT_FEATURES
)


pdf = ordinary_val.to_pandas()


X = (
    pdf[
        FEATURES
    ]
    .to_numpy()
)


observed = (
    pd.to_numeric(
        pdf[
            "state_yardline_100"
        ],
        errors="coerce",
    )
    .to_numpy(
        dtype=float
    )
)


point_prediction = (
    engine.punt_model.predict(
        X
    )
)

point_prediction = (
    np.clip(
        point_prediction,
        0.0,
        100.0,
    )
)


point_error = (
    point_prediction
    -
    observed
)


print(
    "\nORDINARY FIELD-POSITION POINT MODEL"
)

print(
    f"MAE:  "
    f"{np.mean(np.abs(point_error)):.3f} yards"
)

print(
    f"RMSE: "
    f"{np.sqrt(np.mean(point_error ** 2)):.3f} yards"
)

print(
    f"Observed mean receiving yardline_100: "
    f"{np.mean(observed):.3f}"
)

print(
    f"Predicted mean receiving yardline_100: "
    f"{np.mean(point_prediction):.3f}"
)


# ---------------------------------------------------------
# Starting-field-position buckets.
#
# Same broad football interpretation used in the punt audit.
# ---------------------------------------------------------

def field_bucket(
    yardline,
):

    yardline = float(
        yardline
    )

    if yardline >= 80:
        return "own_1_20"

    if yardline >= 60:
        return "own_21_40"

    if yardline >= 40:
        return "midfield"

    if yardline >= 20:
        return "opp_21_40"

    return "opp_1_20"


FIELD_BUCKET_ORDER = [
    "own_1_20",
    "own_21_40",
    "midfield",
    "opp_21_40",
    "opp_1_20",
]


# ---------------------------------------------------------
# Match production residual donor selection.
# ---------------------------------------------------------

pool = engine.punt_pool.copy()


available_buckets = np.array(
    sorted(
        pool[
            "start_yardline_bucket"
        ]
        .dropna()
        .unique()
    )
)


def donor_candidates(
    start_yardline,
):

    start_bucket = int(
        np.floor(
            float(
                start_yardline
            )
            /
            10.0
        )
        *
        10
    )


    candidate = pool[
        pool[
            "start_yardline_bucket"
        ]
        ==
        start_bucket
    ]


    if len(candidate) == 0:

        nearest = (
            available_buckets[
                np.argmin(
                    np.abs(
                        available_buckets
                        -
                        start_bucket
                    )
                )
            ]
        )

        candidate = pool[
            pool[
                "start_yardline_bucket"
            ]
            ==
            nearest
        ]


    return candidate


# ---------------------------------------------------------
# Generate production-matched predictive distribution.
# ---------------------------------------------------------

rng = np.random.default_rng(
    RANDOM_SEED
)


all_simulated = []

observed_by_bucket = {
    bucket: []
    for bucket in FIELD_BUCKET_ORDER
}

simulated_by_bucket = {
    bucket: []
    for bucket in FIELD_BUCKET_ORDER
}


row_records = []


for i, row in pdf.iterrows():

    start_yardline = float(
        row[
            "yardline_100"
        ]
    )

    bucket = field_bucket(
        start_yardline
    )


    prediction = float(
        point_prediction[i]
    )


    donors = (
        donor_candidates(
            start_yardline
        )
    )


    sampled_indices = (
        rng.integers(
            0,
            len(donors),
            size=
                SIMULATIONS_PER_ROW,
        )
    )


    sampled = (
        donors.iloc[
            sampled_indices
        ]
        .reset_index(
            drop=True
        )
    )


    raw_draws = (
        prediction
        +
        sampled[
            "yardline_residual"
        ]
        .to_numpy(
            dtype=float
        )
    )


    draws = np.fromiter(
        (
            engine.clip_yardline(
                value
            )
            for value
            in raw_draws
        ),
        dtype=float,
        count=
            SIMULATIONS_PER_ROW,
    )


    # Preserve the discrete NFL punt-touchback state.
    sampled_touchback = (
        pd.to_numeric(
            sampled[
                "pbp_touchback"
            ],
            errors="coerce",
        )
        .fillna(
            0.0
        )
        .to_numpy(
            dtype=float
        )
        >=
        0.5
    )


    draws[
        sampled_touchback
    ] = 80.0


    actual = float(
        observed[i]
    )


    observed_by_bucket[
        bucket
    ].append(
        actual
    )

    simulated_by_bucket[
        bucket
    ].append(
        draws
    )

    all_simulated.append(
        draws
    )


    q10 = float(
        np.quantile(
            draws,
            0.10,
        )
    )

    q90 = float(
        np.quantile(
            draws,
            0.90,
        )
    )


    row_records.append({

        "season":
            int(
                row[
                    "season"
                ]
            ),

        "game_id":
            row[
                "game_id"
            ],

        "play_id":
            row[
                "play_id"
            ],

        "start_yardline_100":
            start_yardline,

        "field_bucket":
            bucket,

        "observed_yardline_100":
            actual,

        "point_prediction":
            prediction,

        "simulated_mean":
            float(
                np.mean(
                    draws
                )
            ),

        "simulated_median":
            float(
                np.median(
                    draws
                )
            ),

        "simulated_p10":
            q10,

        "simulated_p90":
            q90,

        "observed_in_10_90_interval":
            bool(
                actual >= q10
                and
                actual <= q90
            ),
    })


results = pd.DataFrame(
    row_records
)


all_simulated = (
    np.concatenate(
        all_simulated
    )
)


# ---------------------------------------------------------
# Distribution helpers.
#
# Receiving-team yardline_100:
#
#   >80 => inside own 20
#   >90 => inside own 10
#   >95 => inside own 5
#
# Around 80 is reported separately as an own-20-yard-line
# mass diagnostic because production models field position,
# not an explicit touchback class.
# ---------------------------------------------------------

def distribution_stats(
    values,
):

    values = np.asarray(
        values,
        dtype=float,
    )


    return {

        "mean":
            float(
                np.mean(
                    values
                )
            ),

        "median":
            float(
                np.median(
                    values
                )
            ),

        "p10":
            float(
                np.quantile(
                    values,
                    0.10,
                )
            ),

        "p90":
            float(
                np.quantile(
                    values,
                    0.90,
                )
            ),

        "inside20_rate":
            float(
                np.mean(
                    values > 80.0
                )
            ),

        "inside10_rate":
            float(
                np.mean(
                    values > 90.0
                )
            ),

        "inside5_rate":
            float(
                np.mean(
                    values > 95.0
                )
            ),

        "own20_band_rate":
            float(
                np.mean(
                    np.abs(
                        values
                        -
                        80.0
                    )
                    <=
                    0.5
                )
            ),
    }


observed_stats = (
    distribution_stats(
        observed
    )
)

simulated_stats = (
    distribution_stats(
        all_simulated
    )
)


print(
    "\nORDINARY POST-PUNT DISTRIBUTION"
)

print(
    "                           "
    "Observed     Simulated"
)

for key, label in [
    (
        "mean",
        "Mean yardline_100",
    ),
    (
        "median",
        "Median yardline_100",
    ),
    (
        "p10",
        "P10 yardline_100",
    ),
    (
        "p90",
        "P90 yardline_100",
    ),
]:

    print(
        f"{label:<26}"
        f"{observed_stats[key]:>9.3f}"
        f"{simulated_stats[key]:>14.3f}"
    )


for key, label in [
    (
        "inside20_rate",
        "Inside own 20",
    ),
    (
        "inside10_rate",
        "Inside own 10",
    ),
    (
        "inside5_rate",
        "Inside own 5",
    ),
    (
        "own20_band_rate",
        "Own-20 +/-0.5 yd",
    ),
]:

    print(
        f"{label:<26}"
        f"{100.0 * observed_stats[key]:>8.2f}%"
        f"{100.0 * simulated_stats[key]:>13.2f}%"
    )


coverage = float(
    results[
        "observed_in_10_90_interval"
    ]
    .mean()
)


print(
    "\nObserved target inside simulated "
    "10th-90th percentile interval: "
    f"{100.0 * coverage:.2f}%"
)


# ---------------------------------------------------------
# Conditional distribution by original field position.
# ---------------------------------------------------------

bucket_rows = []


for bucket in FIELD_BUCKET_ORDER:

    obs_values = np.asarray(
        observed_by_bucket[
            bucket
        ],
        dtype=float,
    )


    sim_parts = (
        simulated_by_bucket[
            bucket
        ]
    )


    if (
        len(obs_values) == 0
        or
        len(sim_parts) == 0
    ):

        continue


    sim_values = np.concatenate(
        sim_parts
    )


    obs_stats = (
        distribution_stats(
            obs_values
        )
    )

    sim_stats = (
        distribution_stats(
            sim_values
        )
    )


    bucket_rows.append({

        "field_bucket":
            bucket,

        "n":
            len(
                obs_values
            ),

        "obs_mean":
            obs_stats[
                "mean"
            ],

        "sim_mean":
            sim_stats[
                "mean"
            ],

        "obs_inside20":
            obs_stats[
                "inside20_rate"
            ],

        "sim_inside20":
            sim_stats[
                "inside20_rate"
            ],

        "obs_inside10":
            obs_stats[
                "inside10_rate"
            ],

        "sim_inside10":
            sim_stats[
                "inside10_rate"
            ],

        "obs_inside5":
            obs_stats[
                "inside5_rate"
            ],

        "sim_inside5":
            sim_stats[
                "inside5_rate"
            ],
    })


bucket_table = pd.DataFrame(
    bucket_rows
)


print(
    "\nBY STARTING FIELD POSITION"
)

if len(bucket_table) > 0:

    printable = (
        bucket_table.copy()
    )

    for column in [
        "obs_inside20",
        "sim_inside20",
        "obs_inside10",
        "sim_inside10",
        "obs_inside5",
        "sim_inside5",
    ]:

        printable[
            column
        ] *= 100.0


    print(
        printable.to_string(
            index=False,
            formatters={

                "obs_mean":
                    "{:.2f}".format,

                "sim_mean":
                    "{:.2f}".format,

                "obs_inside20":
                    "{:.2f}%".format,

                "sim_inside20":
                    "{:.2f}%".format,

                "obs_inside10":
                    "{:.2f}%".format,

                "sim_inside10":
                    "{:.2f}%".format,

                "obs_inside5":
                    "{:.2f}%".format,

                "sim_inside5":
                    "{:.2f}%".format,
            },
        )
    )


# ---------------------------------------------------------
# Observed touchback rate.
#
# The production ordinary model does not explicitly sample a
# touchback class. Its effect enters through final field
# position, so compare this only as a descriptive diagnostic.
# ---------------------------------------------------------

if "pbp_touchback" in pdf.columns:

    touchback = (
        pd.to_numeric(
            pdf[
                "pbp_touchback"
            ],
            errors="coerce",
        )
        .fillna(
            0.0
        )
        .to_numpy(
            dtype=float
        )
    )


    print(
        "\nTOUCHBACK DIAGNOSTIC"
    )

    print(
        "Observed ordinary-punt touchback rate: "
        f"{100.0 * np.mean(touchback >= 0.5):.2f}%"
    )

    print(
        "Simulated mass within +/-0.5 yd of own 20: "
        f"{100.0 * simulated_stats['own20_band_rate']:.2f}%"
    )


# ---------------------------------------------------------
# Rare branch rates.
# ---------------------------------------------------------

validation_n = (
    validation.height
)


broken_rate = (
    validation
    .filter(
        pl.col(
            "punt_branch"
        )
        ==
        "BROKEN"
    )
    .height
    /
    validation_n
)


normal_validation = (
    validation
    .filter(
        pl.col(
            "execution_status"
        )
        ==
        "NORMAL"
    )
)


if (
    "pbp_punt_blocked"
    in
    normal_validation.columns
):

    blocked_rate = (
        normal_validation
        .filter(
            pl.col(
                "pbp_punt_blocked"
            )
            .fill_null(
                0
            )
            ==
            1
        )
        .height
        /
        normal_validation.height
    )

else:

    blocked_rate = np.nan


print(
    "\nRARE BRANCH RATES"
)

print(
    "Broken punt:"
)

print(
    f"  production/train = "
    f"{100.0 * engine.punt_spec['broken_probability']:.3f}%"
)

print(
    f"  validation       = "
    f"{100.0 * broken_rate:.3f}%"
)

print(
    "Blocked | normal execution:"
)

print(
    f"  production/train = "
    f"{100.0 * engine.punt_spec['blocked_probability_given_normal_execution']:.3f}%"
)

print(
    f"  validation       = "
    f"{100.0 * blocked_rate:.3f}%"
)


# ---------------------------------------------------------
# Regular transition rates.
# ---------------------------------------------------------

regular_validation = (
    validation
    .filter(
        pl.col(
            "punt_branch"
        )
        ==
        "REGULAR"
    )
)


actual_transition_rates = {

    row[
        "transition_class"
    ]:
        row[
            "len"
        ]
        /
        regular_validation.height

    for row in (
        regular_validation
        .group_by(
            "transition_class"
        )
        .len()
        .iter_rows(
            named=True
        )
    )
}


production_transition_rates = (
    engine.punt_spec[
        "regular_transition_probabilities"
    ]
)


transition_classes = sorted(
    set(
        actual_transition_rates
    )
    |
    set(
        production_transition_rates
    )
)


transition_rows = []


for transition in transition_classes:

    transition_rows.append({

        "transition":
            transition,

        "production_pct":
            100.0
            *
            production_transition_rates.get(
                transition,
                0.0,
            ),

        "validation_pct":
            100.0
            *
            actual_transition_rates.get(
                transition,
                0.0,
            ),
    })


print(
    "\nREGULAR TRANSITION RATES"
)

print(
    pd.DataFrame(
        transition_rows
    )
    .to_string(
        index=False,
        formatters={
            "production_pct":
                "{:.3f}%".format,
            "validation_pct":
                "{:.3f}%".format,
        },
    )
)


print(
    "\nIMPORTANT"
)

print(
    "This audit uses only 2023-2024 validation data."
)

print(
    "2025 remains held out and is not used to tune "
    "the punt model."
)

print(
    "The simulated field-position distribution matches "
    "the production residual-sampling procedure."
)


output_path = (
    "data/"
    "evaluation_validation_punt_distribution.parquet"
)


results.to_parquet(
    output_path,
    index=False,
)


print(
    f"\nSaved: {output_path}"
)
