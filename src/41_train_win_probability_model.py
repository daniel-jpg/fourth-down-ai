import json
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import joblib
import nflreadpy as nfl
import numpy as np
import polars as pl

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    log_loss,
    roc_auc_score,
    accuracy_score,
)


pl.Config.set_tbl_rows(100)
pl.Config.set_fmt_str_lengths(120)

Path("models").mkdir(
    exist_ok=True
)


# =========================================================
# 1. Load DEVELOPMENT seasons only.
#
# Train:      2014-2022
# Validation: 2023-2024
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
# Pregame team strength.
#
# Built using only games BEFORE the current game.
# =========================================================

team_strength = pl.read_parquet(
    "data/team_strength_pregame.parquet"
)

STRENGTH_BASE_FEATURES = [
    "off_epa_per_play_8",
    "off_success_rate_8",
    "off_pass_epa_per_play_8",
    "off_rush_epa_per_play_8",
    "def_epa_allowed_per_play_8",
    "def_success_allowed_rate_8",
    "def_pass_epa_allowed_per_play_8",
    "def_rush_epa_allowed_per_play_8",
]

TEAM_STRENGTH_FEATURES = (
    [
        f"home_{feature}"
        for feature
        in STRENGTH_BASE_FEATURES
    ]
    +
    [
        f"away_{feature}"
        for feature
        in STRENGTH_BASE_FEATURES
    ]
)

home_strength = (
    team_strength
    .select([
        "game_id",
        "team",
        *STRENGTH_BASE_FEATURES,
    ])
    .rename({
        "team": "home_team",
        **{
            feature:
                f"home_{feature}"
            for feature
            in STRENGTH_BASE_FEATURES
        },
    })
)

away_strength = (
    team_strength
    .select([
        "game_id",
        "team",
        *STRENGTH_BASE_FEATURES,
    ])
    .rename({
        "team": "away_team",
        **{
            feature:
                f"away_{feature}"
            for feature
            in STRENGTH_BASE_FEATURES
        },
    })
)

# =========================================================
# 2. Final game outcome.
#
# nflverse 'result' is the final HOME-team scoring margin.
#
# result > 0  -> home win
# result = 0  -> tie
# result < 0  -> away win
#
# We model all three outcomes so:
#
# P(home win) + P(tie) + P(away win) = 1
#
# This is cleaner than pretending ties do not exist.
# =========================================================

if "result" not in pbp.columns:

    raise RuntimeError(
        "Expected nflverse column 'result' "
        "was not found."
    )


game_outcomes = (
    pbp
    .select([
        "game_id",
        "result",
    ])
    .drop_nulls()
    .group_by("game_id")
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

        .cast(pl.Int8)
        .alias(
            "outcome_class"
        )

    )
)

# =========================================================
# Compact pregame team-strength prior.
#
# We reduce the football-only rolling team metrics to a
# single P(home win) prior.
#
# Training-game priors are CROSS-FITTED so a game's own
# final result cannot leak into its WP training feature.
# =========================================================

PRIOR_FEATURES = [
    "epa_edge",
    "success_edge",
    "pass_edge",
    "rush_edge",
]


game_prior = (
    pbp
    .select([
        "season",
        "game_id",
        "home_team",
        "away_team",
    ])
    .unique(
        subset=["game_id"]
    )
    .join(
        game_outcomes,
        on="game_id",
        how="inner",
    )
    .join(
        home_strength,
        on=[
            "game_id",
            "home_team",
        ],
        how="inner",
    )
    .join(
        away_strength,
        on=[
            "game_id",
            "away_team",
        ],
        how="inner",
    )
    .with_columns([

        (
            (
                pl.col(
                    "home_off_epa_per_play_8"
                )
                +
                pl.col(
                    "away_def_epa_allowed_per_play_8"
                )
            )
            -
            (
                pl.col(
                    "away_off_epa_per_play_8"
                )
                +
                pl.col(
                    "home_def_epa_allowed_per_play_8"
                )
            )
        )
        .alias(
            "epa_edge"
        ),

        (
            (
                pl.col(
                    "home_off_success_rate_8"
                )
                +
                pl.col(
                    "away_def_success_allowed_rate_8"
                )
            )
            -
            (
                pl.col(
                    "away_off_success_rate_8"
                )
                +
                pl.col(
                    "home_def_success_allowed_rate_8"
                )
            )
        )
        .alias(
            "success_edge"
        ),

        (
            (
                pl.col(
                    "home_off_pass_epa_per_play_8"
                )
                +
                pl.col(
                    "away_def_pass_epa_allowed_per_play_8"
                )
            )
            -
            (
                pl.col(
                    "away_off_pass_epa_per_play_8"
                )
                +
                pl.col(
                    "home_def_pass_epa_allowed_per_play_8"
                )
            )
        )
        .alias(
            "pass_edge"
        ),

        (
            (
                pl.col(
                    "home_off_rush_epa_per_play_8"
                )
                +
                pl.col(
                    "away_def_rush_epa_allowed_per_play_8"
                )
            )
            -
            (
                pl.col(
                    "away_off_rush_epa_per_play_8"
                )
                +
                pl.col(
                    "home_def_rush_epa_allowed_per_play_8"
                )
            )
        )
        .alias(
            "rush_edge"
        ),

    ])
    .sort([
        "season",
        "game_id",
    ])
)


X_prior = (
    game_prior
    .select(
        PRIOR_FEATURES
    )
    .to_numpy()
)

outcomes_prior = (
    game_prior[
        "outcome_class"
    ]
    .to_numpy()
)

seasons_prior = (
    game_prior[
        "season"
    ]
    .to_numpy()
)


# Binary prior:
# 0 = away win
# 1 = home win
#
# Ties are excluded when fitting the prior.

train_binary_idx = np.where(
    (seasons_prior <= 2022)
    &
    (outcomes_prior != 1)
)[0]

y_prior_train = (
    outcomes_prior[
        train_binary_idx
    ]
    == 2
).astype(int)


prior_probability = np.full(
    game_prior.height,
    np.nan,
)


# ---------------------------------------------------------
# Cross-fitted probabilities for 2014-2022 training games.
# ---------------------------------------------------------

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)


for fit_local, hold_local in cv.split(
    X_prior[
        train_binary_idx
    ],
    y_prior_train,
):

    fit_idx = train_binary_idx[
        fit_local
    ]

    hold_idx = train_binary_idx[
        hold_local
    ]

    fold_model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=42,
        ),
    )

    fold_model.fit(
        X_prior[
            fit_idx
        ],
        (
            outcomes_prior[
                fit_idx
            ]
            == 2
        ).astype(int),
    )

    prior_probability[
        hold_idx
    ] = (
        fold_model
        .predict_proba(
            X_prior[
                hold_idx
            ]
        )[:, 1]
    )


# ---------------------------------------------------------
# Final prior model.
#
# Fit on all 2014-2022 non-tied games.
#
# This model supplies:
# - 2023-2024 validation priors
# - priors for tied training games
# ---------------------------------------------------------

prior_model = make_pipeline(
    StandardScaler(),
    LogisticRegression(
        C=1.0,
        max_iter=1000,
        random_state=42,
    ),
)

prior_model.fit(
    X_prior[
        train_binary_idx
    ],
    y_prior_train,
)


remaining_idx = np.where(
    np.isnan(
        prior_probability
    )
)[0]


prior_probability[
    remaining_idx
] = (
    prior_model
    .predict_proba(
        X_prior[
            remaining_idx
        ]
    )[:, 1]
)


game_prior = (
    game_prior
    .with_columns(
        pl.Series(
            "pregame_home_win_prob",
            prior_probability,
        )
    )
)


# ---------------------------------------------------------
# Validate the compact prior itself.
# ---------------------------------------------------------

prior_validation = (
    game_prior
    .filter(
        pl.col("season")
        .is_in([
            2023,
            2024,
        ])
        &
        (
            pl.col(
                "outcome_class"
            )
            != 1
        )
    )
)


prior_val_y = (
    prior_validation[
        "outcome_class"
    ]
    .to_numpy()
    == 2
).astype(int)

prior_val_p = (
    prior_validation[
        "pregame_home_win_prob"
    ]
    .to_numpy()
)


print(
    "\nCOMPACT TEAM-STRENGTH PRIOR — 2023-2024"
)

print(
    f"Games:   "
    f"{prior_validation.height:,}"
)

print(
    f"Logloss: "
    f"{log_loss(prior_val_y, prior_val_p):.5f}"
)

print(
    f"Brier:   "
    f"{np.mean((prior_val_p - prior_val_y) ** 2):.5f}"
)

print(
    f"AUC:     "
    f"{roc_auc_score(prior_val_y, prior_val_p):.5f}"
)

# =========================================================
# 3. Keep states our decision engine can evaluate.
#
# We deliberately do NOT use the historical action as a
# predictor.
#
# The only play-type indicator we retain is kickoff,
# because a kickoff-pending state has different semantics
# from an ordinary scrimmage state.
# =========================================================

STATE_PLAY_TYPES = [

    "run",
    "pass",

    "punt",
    "field_goal",

    "kickoff",

    "qb_kneel",
    "qb_spike",

    "no_play",
]


states = (
    pbp
    .filter(

        pl.col("play_type")
        .is_in(
            STATE_PLAY_TYPES
        )

        &

        pl.col(
            "game_seconds_remaining"
        )
        .is_not_null()

        &

        pl.col("qtr")
        .is_not_null()

        &

        pl.col("posteam")
        .is_not_null()

        &

        pl.col("home_team")
        .is_not_null()

        &

        pl.col("away_team")
        .is_not_null()

        &

        pl.col("posteam_score")
        .is_not_null()

        &

        pl.col("defteam_score")
        .is_not_null()

    )
    .join(
        game_outcomes,
        on="game_id",
        how="inner",
    )
)

states = (
    states
    .join(
        game_prior.select([
            "game_id",
            "pregame_home_win_prob",
        ]),
        on="game_id",
        how="inner",
    )
)

# =========================================================
# 4. Convert everything into HOME-team perspective.
#
# This is crucial.
#
# The final decision engine can switch possession after
# punts, failed fourth downs, etc. Using HOME win
# probability means we never need ambiguous sign-flipping
# logic inside the ML model itself.
# =========================================================

states = states.with_columns(

    (
        pl.col("posteam")
        ==
        pl.col("home_team")
    )
    .cast(pl.Int8)
    .alias(
        "is_home_posteam"
    )

)


states = states.with_columns([

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

    (
        pl.col("play_type")
        == "kickoff"
    )
    .cast(pl.Int8)
    .alias(
        "is_kickoff"
    ),

    (
        pl.col("qtr") >= 5
    )
    .cast(pl.Int8)
    .alias(
        "is_overtime"
    ),

    (
        (
            pl.col("qtr") == 4
        )
        &
        (
            pl.col(
                "game_seconds_remaining"
            ) <= 300
        )
    )
    .cast(pl.Int8)
    .alias(
        "final_five_minutes"
    ),

])


# =========================================================
# 5. Generic state features.
#
# Null down/distance values occur on kickoffs.
#
# HistGradientBoosting handles NaN natively, so we don't
# invent fake downs or field position values.
# =========================================================

FEATURES = [
    "game_seconds_remaining",
    "half_seconds_remaining",
    "qtr",
    "down",
    "ydstogo",
    "yardline_100",
    "goal_to_go",
    "home_score_differential",
    "home_timeouts_remaining",
    "away_timeouts_remaining",
    "is_home_posteam",
    "is_kickoff",
    "is_overtime",
    "final_five_minutes",
    "pregame_home_win_prob",
    ]



# =========================================================
# 6. Split.
# =========================================================

train = states.filter(
    pl.col("season") <= 2022
)

validation = states.filter(
    pl.col("season")
    .is_in([
        2023,
        2024,
    ])
)


print("\nWP DATASET")

print(
    f"Train rows:      "
    f"{train.height:,}"
)

print(
    f"Validation rows: "
    f"{validation.height:,}"
)

print(
    f"Train games:     "
    f"{train['game_id'].n_unique():,}"
)

print(
    f"Validation games:"
    f"{validation['game_id'].n_unique():,}"
)


print("\nTRAIN GAME OUTCOMES")

print(
    train
    .select([
        "game_id",
        "outcome_class",
    ])
    .unique()
    .group_by(
        "outcome_class"
    )
    .len()
    .sort(
        "outcome_class"
    )
)


print("\nVALIDATION GAME OUTCOMES")

print(
    validation
    .select([
        "game_id",
        "outcome_class",
    ])
    .unique()
    .group_by(
        "outcome_class"
    )
    .len()
    .sort(
        "outcome_class"
    )
)


# =========================================================
# 7. Train nonlinear WP model.
#
# No hyperparameter sweep.
#
# We use a fixed conservative tree specification so we
# don't spend another dozen scripts tuning this component.
# =========================================================

X_train = (
    train
    .select(FEATURES)
    .to_numpy()
)

y_train = (
    train[
        "outcome_class"
    ]
    .to_numpy()
)


model = HistGradientBoostingClassifier(

    learning_rate=0.05,

    max_iter=300,

    max_leaf_nodes=31,

    min_samples_leaf=100,

    l2_regularization=2.0,

    early_stopping=False,

    random_state=42,
)


print("\nTraining WP model...")

model.fit(
    X_train,
    y_train,
)

print("Training complete.")


# =========================================================
# 8. Helper for outcome probabilities.
#
# Class encoding:
#
# 0 = away win
# 1 = tie
# 2 = home win
# =========================================================

class_index = {
    int(value): index
    for index, value
    in enumerate(
        model.classes_
    )
}


assert 0 in class_index
assert 2 in class_index


def get_probabilities(frame):

    X = (
        frame
        .select(FEATURES)
        .to_numpy()
    )

    probabilities = (
        model.predict_proba(X)
    )

    p_away = probabilities[
        :,
        class_index[0],
    ]


    if 1 in class_index:

        p_tie = probabilities[
            :,
            class_index[1],
        ]

    else:

        p_tie = np.zeros(
            frame.height
        )


    p_home = probabilities[
        :,
        class_index[2],
    ]


    return (
        probabilities,
        p_away,
        p_tie,
        p_home,
    )


# =========================================================
# 9. Evaluation helper.
# =========================================================

def evaluate(
    name,
    frame,
):

    if frame.height == 0:

        return


    y = (
        frame[
            "outcome_class"
        ]
        .to_numpy()
    )


    (
        probabilities,
        p_away,
        p_tie,
        p_home,
    ) = get_probabilities(
        frame
    )


    multiclass_ll = log_loss(

        y,

        probabilities,

        labels=
            model.classes_,
    )


    prediction = (
        model.classes_[
            np.argmax(
                probabilities,
                axis=1,
            )
        ]
    )


    accuracy = accuracy_score(
        y,
        prediction,
    )


    home_target = (
        y == 2
    ).astype(float)


    away_target = (
        y == 0
    ).astype(float)


    home_brier = np.mean(
        (
            p_home
            -
            home_target
        )
        ** 2
    )


    away_brier = np.mean(
        (
            p_away
            -
            away_target
        )
        ** 2
    )


    if len(
        np.unique(
            home_target
        )
    ) > 1:

        home_auc = roc_auc_score(
            home_target,
            p_home,
        )

    else:

        home_auc = float("nan")


    if len(
        np.unique(
            away_target
        )
    ) > 1:

        away_auc = roc_auc_score(
            away_target,
            p_away,
        )

    else:

        away_auc = float("nan")


    actual_home_rate = (
        home_target.mean()
    )


    actual_away_rate = (
        away_target.mean()
    )


    actual_tie_rate = (
        (y == 1)
        .mean()
    )


    print(
        f"\n{name}"
    )

    print(
        f"Rows:              "
        f"{frame.height:,}"
    )

    print(
        f"Multiclass logloss:"
        f" {multiclass_ll:.5f}"
    )

    print(
        f"Accuracy:          "
        f" {accuracy:.5f}"
    )

    print(
        f"Home Brier:        "
        f" {home_brier:.5f}"
    )

    print(
        f"Away Brier:        "
        f" {away_brier:.5f}"
    )

    print(
        f"Home AUC:          "
        f" {home_auc:.5f}"
    )

    print(
        f"Away AUC:          "
        f" {away_auc:.5f}"
    )

    print(
        f"Actual home win:   "
        f" {actual_home_rate:.5f}"
    )

    print(
        f"Pred home win:     "
        f" {p_home.mean():.5f}"
    )

    print(
        f"Actual away win:   "
        f" {actual_away_rate:.5f}"
    )

    print(
        f"Pred away win:     "
        f" {p_away.mean():.5f}"
    )

    print(
        f"Actual tie:        "
        f" {actual_tie_rate:.5f}"
    )

    print(
        f"Pred tie:          "
        f" {p_tie.mean():.5f}"
    )


# =========================================================
# 10. Validation.
# =========================================================

evaluate(
    "VALIDATION — 2023-2024",
    validation,
)


evaluate(
    "VALIDATION — 2023",
    validation.filter(
        pl.col("season")
        == 2023
    ),
)


evaluate(
    "VALIDATION — 2024",
    validation.filter(
        pl.col("season")
        == 2024
    ),
)


evaluate(
    "VALIDATION — KICKOFF STATES",
    validation.filter(
        pl.col("is_kickoff")
        == 1
    ),
)


evaluate(
    "VALIDATION — FINAL 5 MINUTES",
    validation.filter(

        (
            pl.col("qtr")
            == 4
        )

        &

        (
            pl.col(
                "game_seconds_remaining"
            ) <= 300
        )

    ),
)



# =========================================================
# 10A. Validation-only WP calibration diagnostics.
#
# IMPORTANT:
# - Uses only 2023-2024 validation data.
# - Does not modify the production model.
# - 2025 must not be used to fit or select calibration.
# =========================================================

from sklearn.linear_model import LogisticRegression


def home_wp_arrays(
    frame,
):

    (
        _,
        _,
        _,
        p_home,
    ) = get_probabilities(
        frame
    )

    y_home = (
        frame[
            "outcome_class"
        ]
        .to_numpy()
        ==
        2
    ).astype(int)

    p_home = np.clip(
        np.asarray(
            p_home,
            dtype=float,
        ),
        1e-6,
        1.0 - 1e-6,
    )

    return (
        y_home,
        p_home,
    )


def binary_wp_metrics(
    y,
    p,
):

    y = np.asarray(
        y,
        dtype=int,
    )

    p = np.clip(
        np.asarray(
            p,
            dtype=float,
        ),
        1e-9,
        1.0 - 1e-9,
    )

    return {
        "logloss":
            float(
                log_loss(
                    y,
                    p,
                    labels=[
                        0,
                        1,
                    ],
                )
            ),

        "brier":
            float(
                np.mean(
                    (p - y) ** 2
                )
            ),

        "auc":
            float(
                roc_auc_score(
                    y,
                    p,
                )
            ),

        "actual":
            float(
                np.mean(y)
            ),

        "predicted":
            float(
                np.mean(p)
            ),
    }


def print_home_wp_calibration(
    name,
    frame,
):

    y, p = home_wp_arrays(
        frame
    )

    print(
        f"\n{name}"
    )

    print(
        "probability_bin  plays  "
        "mean_predicted  actual_rate  "
        "calibration_gap"
    )

    weighted_absolute_gap = 0.0
    total_rows = 0

    for i in range(10):

        lower = (
            i / 10.0
        )

        upper = (
            (i + 1)
            /
            10.0
        )

        if i == 0:

            mask = (
                (p >= lower)
                &
                (p <= upper)
            )

        else:

            mask = (
                (p > lower)
                &
                (p <= upper)
            )

        n = int(
            np.sum(mask)
        )

        if n == 0:
            continue

        mean_predicted = float(
            np.mean(
                p[mask]
            )
        )

        actual_rate = float(
            np.mean(
                y[mask]
            )
        )

        gap = (
            actual_rate
            -
            mean_predicted
        )

        weighted_absolute_gap += (
            n
            *
            abs(gap)
        )

        total_rows += n

        label = (
            f"({lower:.1f}, {upper:.1f}]"
            if i > 0
            else
            f"[{lower:.1f}, {upper:.1f}]"
        )

        print(
            f"{label:<16} "
            f"{n:>6}  "
            f"{mean_predicted:>14.3f}  "
            f"{actual_rate:>11.3f}  "
            f"{gap:>+15.3f}"
        )

    if total_rows > 0:

        print(
            "Weighted absolute calibration error: "
            f"{weighted_absolute_gap / total_rows:.4f}"
        )


def fit_platt_calibrator(
    frame,
):

    y, p = home_wp_arrays(
        frame
    )

    logit = np.log(
        p
        /
        (1.0 - p)
    ).reshape(
        -1,
        1,
    )

    calibrator = LogisticRegression(
        C=1e6,
        solver="lbfgs",
        max_iter=1000,
    )

    calibrator.fit(
        logit,
        y,
    )

    return calibrator


def apply_platt_calibrator(
    calibrator,
    p,
):

    p = np.clip(
        np.asarray(
            p,
            dtype=float,
        ),
        1e-6,
        1.0 - 1e-6,
    )

    logit = np.log(
        p
        /
        (1.0 - p)
    ).reshape(
        -1,
        1,
    )

    return (
        calibrator
        .predict_proba(
            logit
        )[:, 1]
    )


def cross_year_wp_calibration(
    fit_year,
    test_year,
):

    fit_frame = (
        validation.filter(
            pl.col("season")
            ==
            fit_year
        )
    )

    test_frame = (
        validation.filter(
            pl.col("season")
            ==
            test_year
        )
    )

    calibrator = (
        fit_platt_calibrator(
            fit_frame
        )
    )

    y_test, raw_p = (
        home_wp_arrays(
            test_frame
        )
    )

    calibrated_p = (
        apply_platt_calibrator(
            calibrator,
            raw_p,
        )
    )

    raw_metrics = (
        binary_wp_metrics(
            y_test,
            raw_p,
        )
    )

    calibrated_metrics = (
        binary_wp_metrics(
            y_test,
            calibrated_p,
        )
    )

    intercept = float(
        calibrator
        .intercept_[0]
    )

    slope = float(
        calibrator
        .coef_[0, 0]
    )

    print(
        f"\nFIT {fit_year} -> TEST {test_year}"
    )

    print(
        f"Fit rows:       "
        f"{fit_frame.height:,}"
    )

    print(
        f"Test rows:      "
        f"{test_frame.height:,}"
    )

    print(
        f"Intercept:      "
        f"{intercept:+.5f}"
    )

    print(
        f"Logit slope:    "
        f"{slope:.5f}"
    )

    print(
        f"Actual home:    "
        f"{raw_metrics['actual']:.5f}"
    )

    print(
        f"Raw pred home:  "
        f"{raw_metrics['predicted']:.5f}"
    )

    print(
        f"Cal pred home:  "
        f"{calibrated_metrics['predicted']:.5f}"
    )

    print(
        f"Logloss:        "
        f"{raw_metrics['logloss']:.5f}"
        f" -> "
        f"{calibrated_metrics['logloss']:.5f}"
    )

    print(
        f"Brier:          "
        f"{raw_metrics['brier']:.5f}"
        f" -> "
        f"{calibrated_metrics['brier']:.5f}"
    )

    print(
        f"AUC:            "
        f"{raw_metrics['auc']:.5f}"
        f" -> "
        f"{calibrated_metrics['auc']:.5f}"
    )


print_home_wp_calibration(
    "WP HOME-WIN CALIBRATION — 2023-2024",
    validation,
)

print_home_wp_calibration(
    "WP HOME-WIN CALIBRATION — 2023",
    validation.filter(
        pl.col("season")
        ==
        2023
    ),
)

print_home_wp_calibration(
    "WP HOME-WIN CALIBRATION — 2024",
    validation.filter(
        pl.col("season")
        ==
        2024
    ),
)


print(
    "\nWP HOME-WIN CROSS-YEAR PLATT CALIBRATION"
)

cross_year_wp_calibration(
    2023,
    2024,
)

cross_year_wp_calibration(
    2024,
    2023,
)


combined_calibrator = (
    fit_platt_calibrator(
        validation
    )
)

print(
    "\nCOMBINED 2023-2024 WP DIAGNOSTIC"
)

print(
    "Intercept: "
    f"{float(combined_calibrator.intercept_[0]):+.5f}"
)

print(
    "Logit slope: "
    f"{float(combined_calibrator.coef_[0, 0]):.5f}"
)

print(
    "Diagnostic only; production WP is unchanged."
)



# =========================================================
# 10B. Current production WP calibration diagnostic.
#
# This evaluates the model actually loaded by the decision
# engine:
#
#   models/win_probability_model.joblib
#
# Calibration is fit only on 2023-2024 validation data.
#
# We calibrate the HOME-vs-AWAY conditional probability:
#
#   q = P(home) / (P(home) + P(away))
#
# and preserve the model's tie probability exactly.
# =========================================================

import json
import joblib


production_wp_model = joblib.load(
    "models/win_probability_model.joblib"
)


with open(
    "data/win_probability_spec.json",
    "r",
) as f:

    production_wp_spec = (
        json.load(f)
    )


PRODUCTION_WP_FEATURES = (
    production_wp_spec[
        "features"
    ]
)


PRODUCTION_WP_CLASS_INDEX = {

    int(value):
        index

    for index, value
    in enumerate(
        production_wp_model.classes_
    )
}


assert 0 in PRODUCTION_WP_CLASS_INDEX
assert 2 in PRODUCTION_WP_CLASS_INDEX


def production_wp_probabilities(
    frame,
):

    X = (
        frame
        .select(
            PRODUCTION_WP_FEATURES
        )
        .to_numpy()
    )

    probabilities = (
        production_wp_model
        .predict_proba(X)
    )

    p_away = probabilities[
        :,
        PRODUCTION_WP_CLASS_INDEX[0],
    ]


    if (
        1
        in
        PRODUCTION_WP_CLASS_INDEX
    ):

        p_tie = probabilities[
            :,
            PRODUCTION_WP_CLASS_INDEX[1],
        ]

    else:

        p_tie = np.zeros(
            frame.height,
            dtype=float,
        )


    p_home = probabilities[
        :,
        PRODUCTION_WP_CLASS_INDEX[2],
    ]


    return (
        probabilities,
        p_away,
        p_tie,
        p_home,
    )


def production_conditional_arrays(
    frame,
):

    (
        probabilities,
        p_away,
        p_tie,
        p_home,
    ) = production_wp_probabilities(
        frame
    )

    y = (
        frame[
            "outcome_class"
        ]
        .to_numpy()
    )

    y_home = (
        y == 2
    ).astype(int)


    non_tie_mass = np.clip(
        p_home
        +
        p_away,
        1e-9,
        1.0,
    )


    q_home = np.clip(
        p_home
        /
        non_tie_mass,
        1e-6,
        1.0 - 1e-6,
    )


    return (
        y,
        y_home,
        probabilities,
        p_away,
        p_tie,
        p_home,
        q_home,
    )


def fit_production_wp_calibrator(
    frame,
):

    (
        _,
        y_home,
        _,
        _,
        _,
        _,
        q_home,
    ) = production_conditional_arrays(
        frame
    )


    conditional_logit = np.log(
        q_home
        /
        (1.0 - q_home)
    ).reshape(
        -1,
        1,
    )


    calibrator = LogisticRegression(
        C=1e6,
        solver="lbfgs",
        max_iter=1000,
    )


    calibrator.fit(
        conditional_logit,
        y_home,
    )


    return calibrator


def apply_production_wp_calibrator(
    frame,
    calibrator,
):

    (
        y,
        _,
        raw_probabilities,
        p_away,
        p_tie,
        p_home,
        q_home,
    ) = production_conditional_arrays(
        frame
    )


    conditional_logit = np.log(
        q_home
        /
        (1.0 - q_home)
    ).reshape(
        -1,
        1,
    )


    calibrated_q = (
        calibrator
        .predict_proba(
            conditional_logit
        )[:, 1]
    )


    non_tie_mass = np.clip(
        1.0
        -
        p_tie,
        0.0,
        1.0,
    )


    calibrated_home = (
        non_tie_mass
        *
        calibrated_q
    )


    calibrated_away = (
        non_tie_mass
        *
        (
            1.0
            -
            calibrated_q
        )
    )


    calibrated_probabilities = (
        raw_probabilities.copy()
    )


    calibrated_probabilities[
        :,
        PRODUCTION_WP_CLASS_INDEX[0],
    ] = calibrated_away


    if (
        1
        in
        PRODUCTION_WP_CLASS_INDEX
    ):

        calibrated_probabilities[
            :,
            PRODUCTION_WP_CLASS_INDEX[1],
        ] = p_tie


    calibrated_probabilities[
        :,
        PRODUCTION_WP_CLASS_INDEX[2],
    ] = calibrated_home


    return (
        y,
        raw_probabilities,
        calibrated_probabilities,
    )


def production_wp_metrics(
    y,
    probabilities,
):

    p_away = probabilities[
        :,
        PRODUCTION_WP_CLASS_INDEX[0],
    ]

    p_home = probabilities[
        :,
        PRODUCTION_WP_CLASS_INDEX[2],
    ]


    y_home = (
        y == 2
    ).astype(float)


    y_away = (
        y == 0
    ).astype(float)


    return {

        "multiclass_logloss":
            float(
                log_loss(
                    y,
                    probabilities,
                    labels=
                        production_wp_model
                        .classes_,
                )
            ),

        "home_brier":
            float(
                np.mean(
                    (
                        p_home
                        -
                        y_home
                    )
                    ** 2
                )
            ),

        "away_brier":
            float(
                np.mean(
                    (
                        p_away
                        -
                        y_away
                    )
                    ** 2
                )
            ),

        "home_auc":
            float(
                roc_auc_score(
                    y_home,
                    p_home,
                )
            ),

        "actual_home":
            float(
                y_home.mean()
            ),

        "pred_home":
            float(
                p_home.mean()
            ),

        "pred_away":
            float(
                p_away.mean()
            ),
    }


def production_cross_year_calibration(
    fit_year,
    test_year,
):

    fit_frame = (
        validation.filter(
            pl.col("season")
            ==
            fit_year
        )
    )


    test_frame = (
        validation.filter(
            pl.col("season")
            ==
            test_year
        )
    )


    calibrator = (
        fit_production_wp_calibrator(
            fit_frame
        )
    )


    (
        y,
        raw_probabilities,
        calibrated_probabilities,
    ) = (
        apply_production_wp_calibrator(
            test_frame,
            calibrator,
        )
    )


    raw = production_wp_metrics(
        y,
        raw_probabilities,
    )


    calibrated = production_wp_metrics(
        y,
        calibrated_probabilities,
    )


    probability_sums = (
        calibrated_probabilities
        .sum(axis=1)
    )


    print(
        f"\nFIT {fit_year} -> TEST {test_year}"
    )

    print(
        f"Fit rows:       "
        f"{fit_frame.height:,}"
    )

    print(
        f"Test rows:      "
        f"{test_frame.height:,}"
    )

    print(
        "Intercept:      "
        f"{float(calibrator.intercept_[0]):+.5f}"
    )

    print(
        "Logit slope:    "
        f"{float(calibrator.coef_[0, 0]):.5f}"
    )

    print(
        "Actual home:    "
        f"{raw['actual_home']:.5f}"
    )

    print(
        "Raw pred home:  "
        f"{raw['pred_home']:.5f}"
    )

    print(
        "Cal pred home:  "
        f"{calibrated['pred_home']:.5f}"
    )

    print(
        "Multiclass LL:  "
        f"{raw['multiclass_logloss']:.5f}"
        " -> "
        f"{calibrated['multiclass_logloss']:.5f}"
    )

    print(
        "Home Brier:     "
        f"{raw['home_brier']:.5f}"
        " -> "
        f"{calibrated['home_brier']:.5f}"
    )

    print(
        "Away Brier:     "
        f"{raw['away_brier']:.5f}"
        " -> "
        f"{calibrated['away_brier']:.5f}"
    )

    print(
        "Home AUC:       "
        f"{raw['home_auc']:.5f}"
        " -> "
        f"{calibrated['home_auc']:.5f}"
    )

    print(
        "Max sum error:  "
        f"{float(np.max(np.abs(probability_sums - 1.0))):.3e}"
    )


print(
    "\nCURRENT PRODUCTION WP — "
    "CROSS-YEAR CONDITIONAL PLATT CALIBRATION"
)


production_cross_year_calibration(
    2023,
    2024,
)


production_cross_year_calibration(
    2024,
    2023,
)


production_combined_calibrator = (
    fit_production_wp_calibrator(
        validation
    )
)


print(
    "\nCURRENT PRODUCTION WP — "
    "COMBINED 2023-2024 DIAGNOSTIC"
)

print(
    "Intercept: "
    f"{float(production_combined_calibrator.intercept_[0]):+.5f}"
)

print(
    "Logit slope: "
    f"{float(production_combined_calibrator.coef_[0, 0]):.5f}"
)

print(
    "Diagnostic only; production WP is unchanged."
)


# =========================================================
# 11. Simple directional sanity test.
#
# Same late-game state, only home score margin changes.
#
# Home WP should increase as the home team's margin moves:
#
# -7 -> 0 -> +7
# =========================================================

sanity_prior = float(
    train[
        "pregame_home_win_prob"
    ].mean()
)

sanity_rows = []


for margin in [
    -7,
    0,
    7,
]:

    sanity_rows.append({

        "game_seconds_remaining":
            120.0,

        "half_seconds_remaining":
            120.0,

        "qtr":
            4.0,

        "down":
            1.0,

        "ydstogo":
            10.0,

        "yardline_100":
            50.0,

        "goal_to_go":
            0.0,

        "home_score_differential":
            float(margin),

        "home_timeouts_remaining":
            2.0,

        "away_timeouts_remaining":
            2.0,

        "is_home_posteam":
            1.0,

        "is_kickoff":
            0.0,

        "is_overtime":
            0.0,

        "final_five_minutes":
            1.0,
        "pregame_home_win_prob":
        sanity_prior,

    })


sanity = pl.DataFrame(
    sanity_rows
)


(
    sanity_probabilities,
    sanity_away,
    sanity_tie,
    sanity_home,
) = get_probabilities(
    sanity
)


print(
    "\nSANITY CHECK — HOME WP VS SCORE MARGIN"
)


for margin, probability in zip(
    [-7, 0, 7],
    sanity_home,
):

    print(
        f"Home margin {margin:+}: "
        f"P(home win) = "
        f"{probability:.4f}"
    )


# =========================================================
# 12. Save.
# =========================================================

joblib.dump(
    model,
    "models/win_probability_model_team_strength_candidate.joblib",
)


spec = {

    "training_seasons":
        "2014-2022",

    "validation_seasons":
        "2023-2024",

    "test_season":
        2025,

    "target_classes": {

        "0":
            "away_win",

        "1":
            "tie",

        "2":
            "home_win",

    },

    "features":
        FEATURES,

    "model":
        (
            "HistGradientBoostingClassifier"
        ),

    "perspective":
        "home_team",

    "decision_value_policy":
        (
            "for a home-team decision use "
            "P(home_win); for an away-team "
            "decision use P(away_win)"
        ),

    "kickoff_policy":
        (
            "kickoff states are explicitly "
            "represented using is_kickoff=1"
        ),
    "team_strength":
    "rolling prior-8-game football-only features, shifted one game",

}


with open(
    "data/win_probability_spec_team_strength_candidate.json",
    "w",
) as f:

    json.dump(
        spec,
        f,
        indent=2,
    )


print("\nSAVED")

print(
    "models/win_probability_model_team_strength_candidate.joblib"
)

print(
    "data/win_probability_spec_team_strength_candidate.json"
)