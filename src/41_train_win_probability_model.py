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