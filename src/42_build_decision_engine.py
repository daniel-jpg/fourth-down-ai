import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# =========================================================
# Configuration
# =========================================================

N_SIMULATIONS = 4000
RANDOM_SEED = 42
# Fake-punt / fake-FG effects are observational and based
# on very small samples. Keep their modeling infrastructure
# available, but do not allow them to drive the production
# recommendation until we have stronger causal evidence.
ALLOW_FAKE_RECOMMENDATIONS = False

ACTIONS = [

    "NORMAL_GO_RUN",
    "NORMAL_GO_PASS",

    "PUNT",
    "FIELD_GOAL",

    "FAKE_PUNT_RUN",
    "FAKE_PUNT_PASS",

    "FAKE_FG_RUN",
    "FAKE_FG_PASS",
]


GO_ACTIONS = {

    "NORMAL_GO_RUN": {
        "is_pass": 0,
        "fake_punt": 0,
        "fake_fg": 0,
    },

    "NORMAL_GO_PASS": {
        "is_pass": 1,
        "fake_punt": 0,
        "fake_fg": 0,
    },

    "FAKE_PUNT_RUN": {
        "is_pass": 0,
        "fake_punt": 1,
        "fake_fg": 0,
    },

    "FAKE_PUNT_PASS": {
        "is_pass": 1,
        "fake_punt": 1,
        "fake_fg": 0,
    },

    "FAKE_FG_RUN": {
        "is_pass": 0,
        "fake_punt": 0,
        "fake_fg": 1,
    },

    "FAKE_FG_PASS": {
        "is_pass": 1,
        "fake_punt": 0,
        "fake_fg": 1,
    },
}


DISPLAY_NAMES = {

    "NORMAL_GO_RUN":
        "GO — RUN",

    "NORMAL_GO_PASS":
        "GO — PASS/DROPBACK",

    "PUNT":
        "PUNT",

    "FIELD_GOAL":
        "FIELD GOAL",

    "FAKE_PUNT_RUN":
        "FAKE PUNT — RUN",

    "FAKE_PUNT_PASS":
        "FAKE PUNT — PASS",

    "FAKE_FG_RUN":
        "FAKE FG — RUN",

    "FAKE_FG_PASS":
        "FAKE FG — PASS",
}


# =========================================================
# Required artifacts
# =========================================================

REQUIRED = [

    "models/fake_conversion_model.joblib",

    "models/field_goal_outcome_model.joblib",

    "models/punt_field_position_model.joblib",

    "models/win_probability_model.joblib",

    "data/go_shared_transition_pool.parquet",

    "data/punt_ordinary_transition_pool.parquet",

    "data/punt_rare_transition_pool.parquet",

    "data/punt_transition_spec.json",

    "data/fg_made_clock_transition_pool.parquet",

    "data/fg_block_transition_pool.parquet",

    "data/fg_live_transition_rates.parquet",

    "data/win_probability_spec.json",

    "data/fourth_down_modeling_split.parquet",
]


for path in REQUIRED:

    if not Path(path).exists():

        raise FileNotFoundError(
            f"Missing required artifact: {path}"
        )


# =========================================================
# Load models
# =========================================================

print("Loading fourth-down decision engine...")


go_model = joblib.load(
    "models/fake_conversion_model.joblib"
)


fg_bundle = joblib.load(
    "models/field_goal_outcome_model.joblib"
)


punt_model = joblib.load(
    "models/punt_field_position_model.joblib"
)


wp_model = joblib.load(
    "models/win_probability_model.joblib"
)


# =========================================================
# Load transition pools
# =========================================================

go_pool = pd.read_parquet(
    "data/go_shared_transition_pool.parquet"
)


punt_pool = pd.read_parquet(
    "data/punt_ordinary_transition_pool.parquet"
)


punt_rare = pd.read_parquet(
    "data/punt_rare_transition_pool.parquet"
)


fg_clock_pool = pd.read_parquet(
    "data/fg_made_clock_transition_pool.parquet"
)


fg_block_pool = pd.read_parquet(
    "data/fg_block_transition_pool.parquet"
)


fg_live_rates = pd.read_parquet(
    "data/fg_live_transition_rates.parquet"
)


with open(
    "data/punt_transition_spec.json",
    "r",
) as f:

    punt_spec = json.load(f)


with open(
    "data/win_probability_spec.json",
    "r",
) as f:

    wp_spec = json.load(f)

# =========================================================
# Empirical support for fake special-teams plays.
#
# Fakes are extremely rare and heavily selected.
# We therefore only allow a fake to be recommended inside
# the central 80% of DEVELOPMENT field-position support.
#
# 2025 test data is explicitly excluded.
# =========================================================

support_data = pd.read_parquet(
    "data/fourth_down_modeling_split.parquet"
)


fake_support_source = support_data[

    (support_data["split"] != "test")

    &

    support_data["action"].isin([
        "FAKE_PUNT_RUN",
        "FAKE_PUNT_PASS",
        "FAKE_FG_RUN",
        "FAKE_FG_PASS",
    ])

].copy()


FAKE_SUPPORT = {}


for action in [
    "FAKE_PUNT_RUN",
    "FAKE_PUNT_PASS",
    "FAKE_FG_RUN",
    "FAKE_FG_PASS",
]:

    sample = fake_support_source[
        fake_support_source["action"]
        ==
        action
    ]


    FAKE_SUPPORT[action] = {

        "n":
            int(len(sample)),

        "yardline_low":
            float(
                sample[
                    "yardline_100"
                ].quantile(0.10)
            ),

        "yardline_high":
            float(
                sample[
                    "yardline_100"
                ].quantile(0.90)
            ),
        "ydstogo_low":
    float(
        sample[
            "ydstogo"
        ].quantile(0.10)
    ),

    "ydstogo_high":
        float(
            sample[
                "ydstogo"
            ].quantile(0.90)
        ),
    }


print("\nFAKE ACTION DEVELOPMENT SUPPORT")

for action, support in (
    FAKE_SUPPORT.items()
):

    print(
    f"{action:16s} "
    f"n={support['n']:3d}  "
    f"yardline_100="
    f"{support['yardline_low']:.1f}"
    f" to "
    f"{support['yardline_high']:.1f}  "
    f"ydstogo="
    f"{support['ydstogo_low']:.1f}"
    f" to "
    f"{support['ydstogo_high']:.1f}"
)

WP_FEATURES = wp_spec[
    "features"
]


# =========================================================
# Detect FG block-pool columns robustly.
# =========================================================

BLOCK_RESIDUAL_COLUMN = None

for column in fg_block_pool.columns:

    if "resid" in column.lower():

        BLOCK_RESIDUAL_COLUMN = column
        break


if BLOCK_RESIDUAL_COLUMN is None:

    raise RuntimeError(
        "Could not find residual column in "
        "fg_block_transition_pool.parquet. "
        f"Columns: {list(fg_block_pool.columns)}"
    )


BLOCK_CLOCK_COLUMN = None

for column in fg_block_pool.columns:

    lowered = column.lower()

    if (
        "second" in lowered
        or
        "clock" in lowered
    ):

        BLOCK_CLOCK_COLUMN = column
        break


if BLOCK_CLOCK_COLUMN is None:

    raise RuntimeError(
        "Could not find clock column in "
        "fg_block_transition_pool.parquet. "
        f"Columns: {list(fg_block_pool.columns)}"
    )


print("All models and transition pools loaded.")


# =========================================================
# Helpers
# =========================================================

def finite(value):

    try:

        return bool(
            np.isfinite(
                float(value)
            )
        )

    except (
        TypeError,
        ValueError,
    ):

        return False


def safe_float(
    value,
    default=0.0,
):

    if finite(value):

        return float(value)

    return float(default)


def clip_yardline(value):

    return float(
        np.clip(
            value,
            0.5,
            99.5,
        )
    )


# =========================================================
# Game-clock helpers
# =========================================================

def derive_half_seconds(
    qtr,
    game_seconds_remaining,
):

    if qtr <= 2:

        return max(
            0.0,
            game_seconds_remaining
            - 1800.0,
        )

    return max(
        0.0,
        game_seconds_remaining,
    )


def advance_clock(
    base,
    elapsed_seconds,
):

    qtr = int(
        base["qtr"]
    )

    game_seconds = float(
        base[
            "game_seconds_remaining"
        ]
    )

    elapsed = max(
        0.0,
        float(elapsed_seconds),
    )


    if qtr == 1:

        lower_boundary = 2700.0

    elif qtr == 2:

        lower_boundary = 1800.0

    elif qtr == 3:

        lower_boundary = 900.0

    else:

        lower_boundary = 0.0


    proposed = (
        game_seconds
        -
        elapsed
    )


    if (
        proposed
        <
        lower_boundary
    ):

        new_game_seconds = (
            lower_boundary
        )

        if qtr < 4:

            new_qtr = (
                qtr + 1
            )

        else:

            new_qtr = qtr

    else:

        new_game_seconds = (
            proposed
        )

        new_qtr = qtr


    half_seconds = (
        derive_half_seconds(
            new_qtr,
            new_game_seconds,
        )
    )


    return (
        new_qtr,
        new_game_seconds,
        half_seconds,
    )


# =========================================================
# Normalize a user-supplied fourth-down state.
#
# score_differential is from the CURRENT OFFENSE'S
# perspective.
# =========================================================

def normalize_state(state):

    required = [

        "qtr",

        "game_seconds_remaining",

        "yardline_100",

        "ydstogo",

        "score_differential",

        "is_home",
    ]


    missing = [

        key
        for key in required
        if key not in state
    ]


    if missing:

        raise ValueError(
            "Missing state fields: "
            + ", ".join(missing)
        )


    s = dict(state)


    s["qtr"] = int(
        s["qtr"]
    )


    s[
        "game_seconds_remaining"
    ] = float(
        s[
            "game_seconds_remaining"
        ]
    )


    s["yardline_100"] = float(
        s["yardline_100"]
    )


    s["ydstogo"] = float(
        s["ydstogo"]
    )


    s["score_differential"] = float(
        s[
            "score_differential"
        ]
    )


    s["is_home"] = int(
        s["is_home"]
    )


    if s["is_home"] not in [0, 1]:

        raise ValueError(
            "is_home must be 0 or 1."
        )


    # -----------------------------------------------------
    # Overtime phase.
    #
    # Current 2025+ NFL regular-season OT requires both
    # teams to receive an opportunity to possess before
    # ordinary sudden death, subject to the 10-minute clock.
    #
    # The phase cannot be inferred reliably from ordinary
    # down / distance / score / clock state, so require it
    # explicitly for overtime decisions.
    # -----------------------------------------------------

    if s["qtr"] >= 5:

        if "ot_phase" not in s:

            raise ValueError(
                "Overtime states require ot_phase: "
                "OPENING, RESPONSE, or SUDDEN_DEATH."
            )

        s["ot_phase"] = str(
            s["ot_phase"]
        ).strip().upper()

        allowed_ot_phases = {
            "OPENING",
            "RESPONSE",
            "SUDDEN_DEATH",
        }

        if (
            s["ot_phase"]
            not in allowed_ot_phases
        ):

            raise ValueError(
                "ot_phase must be OPENING, "
                "RESPONSE, or SUDDEN_DEATH."
            )

    else:

        s["ot_phase"] = "REGULATION"


    default_timeouts = (
        2.0
        if s["qtr"] >= 5
        else 3.0
    )


    s.setdefault(
        "posteam_timeouts_remaining",
        default_timeouts,
    )


    s.setdefault(
        "defteam_timeouts_remaining",
        default_timeouts,
    )

    s.setdefault(
        "roof",
        "outdoors",
    )


    s[
        "posteam_timeouts_remaining"
    ] = float(
        s[
            "posteam_timeouts_remaining"
        ]
    )


    s[
        "defteam_timeouts_remaining"
    ] = float(
        s[
            "defteam_timeouts_remaining"
        ]
    )


    s[
        "half_seconds_remaining"
    ] = float(
        s.get(
            "half_seconds_remaining",
            derive_half_seconds(
                s["qtr"],
                s[
                    "game_seconds_remaining"
                ],
            ),
        )
    )


    s["goal_to_go"] = int(

        s.get(
            "goal_to_go",
            (
                s["yardline_100"]
                <=
                s["ydstogo"]
            ),
        )

    )


    if s["is_home"] == 1:

        s[
            "home_score_differential"
        ] = (
            s[
                "score_differential"
            ]
        )

        s[
            "home_timeouts_remaining"
        ] = (
            s[
                "posteam_timeouts_remaining"
            ]
        )

        s[
            "away_timeouts_remaining"
        ] = (
            s[
                "defteam_timeouts_remaining"
            ]
        )

    else:

        s[
            "home_score_differential"
        ] = (
            -
            s[
                "score_differential"
            ]
        )

        s[
            "home_timeouts_remaining"
        ] = (
            s[
                "defteam_timeouts_remaining"
            ]
        )

        s[
            "away_timeouts_remaining"
        ] = (
            s[
                "posteam_timeouts_remaining"
            ]
        )


    return s


# =========================================================
# Build one state accepted by the WP model.
# =========================================================

def make_wp_state(

    base,

    possession_original,

    yardline_100,

    down,

    ydstogo,

    elapsed_seconds=0.0,

    score_change_original=0.0,

    is_kickoff=False,

):

    (
        qtr,
        game_seconds,
        half_seconds,
    ) = advance_clock(
        base,
        elapsed_seconds,
    )


    if base["is_home"] == 1:

        home_score_diff = (

            base[
                "home_score_differential"
            ]

            +

            score_change_original
        )

    else:

        home_score_diff = (

            base[
                "home_score_differential"
            ]

            -

            score_change_original
        )


    if possession_original:

        is_home_posteam = int(
            base["is_home"]
        )

    else:

        is_home_posteam = int(
            1 - base["is_home"]
        )


    if is_kickoff:

        down = np.nan
        ydstogo = np.nan
        yardline_100 = np.nan
        goal_to_go = 0

    else:

        yardline_100 = (
            clip_yardline(
                yardline_100
            )
        )

        down = float(down)

        ydstogo = max(
            0.1,
            float(ydstogo),
        )

        goal_to_go = int(
            yardline_100
            <=
            ydstogo
        )


    return {

        "_possession_original":
            bool(possession_original),

        "_score_change_original":
            float(score_change_original),

        "game_seconds_remaining":
            game_seconds,

        "half_seconds_remaining":
            half_seconds,

        "qtr":
            float(qtr),

        "down":
            down,

        "ydstogo":
            ydstogo,

        "yardline_100":
            yardline_100,

        "goal_to_go":
            float(goal_to_go),

        "home_score_differential":
            home_score_diff,

        "home_timeouts_remaining":
            base[
                "home_timeouts_remaining"
            ],

        "away_timeouts_remaining":
            base[
                "away_timeouts_remaining"
            ],

        "is_home_posteam":
            float(
                is_home_posteam
            ),

        "is_kickoff":
            float(
                int(is_kickoff)
            ),

        "is_overtime":
            float(
                int(
                    qtr >= 5
                )
            ),

        "final_five_minutes":
            float(
                int(
                    qtr == 4
                    and
                    game_seconds
                    <= 300
                )
            ),
    }


# =========================================================
# WP model
# =========================================================

WP_CLASS_INDEX = {

    int(value):
        index

    for index, value
    in enumerate(
        wp_model.classes_
    )
}


def predict_original_team_wp(

    wp_states,
    original_is_home,

):

    frame = pd.DataFrame(
        wp_states
    )


    X = (
        frame[
            WP_FEATURES
        ]
        .to_numpy()
    )


    probabilities = (
        wp_model
        .predict_proba(X)
    )


    if original_is_home:

        class_value = 2

    else:

        class_value = 0


    return probabilities[
        :,
        WP_CLASS_INDEX[
            class_value
        ],
    ]


# =========================================================
# Evaluate post-play states.
#
# Regulation:
#     use the learned WP model unchanged.
#
# 2025+ regular-season overtime:
#     resolve rule-defined terminal outcomes exactly, then
#     leave nonterminal states to the learned WP model.
# =========================================================

def original_score_diff_from_wp_state(
    state,
    original_is_home,
):

    home_diff = float(
        state[
            "home_score_differential"
        ]
    )

    if original_is_home:

        return home_diff

    return -home_diff


def evaluate_post_play_states(
    base,
    states,
):

    wp = (
        predict_original_team_wp(
            states,
            bool(
                base["is_home"]
            ),
        )
        .astype(
            float,
            copy=True,
        )
    )


    # Regulation requires no rule override.
    if base["qtr"] < 5:

        return wp


    phase = base["ot_phase"]

    eps = 1e-9

    original_is_home = bool(
        base["is_home"]
    )


    for index, state in enumerate(
        states
    ):

        post_diff = (
            original_score_diff_from_wp_state(
                state,
                original_is_home,
            )
        )


        score_change = float(
            state.get(
                "_score_change_original",
                (
                    post_diff
                    -
                    float(
                        base[
                            "score_differential"
                        ]
                    )
                ),
            )
        )


        possession_original = bool(
            state.get(
                "_possession_original",
                (
                    int(
                        round(
                            float(
                                state[
                                    "is_home_posteam"
                                ]
                            )
                        )
                    )
                    ==
                    int(
                        base["is_home"]
                    )
                ),
            )
        )


        # -------------------------------------------------
        # Regular-season OT ends after 10 minutes even if
        # the second team's guaranteed opportunity has not
        # completed.
        #
        # A tie is not a win, so its contribution to the
        # engine's literal win-probability objective is 0.
        # -------------------------------------------------

        if (
            float(
                state[
                    "game_seconds_remaining"
                ]
            )
            <= eps
        ):

            if post_diff > eps:

                wp[index] = 1.0

            elif post_diff < -eps:

                wp[index] = 0.0

            else:

                # A tie is worth half a win for
                # fourth-down decision utility.
                wp[index] = 0.5

            continue


        # -------------------------------------------------
        # Opening possession.
        #
        # An offensive FG/TD does NOT automatically end the
        # game under the current rule; the opponent normally
        # receives its response opportunity.
        #
        # A defensive score means the other team gained
        # possession and scored, so the original team loses.
        # -------------------------------------------------

        if phase == "OPENING":

            if score_change < -eps:

                wp[index] = 0.0

            continue


        # -------------------------------------------------
        # Response possession.
        #
        # Once this possession ends, both teams have had
        # their required opportunity.
        #
        # Lead  -> terminal winner.
        # Trail -> terminal loser.
        # Tie   -> sudden death, so retain model WP.
        # -------------------------------------------------

        if phase == "RESPONSE":

            possession_ended = (
                abs(score_change) > eps
                or
                not possession_original
            )

            if possession_ended:

                if post_diff > eps:

                    wp[index] = 1.0

                elif post_diff < -eps:

                    wp[index] = 0.0

            continue


        # -------------------------------------------------
        # Sudden death.
        #
        # Any score ends the game.
        # -------------------------------------------------

        if phase == "SUDDEN_DEATH":

            if abs(score_change) > eps:

                if post_diff > eps:

                    wp[index] = 1.0

                else:

                    wp[index] = 0.0


    return wp


# =========================================================
# Current pre-decision win probability
# =========================================================

def current_win_probability(
    base,
):

    state = make_wp_state(

        base=base,

        possession_original=True,

        yardline_100=
            base[
                "yardline_100"
            ],

        down=4,

        ydstogo=
            base[
                "ydstogo"
            ],

        elapsed_seconds=0.0,

        score_change_original=0.0,

        is_kickoff=False,
    )


    return float(

        predict_original_team_wp(
            [state],
            bool(
                base["is_home"]
            ),
        )[0]

    )


# =========================================================
# Eligibility
# =========================================================

def action_eligible(
    base,
    action,
):

    fg_distance = (

        base[
            "yardline_100"
        ]
        +
        18.0
    )


    # -----------------------------------------------------
    # Normal designed-run development support.
    #
    # Normal fourth-down runs are overwhelmingly concentrated
    # in short-yardage situations in the development data.
    # Beyond 3 yards, the sample becomes too sparse to support
    # a reliable action-specific RUN recommendation.
    #
    # GO itself remains available through PASS/DROPBACK.
    # -----------------------------------------------------

    if (
        action == "NORMAL_GO_RUN"
        and
        base["ydstogo"] > 3
    ):

        return (
            False,
            "outside normal-run development support (ydstogo > 3)",
        )


    # -----------------------------------------------------
    # Fake-play empirical support.
    # -----------------------------------------------------

    if action in FAKE_SUPPORT:

        support = (
            FAKE_SUPPORT[
                action
            ]
        )


        # -------------------------
        # Field-position support
        # -------------------------

        yardline = (
            base[
                "yardline_100"
            ]
        )


        if not (

            support[
                "yardline_low"
            ]
            <=
            yardline
            <=
            support[
                "yardline_high"
            ]

        ):

            return (

                False,

                (
                    "outside central development support "
                    f"({support['yardline_low']:.0f}-"
                    f"{support['yardline_high']:.0f} "
                    "yardline_100)"
                ),
            )


        # -------------------------
        # Yards-to-go support
        # -------------------------

        ydstogo = (
            base[
                "ydstogo"
            ]
        )


        if not (

            support[
                "ydstogo_low"
            ]
            <=
            ydstogo
            <=
            support[
                "ydstogo_high"
            ]

        ):

            return (

                False,

                (
                    "outside central development support "
                    f"(ydstogo "
                    f"{support['ydstogo_low']:.1f}-"
                    f"{support['ydstogo_high']:.1f})"
                ),
            )


        # -------------------------
        # Production fake policy
        # -------------------------

        if not ALLOW_FAKE_RECOMMENDATIONS:

            return (

                False,

                (
                    "experimental only — "
                    "fake-play uplift is not "
                    "causally identified"
                ),
            )


    # -----------------------------------------------------
    # Field-goal distance support.
    # -----------------------------------------------------

    if action in [

        "FIELD_GOAL",
        "FAKE_FG_RUN",
        "FAKE_FG_PASS",

    ]:

        if fg_distance > 66:

            return (
                False,
                "FG distance beyond development support",
            )


    # -----------------------------------------------------
    # Punt-formation support.
    # -----------------------------------------------------

    if action in [

        "PUNT",
        "FAKE_PUNT_RUN",
        "FAKE_PUNT_PASS",

    ]:

        if (
            base[
                "yardline_100"
            ]
            < 20
        ):

            return (
                False,
                "punt formation outside development support",
            )


    return (
        True,
        "",
    )


# =========================================================
# GO / FAKE conversion model
# =========================================================

GO_FEATURES = [

    "ydstogo",
    "yardline_100",

    "goal_to_go",

    "qtr",
    "game_seconds_remaining",

    "score_differential",

    "posteam_timeouts_remaining",
    "defteam_timeouts_remaining",

    "is_home",

    "short_yardage",
    "inside_10",
    "inside_20",
    "final_two_minutes",

    "is_pass",

    "fake_punt",
    "fake_fg",
]


def go_conversion_probability(
    base,
    action,
):

    config = GO_ACTIONS[
        action
    ]


    row = {

        "ydstogo":
            base["ydstogo"],

        "yardline_100":
            base[
                "yardline_100"
            ],

        "goal_to_go":
            base["goal_to_go"],

        "qtr":
            base["qtr"],

        "game_seconds_remaining":
            base[
                "game_seconds_remaining"
            ],

        "score_differential":
            base[
                "score_differential"
            ],

        "posteam_timeouts_remaining":
            base[
                "posteam_timeouts_remaining"
            ],

        "defteam_timeouts_remaining":
            base[
                "defteam_timeouts_remaining"
            ],

        "is_home":
            base["is_home"],

        "short_yardage":
            int(
                base["ydstogo"]
                <= 2
            ),

        "inside_10":
            int(
                base[
                    "yardline_100"
                ]
                <= 10
            ),

        "inside_20":
            int(
                base[
                    "yardline_100"
                ]
                <= 20
            ),

        "final_two_minutes":
            int(
                base[
                    "half_seconds_remaining"
                ]
                <= 120
            ),

        "is_pass":
            config[
                "is_pass"
            ],

        "fake_punt":
            config[
                "fake_punt"
            ],

        "fake_fg":
            config[
                "fake_fg"
            ],
    }


    X = np.array(
        [[
            row[feature]
            for feature
            in GO_FEATURES
        ]],
        dtype=float,
    )


    return float(

        go_model
        .predict_proba(X)[0, 1]

    )


# =========================================================
# Transfer a donor play's field-position movement into the
# current fourth-down situation.
# =========================================================

def transfer_yardline(

    current_start,

    donor_start,

    donor_state_yardline,

    possession_original,

):

    if not (
        finite(donor_start)
        and
        finite(
            donor_state_yardline
        )
    ):

        if possession_original:

            return clip_yardline(
                current_start
            )

        return clip_yardline(
            100.0
            -
            current_start
        )


    donor_start = float(
        donor_start
    )

    donor_state_yardline = float(
        donor_state_yardline
    )


    if possession_original:

        movement = (

            donor_state_yardline
            -
            donor_start
        )


        result = (

            current_start
            +
            movement
        )

    else:

        donor_original_final = (

            100.0
            -
            donor_state_yardline
        )


        movement = (

            donor_original_final
            -
            donor_start
        )


        current_original_final = (

            current_start
            +
            movement
        )


        result = (

            100.0
            -
            current_original_final
        )


    return clip_yardline(
        result
    )


# =========================================================
# Build post-play state from GO donor
# =========================================================

def go_donor_to_state(
    base,
    donor,
    converted,
):

    transition = str(
        donor[
            "transition_class"
        ]
    )


    elapsed = safe_float(
        donor[
            "seconds_to_state"
        ],
        6.0,
    )


    # -----------------------------------------------------
    # Offensive score
    # -----------------------------------------------------

    if transition == "offense_scored":

        score_change = safe_float(
            donor.get(
                "score_change",
                7.0,
            ),
            7.0,
        )


        if score_change <= 0:

            score_change = 7.0


        # The donor clock can include the ensuing kickoff.
        # For kickoff-pending representation, cap it.
        elapsed = min(
            elapsed,
            10.0,
        )


        return make_wp_state(

            base,

            possession_original=False,

            yardline_100=np.nan,

            down=np.nan,

            ydstogo=np.nan,

            elapsed_seconds=elapsed,

            score_change_original=
                score_change,

            is_kickoff=True,
        )


    # -----------------------------------------------------
    # Defensive score
    # -----------------------------------------------------

    if transition == "opponent_scored":

        score_change = safe_float(
            donor.get(
                "score_change",
                -7.0,
            ),
            -7.0,
        )


        if score_change >= 0:

            score_change = -7.0


        elapsed = min(
            elapsed,
            10.0,
        )


        return make_wp_state(

            base,

            possession_original=True,

            yardline_100=np.nan,

            down=np.nan,

            ydstogo=np.nan,

            elapsed_seconds=elapsed,

            score_change_original=
                score_change,

            is_kickoff=True,
        )


    # -----------------------------------------------------
    # Original offense retains ball
    # -----------------------------------------------------

    if (
        transition
        ==
        "same_offense_no_score"
    ):

        new_yardline = transfer_yardline(

            base[
                "yardline_100"
            ],

            donor.get(
                "yardline_100",
                np.nan,
            ),

            donor.get(
                "state_yardline_100",
                np.nan,
            ),

            possession_original=True,
        )


        down = safe_float(
            donor.get(
                "state_down",
                1,
            ),
            1,
        )


        if int(round(down)) == 1:

            ydstogo = min(
                10.0,
                new_yardline,
            )

        else:

            ydstogo = safe_float(
                donor.get(
                    "state_ydstogo",
                    base[
                        "ydstogo"
                    ],
                ),
                base[
                    "ydstogo"
                ],
            )


        return make_wp_state(

            base,

            possession_original=True,

            yardline_100=
                new_yardline,

            down=down,

            ydstogo=ydstogo,

            elapsed_seconds=
                elapsed,
        )


    # -----------------------------------------------------
    # Opponent takes possession
    # -----------------------------------------------------

    if (
        transition
        ==
        "opponent_ball_no_score"
    ):

        new_yardline = transfer_yardline(

            base[
                "yardline_100"
            ],

            donor.get(
                "yardline_100",
                np.nan,
            ),

            donor.get(
                "state_yardline_100",
                np.nan,
            ),

            possession_original=False,
        )


        return make_wp_state(

            base,

            possession_original=False,

            yardline_100=
                new_yardline,

            down=1,

            ydstogo=min(
                10.0,
                new_yardline,
            ),

            elapsed_seconds=
                elapsed,
        )


    # -----------------------------------------------------
    # Fallback
    # -----------------------------------------------------

    if converted:

        new_yardline = clip_yardline(

            base[
                "yardline_100"
            ]
            -
            base[
                "ydstogo"
            ]

        )


        return make_wp_state(

            base,

            possession_original=True,

            yardline_100=
                new_yardline,

            down=1,

            ydstogo=min(
                10.0,
                new_yardline,
            ),

            elapsed_seconds=6.0,
        )


    opponent_yardline = clip_yardline(

        100.0
        -
        base[
            "yardline_100"
        ]

    )


    return make_wp_state(

        base,

        possession_original=False,

        yardline_100=
            opponent_yardline,

        down=1,

        ydstogo=min(
            10.0,
            opponent_yardline,
        ),

        elapsed_seconds=6.0,
    )


# =========================================================
# Simulate GO / fake GO action
# =========================================================

def simulate_go(

    base,
    action,
    n,
    rng,

):

    p_conversion = (
        go_conversion_probability(
            base,
            action,
        )
    )


    converted = (
        rng.random(n)
        <
        p_conversion
    )


    is_pass = (
        GO_ACTIONS[
            action
        ][
            "is_pass"
        ]
    )


    states = []


    for outcome in [
        0,
        1,
    ]:

        count = int(
            np.sum(
                converted
                ==
                bool(outcome)
            )
        )


        if count == 0:

            continue


        donor_pool = go_pool[

            (
                go_pool[
                    "is_pass"
                ]
                ==
                is_pass
            )

            &

            (
                go_pool[
                    "converted"
                ]
                ==
                outcome
            )

        ].copy()


        if len(
            donor_pool
        ) == 0:

            raise RuntimeError(
                "No GO transition donors for "
                f"is_pass={is_pass}, "
                f"converted={outcome}"
            )

                # -------------------------------------------------
        # Local transition matching.
        #
        # Prevent fourth downs in one part of the field
        # from borrowing transitions from very different
        # field-position / distance situations.
        # -------------------------------------------------

        donor_pool[
            "_yardline_gap"
        ] = (

            donor_pool[
                "yardline_100"
            ]
            -
            base[
                "yardline_100"
            ]

        ).abs()


        donor_pool[
            "_ydstogo_gap"
        ] = (

            donor_pool[
                "ydstogo"
            ]
            -
            base[
                "ydstogo"
            ]

        ).abs()


        donor_goal_to_go = (

            donor_pool[
                "yardline_100"
            ]
            <=
            donor_pool[
                "ydstogo"
            ]

        ).astype(int)


        donor_pool[
            "_goal_to_go_gap"
        ] = (

            donor_goal_to_go
            -
            int(
                base[
                    "goal_to_go"
                ]
            )

        ).abs()


        donor_pool[
            "_similarity"
        ] = (

            donor_pool[
                "_yardline_gap"
            ]
            / 10.0

            +

            donor_pool[
                "_ydstogo_gap"
            ]
            / 2.0

            +

            5.0
            *
            donor_pool[
                "_goal_to_go_gap"
            ]

        )


        n_local_donors = min(
            250,
            len(
                donor_pool
            ),
        )


        donor_pool = (

            donor_pool
            .nsmallest(
                n_local_donors,
                "_similarity",
            )

        )

        donor_indices = (
            rng.integers(
                0,
                len(
                    donor_pool
                ),
                size=count,
            )
        )


        donors = (
            donor_pool
            .iloc[
                donor_indices
            ]
        )


        for _, donor in (
            donors.iterrows()
        ):

            states.append(

                go_donor_to_state(

                    base,

                    donor,

                    converted=
                        bool(outcome),
                )

            )


    wp = (
        evaluate_post_play_states(
            base,
            states,
        )
    )


    return {

        "q_value":
            float(
                np.mean(wp)
            ),

        "mc_se":
            float(
                np.std(wp)
                /
                np.sqrt(
                    len(wp)
                )
            ),

        "detail":
            (
                f"conversion="
                f"{p_conversion:.3f}"
            ),
    }


# =========================================================
# Field-goal outcome probabilities
# =========================================================

def field_goal_probabilities(
    base,
):

    distance = (

        base[
            "yardline_100"
        ]
        +
        18.0
    )


    roof = str(
        base.get(
            "roof",
            "outdoors",
        )
    ).lower()


    roof_dome = int(
        "dome" in roof
        or
        "indoor" in roof
    )


    roof_closed = int(
        "closed" in roof
    )


    roof_open = int(
        "open" in roof
        and
        not roof_dome
    )


    kicker_skill = float(

        base.get(

            "adjusted_kicker_skill",

            fg_bundle[
                "default_adjusted_kicker_skill"
            ],

        )

    )


    if (
        "log_adjusted_prior_attempts"
        in base
    ):

        log_prior = float(
            base[
                "log_adjusted_prior_attempts"
            ]
        )

    elif (
        "adjusted_prior_attempts"
        in base
    ):

        log_prior = float(
            np.log1p(
                max(
                    0.0,
                    float(
                        base[
                            "adjusted_prior_attempts"
                        ]
                    ),
                )
            )
        )

    else:

        log_prior = float(

            fg_bundle[
                "default_log_adjusted_prior_attempts"
            ]

        )


    make_row = pd.DataFrame([{

        "fg_distance_estimate":
            distance,

        "fg_distance_sq":
            distance ** 2,

        "roof_dome":
            roof_dome,

        "roof_closed":
            roof_closed,

        "roof_open":
            roof_open,

        "adjusted_kicker_skill":
            kicker_skill,

        "log_adjusted_prior_attempts":
            log_prior,
    }])


    p_make_clean = float(

        fg_bundle[
            "make_model"
        ]
        .predict_proba(

            make_row[
                fg_bundle[
                    "make_features"
                ]
            ]

        )[0, 1]

    )


    block_row = pd.DataFrame([{

        "fg_distance_estimate":
            distance,
    }])


    p_block_failure = float(

        fg_bundle[
            "block_model"
        ]
        .predict_proba(

            block_row[
                fg_bundle[
                    "block_features"
                ]
            ]

        )[0, 1]

    )


    p_broken = float(
        fg_bundle[
            "p_broken"
        ]
    )


    p_clean = (
        1.0
        -
        p_broken
    )


    p_made = (

        p_clean
        *
        p_make_clean

    )


    p_clean_failure = (

        p_clean
        *
        (
            1.0
            -
            p_make_clean
        )

    )


    p_blocked = (

        p_clean_failure
        *
        p_block_failure

    )


    p_missed = (

        p_clean_failure
        *
        (
            1.0
            -
            p_block_failure
        )

    )


    probabilities = {

        "made":
            p_made,

        "missed":
            p_missed,

        "blocked":
            p_blocked,

        "broken":
            p_broken,
    }


    total = sum(
        probabilities.values()
    )


    for key in probabilities:

        probabilities[key] /= (
            total
        )


    return probabilities


# =========================================================
# FG clock sampler
# =========================================================

def fg_distance_bucket(
    distance,
):

    if distance <= 39:

        return "<=39"

    if distance <= 49:

        return "40-49"

    return "50+"


def sample_fg_clock(
    distance,
    rng,
):

    bucket = (
        fg_distance_bucket(
            distance
        )
    )


    sample = fg_clock_pool[

        fg_clock_pool[
            "distance_bucket"
        ]
        ==
        bucket

    ]


    if len(sample) == 0:

        sample = (
            fg_clock_pool
        )


    index = int(
        rng.integers(
            0,
            len(sample),
        )
    )


    return safe_float(

        sample
        .iloc[index][
            "fg_clock_cost"
        ],

        4.0,
    )


# =========================================================
# FG live-ball rate
# =========================================================

def blocked_fg_score_probability():

    row = fg_live_rates[

        fg_live_rates[
            "transition_class"
        ]
        ==
        "opponent_scored"

    ]


    if len(row) == 0:

        return (
            5.0
            /
            154.0
        )


    return float(
        row.iloc[0][
            "probability"
        ]
    )


# =========================================================
# Simulate FIELD GOAL
# =========================================================

def simulate_field_goal(

    base,
    n,
    rng,

):

    probabilities = (
        field_goal_probabilities(
            base
        )
    )


    labels = list(
        probabilities.keys()
    )


    p = np.array(
        [
            probabilities[
                label
            ]
            for label in labels
        ]
    )


    outcomes = rng.choice(

        labels,

        size=n,

        p=p,
    )


    distance = (

        base[
            "yardline_100"
        ]
        +
        18.0
    )


    p_block_score = (
        blocked_fg_score_probability()
    )


    states = []


    for outcome in outcomes:

        # -------------------------------------------------
        # MADE
        # -------------------------------------------------

        if outcome == "made":

            elapsed = sample_fg_clock(
                distance,
                rng,
            )


            states.append(

                make_wp_state(

                    base,

                    possession_original=False,

                    yardline_100=np.nan,

                    down=np.nan,

                    ydstogo=np.nan,

                    elapsed_seconds=
                        elapsed,

                    score_change_original=
                        3.0,

                    is_kickoff=True,
                )

            )

            continue


        # -------------------------------------------------
        # MISSED
        # -------------------------------------------------

        if outcome == "missed":

            elapsed = sample_fg_clock(
                distance,
                rng,
            )


            opponent_yardline = min(

                80.0,

                92.0
                -
                base[
                    "yardline_100"
                ],
            )


            opponent_yardline = (
                clip_yardline(
                    opponent_yardline
                )
            )


            states.append(

                make_wp_state(

                    base,

                    possession_original=False,

                    yardline_100=
                        opponent_yardline,

                    down=1,

                    ydstogo=min(
                        10.0,
                        opponent_yardline,
                    ),

                    elapsed_seconds=
                        elapsed,
                )

            )

            continue


        # -------------------------------------------------
        # BLOCKED / BROKEN
        #
        # Broken executions borrow the blocked-FG live-ball
        # transition system.
        # -------------------------------------------------

        if (
            rng.random()
            <
            p_block_score
        ):

            states.append(

                make_wp_state(

                    base,

                    possession_original=True,

                    yardline_100=np.nan,

                    down=np.nan,

                    ydstogo=np.nan,

                    elapsed_seconds=8.0,

                    score_change_original=
                        -7.0,

                    is_kickoff=True,
                )

            )

            continue


        donor_index = int(

            rng.integers(
                0,
                len(
                    fg_block_pool
                ),
            )

        )


        donor = (
            fg_block_pool
            .iloc[
                donor_index
            ]
        )


        residual = safe_float(

            donor[
                BLOCK_RESIDUAL_COLUMN
            ],

            0.0,
        )


        elapsed = safe_float(

            donor[
                BLOCK_CLOCK_COLUMN
            ],

            6.0,
        )


        miss_rule_yardline = min(

            80.0,

            92.0
            -
            base[
                "yardline_100"
            ],
        )


        opponent_yardline = (
            clip_yardline(

                miss_rule_yardline
                +
                residual

            )
        )


        states.append(

            make_wp_state(

                base,

                possession_original=False,

                yardline_100=
                    opponent_yardline,

                down=1,

                ydstogo=min(
                    10.0,
                    opponent_yardline,
                ),

                elapsed_seconds=
                    elapsed,
            )

        )


    wp = evaluate_post_play_states(
        base,
        states,
    )


    return {

        "q_value":
            float(
                np.mean(wp)
            ),

        "mc_se":
            float(
                np.std(wp)
                /
                np.sqrt(
                    len(wp)
                )
            ),

        "detail":
            (
                f"make="
                f"{probabilities['made']:.3f}, "
                f"miss="
                f"{probabilities['missed']:.3f}, "
                f"block="
                f"{probabilities['blocked']:.3f}"
            ),
    }


# =========================================================
# Punt helpers
# =========================================================

PUNT_FEATURES = [

    "yardline_100",
    "ydstogo",

    "qtr",
    "game_seconds_remaining",

    "score_differential",

    "posteam_timeouts_remaining",
    "defteam_timeouts_remaining",

    "is_home",
]


def punt_field_prediction(
    base,
):

    row = np.array([[
        base[
            feature
        ]
        for feature
        in PUNT_FEATURES
    ]])


    return float(

        punt_model.predict(
            row
        )[0]

    )


def sample_ordinary_punt_state(

    base,
    predicted_yardline,
    rng,

):

    start_bucket = int(

        np.floor(
            base[
                "yardline_100"
            ]
            /
            10.0
        )
        *
        10

    )


    candidate = punt_pool[

        punt_pool[
            "start_yardline_bucket"
        ]
        ==
        start_bucket

    ]


    if len(candidate) == 0:

        available = np.array(

            sorted(
                punt_pool[
                    "start_yardline_bucket"
                ]
                .dropna()
                .unique()
            )

        )


        nearest = available[

            np.argmin(
                np.abs(
                    available
                    -
                    start_bucket
                )
            )

        ]


        candidate = punt_pool[

            punt_pool[
                "start_yardline_bucket"
            ]
            ==
            nearest

        ]


    donor = candidate.iloc[

        int(
            rng.integers(
                0,
                len(candidate),
            )
        )

    ]


    receiving_yardline = (
        clip_yardline(

            predicted_yardline

            +

            safe_float(
                donor[
                    "yardline_residual"
                ],
                0.0,
            )

        )
    )


    elapsed = safe_float(

        donor[
            "seconds_to_state"
        ],

        9.0,
    )


    return make_wp_state(

        base,

        possession_original=False,

        yardline_100=
            receiving_yardline,

        down=1,

        ydstogo=min(
            10.0,
            receiving_yardline,
        ),

        elapsed_seconds=
            elapsed,
    )


def punt_rare_donor_state(

    base,
    donor,

):

    transition = str(
        donor.get(
            "transition_class",
            "",
        )
    )


    elapsed = safe_float(
        donor.get(
            "seconds_to_state",
            9.0,
        ),
        9.0,
    )


    # Opponent return / defensive score.
    if (
        transition
        ==
        "opponent_scored"
    ):

        score_change = safe_float(
            donor.get(
                "score_change",
                -7.0,
            ),
            -7.0,
        )


        if score_change >= 0:

            score_change = -7.0


        return make_wp_state(

            base,

            possession_original=True,

            yardline_100=np.nan,

            down=np.nan,

            ydstogo=np.nan,

            elapsed_seconds=min(
                elapsed,
                15.0,
            ),

            score_change_original=
                score_change,

            is_kickoff=True,
        )


    # Kicking team scores.
    if (
        transition
        ==
        "kicking_team_scored"
    ):

        score_change = safe_float(
            donor.get(
                "score_change",
                7.0,
            ),
            7.0,
        )


        if score_change <= 0:

            score_change = 7.0


        return make_wp_state(

            base,

            possession_original=False,

            yardline_100=np.nan,

            down=np.nan,

            ydstogo=np.nan,

            elapsed_seconds=min(
                elapsed,
                15.0,
            ),

            score_change_original=
                score_change,

            is_kickoff=True,
        )


    if (
        transition
        ==
        "kicking_team_ball_no_score"
    ):

        new_yardline = transfer_yardline(

            base[
                "yardline_100"
            ],

            donor.get(
                "yardline_100",
                np.nan,
            ),

            donor.get(
                "state_yardline_100",
                np.nan,
            ),

            possession_original=True,
        )


        down = safe_float(
            donor.get(
                "state_down",
                1.0,
            ),
            1.0,
        )


        ydstogo = safe_float(
            donor.get(
                "state_ydstogo",
                min(
                    10.0,
                    new_yardline,
                ),
            ),
            min(
                10.0,
                new_yardline,
            ),
        )


        return make_wp_state(

            base,

            possession_original=True,

            yardline_100=
                new_yardline,

            down=down,

            ydstogo=ydstogo,

            elapsed_seconds=
                elapsed,
        )


    if (
        transition
        ==
        "opponent_ball_no_score"
    ):

        new_yardline = transfer_yardline(

            base[
                "yardline_100"
            ],

            donor.get(
                "yardline_100",
                np.nan,
            ),

            donor.get(
                "state_yardline_100",
                np.nan,
            ),

            possession_original=False,
        )


        return make_wp_state(

            base,

            possession_original=False,

            yardline_100=
                new_yardline,

            down=1,

            ydstogo=min(
                10.0,
                new_yardline,
            ),

            elapsed_seconds=
                elapsed,
        )


    return None


# =========================================================
# Simulate PUNT
# =========================================================

def simulate_punt(

    base,
    n,
    rng,

):

    predicted_yardline = (
        punt_field_prediction(
            base
        )
    )


    p_broken = float(

        punt_spec[
            "broken_probability"
        ]

    )


    p_block_given_normal = float(

        punt_spec[
            "blocked_probability_given_normal_execution"
        ]

    )


    p_blocked = (

        (
            1.0
            -
            p_broken
        )

        *

        p_block_given_normal

    )


    regular_probabilities = dict(

        punt_spec[
            "regular_transition_probabilities"
        ]

    )


    regular_classes = list(
        regular_probabilities.keys()
    )


    regular_p = np.array(
        [
            regular_probabilities[
                key
            ]
            for key
            in regular_classes
        ],
        dtype=float,
    )


    regular_p /= (
        regular_p.sum()
    )


    states = []


    for _ in range(n):

        u = rng.random()


        # -------------------------------------------------
        # Broken punt
        # -------------------------------------------------

        if u < p_broken:

            donors = punt_rare[

                punt_rare[
                    "punt_branch"
                ]
                ==
                "BROKEN"

            ]


            if len(donors) > 0:

                donor = donors.iloc[

                    int(
                        rng.integers(
                            0,
                            len(donors),
                        )
                    )

                ]


                state = (
                    punt_rare_donor_state(
                        base,
                        donor,
                    )
                )


                if state is not None:

                    states.append(
                        state
                    )

                    continue


        # -------------------------------------------------
        # Blocked punt
        # -------------------------------------------------

        elif (
            u
            <
            p_broken
            +
            p_blocked
        ):

            donors = punt_rare[

                punt_rare[
                    "punt_branch"
                ]
                ==
                "BLOCKED"

            ]


            if len(donors) > 0:

                donor = donors.iloc[

                    int(
                        rng.integers(
                            0,
                            len(donors),
                        )
                    )

                ]


                state = (
                    punt_rare_donor_state(
                        base,
                        donor,
                    )
                )


                if state is not None:

                    states.append(
                        state
                    )

                    continue


        # -------------------------------------------------
        # Regular punt
        # -------------------------------------------------

        else:

            transition = rng.choice(

                regular_classes,

                p=regular_p,
            )


            if (
                transition
                ==
                "opponent_ball_no_score"
            ):

                states.append(

                    sample_ordinary_punt_state(

                        base,

                        predicted_yardline,

                        rng,
                    )

                )

                continue


            donors = punt_rare[

                (
                    punt_rare[
                        "punt_branch"
                    ]
                    ==
                    "REGULAR"
                )

                &

                (
                    punt_rare[
                        "transition_class"
                    ]
                    ==
                    transition
                )

            ]


            if len(donors) > 0:

                donor = donors.iloc[

                    int(
                        rng.integers(
                            0,
                            len(donors),
                        )
                    )

                ]


                state = (
                    punt_rare_donor_state(
                        base,
                        donor,
                    )
                )


                if state is not None:

                    states.append(
                        state
                    )

                    continue


        # -------------------------------------------------
        # Very rare unsupported boundary fallback.
        # -------------------------------------------------

        states.append(

            sample_ordinary_punt_state(

                base,

                predicted_yardline,

                rng,
            )

        )


    wp = evaluate_post_play_states(
        base,
        states,
    )


    return {

        "q_value":
            float(
                np.mean(wp)
            ),

        "mc_se":
            float(
                np.std(wp)
                /
                np.sqrt(
                    len(wp)
                )
            ),

        "detail":
            (
                f"regular="
                f"{1 - p_broken - p_blocked:.3f}, "
                f"blocked="
                f"{p_blocked:.4f}, "
                f"broken="
                f"{p_broken:.4f}"
            ),
    }


# =========================================================
# Main recommendation function
# =========================================================

def recommend(

    state,

    n_simulations=
        N_SIMULATIONS,

    seed=
        RANDOM_SEED,

):

    base = normalize_state(
        state
    )


    rng = (
        np.random.default_rng(
            seed
        )
    )


    pre_wp = (
        current_win_probability(
            base
        )
    )


    rows = []


    for action in ACTIONS:

        (
            eligible,
            reason,
        ) = action_eligible(
            base,
            action,
        )


        if not eligible:

            rows.append({

                "action":
                    action,

                "decision":
                    DISPLAY_NAMES[
                        action
                    ],

                "eligible":
                    False,

                "expected_win_probability":
                    np.nan,

                "mc_se":
                    np.nan,

                "detail":
                    reason,
            })

            continue


        if action == "PUNT":

            result = simulate_punt(

                base,

                n_simulations,

                rng,
            )


        elif (
            action
            ==
            "FIELD_GOAL"
        ):

            result = (
                simulate_field_goal(

                    base,

                    n_simulations,

                    rng,
                )
            )


        else:

            result = simulate_go(

                base,

                action,

                n_simulations,

                rng,
            )


        rows.append({

            "action":
                action,

            "decision":
                DISPLAY_NAMES[
                    action
                ],

            "eligible":
                True,

            "expected_win_probability":
                result[
                    "q_value"
                ],

            "mc_se":
                result[
                    "mc_se"
                ],

            "detail":
                result[
                    "detail"
                ],
        })


    results = pd.DataFrame(
        rows
    )


    eligible = (
        results[
            results[
                "eligible"
            ]
        ]
        .sort_values(

            "expected_win_probability",

            ascending=False,
        )
        .copy()
    )


    ineligible = (
        results[
            ~results[
                "eligible"
            ]
        ]
        .copy()
    )


    results = pd.concat(
        [
            eligible,
            ineligible,
        ],
        ignore_index=True,
    )


    results[
        "win_probability_pct"
    ] = (

        100.0

        *

        results[
            "expected_win_probability"
        ]

    )


    results[
        "mc_se_pct"
    ] = (

        100.0

        *

        results[
            "mc_se"
        ]

    )


    if len(eligible) >= 2:

        best_wp = float(

            eligible.iloc[0][
                "expected_win_probability"
            ]

        )


        second_wp = float(

            eligible.iloc[1][
                "expected_win_probability"
            ]

        )


        edge = (

            100.0

            *

            (
                best_wp
                -
                second_wp
            )

        )

    else:

        edge = np.nan


    return {

        "state":
            base,

        "pre_decision_wp":
            pre_wp,

        "recommendation":
            (
                eligible.iloc[0][
                    "decision"
                ]
                if len(eligible) > 0
                else None
            ),

        "edge_over_second_best_pct":
            edge,

        "results":
            results,
    }


# =========================================================
# Built-in integration test.
#
# 4th-and-2 from opponent 38.
# Down 3 points, 7:00 left in Q4.
#
# Estimated FG distance = 56 yards.
#
# Normal go, punt, and field goal should be eligible.
# Fake actions are experimental-only and are not eligible
# for production recommendations.
# =========================================================

if __name__ == "__main__":

    example_state = {

        "qtr":
            4,

        "game_seconds_remaining":
            420,

        "yardline_100":
            38,

        "ydstogo":
            2,

        "score_differential":
            -3,

        "posteam_timeouts_remaining":
            3,

        "defteam_timeouts_remaining":
            3,

        # Current offense is the home team.
        "is_home":
            1,

        "roof":
            "outdoors",
    }


    result = recommend(

        example_state,

        n_simulations=
            N_SIMULATIONS,

        seed=
            RANDOM_SEED,
    )


    print(
        "\nEXAMPLE FOURTH-DOWN STATE"
    )

    print(
        "4th-and-2 at opponent 38, "
        "Q4 7:00, offense down 3"
    )


    print(
        "\nPre-decision team WP: "
        f"{100 * result['pre_decision_wp']:.2f}%"
    )


    print(
        "\nACTION VALUES"
    )


    display = (

        result[
            "results"
        ][[

            "decision",

            "eligible",

            "win_probability_pct",

            "mc_se_pct",

            "detail",

        ]]
        .copy()

    )


    display[
        "win_probability_pct"
    ] = (

        display[
            "win_probability_pct"
        ]
        .round(2)

    )


    display[
        "mc_se_pct"
    ] = (

        display[
            "mc_se_pct"
        ]
        .round(3)

    )


    print(
        display.to_string(
            index=False
        )
    )


    print(
        "\nRECOMMENDATION:"
    )

    print(
        result[
            "recommendation"
        ]
    )


    print(
        "\nEdge over second-best: "
        f"{result['edge_over_second_best_pct']:.2f} "
        "percentage points"
    )