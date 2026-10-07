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



    # -----------------------------------------------------
    # Distance-to-go input validation.
    #
    # A first-down line cannot lie beyond the opponent goal
    # line. Inside the 1-yard line, allow the conventional
    # integer representation of 1 yard to go.
    # -----------------------------------------------------

    max_yards_to_go = max(
        1.0,
        float(
            s["yardline_100"]
        ),
    )

    if (
        float(
            s["ydstogo"]
        )
        >
        max_yards_to_go
        + 1e-9
    ):

        raise ValueError(
            "ydstogo cannot exceed the distance "
            "to the opponent goal line."
        )


    # -----------------------------------------------------
    # Public game-site contract.
    #
    # HOME:
    #     current offense is playing at home.
    #
    # AWAY:
    #     current offense is playing away.
    #
    # NEUTRAL:
    #     neither team receives home-field advantage.
    #
    # Legacy is_home / is_neutral_site inputs are accepted
    # temporarily so old audits and scripts still run, but
    # they are immediately canonicalized to "site".
    # -----------------------------------------------------

    raw_site = s.get(
        "site"
    )

    if raw_site is None:

        legacy_neutral = int(
            s.get(
                "is_neutral_site",
                0,
            )
        )

        if legacy_neutral not in [0, 1]:

            raise ValueError(
                "is_neutral_site must be 0 or 1."
            )

        legacy_home = s.get(
            "is_home"
        )

        if legacy_neutral == 1:

            raw_site = "NEUTRAL"

        elif legacy_home is not None:

            legacy_home = int(
                legacy_home
            )

            if legacy_home not in [0, 1]:

                raise ValueError(
                    "is_home must be 0 or 1."
                )

            raw_site = (
                "HOME"
                if legacy_home == 1
                else "AWAY"
            )

        else:

            raise ValueError(
                "site must be HOME, AWAY, or NEUTRAL."
            )


    site = (
        str(raw_site)
        .strip()
        .upper()
    )

    if site not in {
        "HOME",
        "AWAY",
        "NEUTRAL",
    }:

        raise ValueError(
            "site must be HOME, AWAY, or NEUTRAL."
        )


    s["site"] = site

    # Remove legacy public representation after
    # canonicalization. Downstream logic sees only `site`.
    s.pop(
        "is_home",
        None,
    )

    s.pop(
        "is_neutral_site",
        None,
    )


    s["site_advantage"] = {
        "HOME": 1,
        "AWAY": -1,
        "NEUTRAL": 0,
    }[
        site
    ]


    # -----------------------------------------------------
    # Private WP-model coordinate axis.
    #
    # The trained WP artifact expresses score, timeout, and
    # possession features in a "home-team" coordinate system.
    #
    # For HOME/AWAY, this naturally follows the real site.
    #
    # For NEUTRAL, we pick one canonical coordinate ONLY to
    # construct model features. It has no football meaning:
    # neutral predictions are evaluated in both mirrored
    # orientations and averaged below.
    # -----------------------------------------------------

    s[
        "_wp_original_on_home_axis"
    ] = (
        0
        if site == "AWAY"
        else 1
    )


    # -----------------------------------------------------
    # Overtime rule metadata.
    #
    # qtr == 5 remains the model's generic overtime bucket.
    # ot_format and ot_period carry 2026 rule information
    # without inventing qtr == 6, 7, ... model states.
    # -----------------------------------------------------

    if s["qtr"] >= 5:

        s["ot_format"] = str(
            s.get(
                "ot_format",
                "REGULAR_SEASON",
            )
        ).strip().upper()

        allowed_ot_formats = {
            "REGULAR_SEASON",
            "POSTSEASON",
        }

        if (
            s["ot_format"]
            not in allowed_ot_formats
        ):

            raise ValueError(
                "ot_format must be REGULAR_SEASON "
                "or POSTSEASON."
            )

        s["ot_period"] = int(
            s.get(
                "ot_period",
                1,
            )
        )

        if s["ot_period"] < 1:

            raise ValueError(
                "ot_period must be at least 1 "
                "in overtime."
            )

        if (
            s["ot_format"]
            == "REGULAR_SEASON"
            and
            s["ot_period"] != 1
        ):

            raise ValueError(
                "2026 regular-season overtime "
                "has only one period."
            )

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

        s["ot_format"] = "REGULATION"
        s["ot_period"] = 0
        s["ot_phase"] = "REGULATION"


    if s["qtr"] >= 5:

        default_timeouts = (
            3.0
            if s["ot_format"] == "POSTSEASON"
            else 2.0
        )

    else:

        default_timeouts = 3.0


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


    if s["_wp_original_on_home_axis"] == 1:

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

    ot_opportunity_completed=False,

    touchdown_original=0,

):

    (
        qtr,
        game_seconds,
        half_seconds,
    ) = advance_clock(
        base,
        elapsed_seconds,
    )


    # -----------------------------------------------------
    # Postseason overtime period crossing.
    #
    # The model continues to use qtr == 5 for all overtime.
    # Rule-only ot_period tracks OT1, OT2, ...
    #
    # If a play reaches the end of a postseason OT period,
    # the next period begins with a fresh 15:00 clock.
    # Elapsed time does not spill into the next period.
    # -----------------------------------------------------

    ot_period = int(
        base.get(
            "ot_period",
            0,
        )
    )

    crossed_postseason_ot_period = (
        int(base["qtr"]) >= 5
        and
        base.get("ot_format")
        == "POSTSEASON"
        and
        float(
            base[
                "game_seconds_remaining"
            ]
        )
        > 0.0
        and
        float(elapsed_seconds)
        >=
        float(
            base[
                "game_seconds_remaining"
            ]
        )
    )

    if crossed_postseason_ot_period:

        ot_period += 1

        # qtr == 5 is the model's generic OT bucket.
        qtr = 5

        # Every postseason overtime period starts at 15:00.
        game_seconds = 900.0
        half_seconds = 900.0


    # Postseason OT uses three timeouts per overtime half:
    #
    #   OT1 + OT2 -> one timeout pool
    #   OT3 + OT4 -> fresh timeout pool
    #   OT5 + OT6 -> fresh timeout pool
    #
    # Therefore reset when a period crossing enters an
    # odd-numbered OT period after OT1.
    reset_postseason_ot_timeouts = (
        crossed_postseason_ot_period
        and
        ot_period >= 3
        and
        ot_period % 2 == 1
    )


    # -----------------------------------------------------
    # Halftime crossing.
    #
    # If a Q2 play consumes the remaining half clock, the
    # next football state is the Q3 opening kickoff:
    #
    #   - Q3 15:00
    #   - both teams have 3 timeouts
    #   - no scrimmage down / distance / field position
    #   - possession orientation follows the known
    #     second-half kickoff receiver
    #
    # Legacy programmatic callers that do not provide
    # second_half_receiver retain the action-derived
    # possession orientation, but app states provide it.
    # -----------------------------------------------------

    crossed_halftime = (
        int(base["qtr"]) == 2
        and
        float(elapsed_seconds)
        >=
        float(
            base["half_seconds_remaining"]
        )
    )


    if crossed_halftime:

        qtr = 3
        game_seconds = 1800.0
        half_seconds = 1800.0

        second_half_receiver = (
            base.get(
                "second_half_receiver"
            )
        )

        if second_half_receiver is not None:

            second_half_receiver = (
                str(second_half_receiver)
                .strip()
                .upper()
            )

            if (
                second_half_receiver
                ==
                "OFFENSE"
            ):

                possession_original = True

            elif (
                second_half_receiver
                ==
                "DEFENSE"
            ):

                possession_original = False

            else:

                raise ValueError(
                    "second_half_receiver must be "
                    "OFFENSE or DEFENSE."
                )


        is_kickoff = True


    if base["_wp_original_on_home_axis"] == 1:

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
            base["_wp_original_on_home_axis"]
        )

    else:

        is_home_posteam = int(
            1 - base["_wp_original_on_home_axis"]
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

        "_ot_opportunity_completed":
            bool(
                ot_opportunity_completed
            ),

        "_touchdown_original":
            int(
                touchdown_original
            ),

        "_crossed_postseason_ot_period":
            bool(
                crossed_postseason_ot_period
            ),

        "ot_format":
            base.get(
                "ot_format",
                "REGULATION",
            ),

        "ot_period":
            float(ot_period),

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
            (
                3.0
                if (
                    crossed_halftime
                    or
                    reset_postseason_ot_timeouts
                )
                else
                base[
                    "home_timeouts_remaining"
                ]
            ),

        "away_timeouts_remaining":
            (
                3.0
                if (
                    crossed_halftime
                    or
                    reset_postseason_ot_timeouts
                )
                else
                base[
                    "away_timeouts_remaining"
                ]
            ),

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
    original_on_home_axis,
    neutral_site=False,

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


    if original_on_home_axis:

        class_value = 2

    else:

        class_value = 0


    original_probability = (
        probabilities[
            :,
            WP_CLASS_INDEX[
                class_value
            ],
        ]
    )


    # -----------------------------------------------------
    # Neutral-site WP symmetry.
    #
    # The learned WP artifact uses a home-team coordinate
    # system. For a neutral game, that coordinate must not
    # create home-field advantage.
    #
    # Evaluate the same football state in both mirrored
    # model orientations and average the original team's
    # win probability.
    # -----------------------------------------------------

    if not neutral_site:

        return original_probability


    mirrored = frame.copy()


    mirrored[
        "home_score_differential"
    ] = (
        -
        mirrored[
            "home_score_differential"
        ]
    )


    home_timeouts = (
        mirrored[
            "home_timeouts_remaining"
        ]
        .copy()
    )


    mirrored[
        "home_timeouts_remaining"
    ] = (
        mirrored[
            "away_timeouts_remaining"
        ]
        .to_numpy()
    )


    mirrored[
        "away_timeouts_remaining"
    ] = (
        home_timeouts
        .to_numpy()
    )


    mirrored[
        "is_home_posteam"
    ] = (
        1.0
        -
        mirrored[
            "is_home_posteam"
        ]
    )


    mirrored_X = (
        mirrored[
            WP_FEATURES
        ]
        .to_numpy()
    )


    mirrored_probabilities = (
        wp_model
        .predict_proba(
            mirrored_X
        )
    )


    if original_on_home_axis:

        mirrored_class_value = 0

    else:

        mirrored_class_value = 2


    mirrored_probability = (
        mirrored_probabilities[
            :,
            WP_CLASS_INDEX[
                mirrored_class_value
            ],
        ]
    )


    return (
        0.5
        *
        (
            original_probability
            +
            mirrored_probability
        )
    )


# =========================================================
# Evaluate post-play states.
#
# Regulation:
#     resolve conservative clock-kill terminal wins exactly,
#     then leave all other states to the learned WP model.
#
# 2026 overtime:
#     resolve rule-defined terminal outcomes exactly.
#     Regular season may terminate when its 10-minute
#     period expires; postseason period boundaries continue
#     into another 15-minute period.
# =========================================================

def original_score_diff_from_wp_state(
    state,
    original_on_home_axis,
):

    home_diff = float(
        state[
            "home_score_differential"
        ]
    )

    if original_on_home_axis:

        return home_diff

    return -home_diff



# Development-data Try outcome rates.
#
# NFL play-by-play, 2015-2024.
#
# XP:
#   offense +1: 12,015 / 12,749
#   defense +2:      9 / 12,749
#   no score:       725 / 12,749
#
# Two-point attempt:
#   offense +2:   610 / 1,274
#   defense +2:     5 / 1,274
#   no score:      659 / 1,274
#
# Defensive Try returns became live scoring outcomes in
# the modern rules era, so use 2015+ consistently here.
XP_SUCCESS_PROB = (
    12015.0
    /
    12749.0
)

XP_DEFENSIVE_RETURN_PROB = (
    9.0
    /
    12749.0
)

XP_NO_SCORE_PROB = (
    725.0
    /
    12749.0
)


TWO_POINT_SUCCESS_PROB = (
    610.0
    /
    1274.0
)

TWO_POINT_DEFENSIVE_RETURN_PROB = (
    5.0
    /
    1274.0
)

TWO_POINT_NO_SCORE_PROB = (
    659.0
    /
    1274.0
)



def touchdown_try_required(
    base,
    state,
):

    scorer = int(
        state.get(
            "_touchdown_original",
            0,
        )
    )

    if scorer == 0:

        return False


    eps = 1e-9

    post_diff = (
        original_score_diff_from_wp_state(
            state,
            bool(
                base[
                    "_wp_original_on_home_axis"
                ]
            ),
        )
    )

    # Positive means the team that scored the touchdown
    # is ahead AFTER the six touchdown points.
    scorer_post_diff = (
        float(scorer)
        *
        float(post_diff)
    )


    # -----------------------------------------------------
    # Regulation.
    #
    # Normally every touchdown gets a Try.
    #
    # On a Q4-expiring touchdown, skip it if even a
    # successful Try cannot affect the game's result.
    # -----------------------------------------------------

    if int(base["qtr"]) < 5:

        if (
            int(base["qtr"]) == 4
            and
            float(
                state[
                    "game_seconds_remaining"
                ]
            )
            <= eps
        ):

            return (
                scorer_post_diff
                <= eps
                and
                scorer_post_diff
                >= -2.0 - eps
            )

        return True


    phase = base["ot_phase"]

    opportunity_completed = bool(
        state.get(
            "_ot_opportunity_completed",
            False,
        )
    )

    remaining_seconds = float(
        state[
            "game_seconds_remaining"
        ]
    )

    regular_season_ot = (
        str(
            base.get(
                "ot_format",
                "REGULAR_SEASON",
            )
        )
        ==
        "REGULAR_SEASON"
    )


    # -----------------------------------------------------
    # Ordinary sudden death:
    # touchdown itself ends the game.
    # -----------------------------------------------------

    if phase == "SUDDEN_DEATH":

        return False


    # -----------------------------------------------------
    # Regular-season clock expiration.
    #
    # If the TD itself already decides the winner, no Try.
    # If the scoring team is tied or within two points,
    # the untimed Try can still affect the result.
    # -----------------------------------------------------

    if (
        regular_season_ot
        and
        remaining_seconds <= eps
    ):

        return (
            scorer_post_diff
            <= eps
            and
            scorer_post_diff
            >= -2.0 - eps
        )


    # -----------------------------------------------------
    # Opening possession.
    #
    # Normal offensive TD:
    # opponent still receives its guaranteed opportunity,
    # so the scoring team takes its Try.
    #
    # Defensive TD, or a touchdown after the receiving
    # team's opportunity was already completed on a kick:
    # game is already over.
    # -----------------------------------------------------

    if phase == "OPENING":

        if scorer < 0:

            return False

        if opportunity_completed:

            return False

        return True


    # -----------------------------------------------------
    # Response possession.
    #
    # Both guaranteed opportunities are now ending.
    #
    # A Try is useful only if the scoring team is:
    #   tied,
    #   down 1,
    #   or down 2
    # after the six touchdown points.
    # -----------------------------------------------------

    if phase == "RESPONSE":

        return (
            scorer_post_diff
            <= eps
            and
            scorer_post_diff
            >= -2.0 - eps
        )


    return True



def touchdown_try_state(
    base,
    state,
    try_points,
):

    scorer = int(
        state.get(
            "_touchdown_original",
            0,
        )
    )

    if scorer not in {
        -1,
        1,
    }:

        raise ValueError(
            "touchdown_try_state requires "
            "_touchdown_original of -1 or 1."
        )


    child = dict(
        state
    )


    # Convert points scored by the touchdown team into the
    # original fourth-down team's perspective.
    original_points = (
        float(scorer)
        *
        float(try_points)
    )


    if bool(
        base[
            "_wp_original_on_home_axis"
        ]
    ):

        child[
            "home_score_differential"
        ] = (
            float(
                child[
                    "home_score_differential"
                ]
            )
            +
            original_points
        )

    else:

        child[
            "home_score_differential"
        ] = (
            float(
                child[
                    "home_score_differential"
                ]
            )
            -
            original_points
        )


    child[
        "_score_change_original"
    ] = (
        float(
            child.get(
                "_score_change_original",
                0.0,
            )
        )
        +
        original_points
    )


    # The Try is now resolved. Prevent recursive
    # re-processing as another touchdown.
    child[
        "_touchdown_original"
    ] = 0


    # The Try is untimed. Keep the game clock exactly where
    # the touchdown left it.
    return child



def evaluate_post_play_states(
    base,
    states,
):

    wp = (
        predict_original_team_wp(
            states,
            bool(
                base["_wp_original_on_home_axis"]
            ),
            (
                base["site"]
                == "NEUTRAL"
            ),
        )
        .astype(
            float,
            copy=True,
        )
    )


    # -----------------------------------------------------
    # Resolve touchdown Trys.
    #
    # Every touchdown state is stored as exactly +/-6.
    #
    # For a required Try:
    #   kick  -> +1 with XP_SUCCESS_PROB
    #   two   -> +2 with TWO_POINT_SUCCESS_PROB
    #
    # The touchdown-scoring team chooses the option that
    # maximizes its own win probability.
    #
    # Therefore:
    #   original team TD -> maximize original-team WP
    #   opponent TD      -> minimize original-team WP
    #
    # Evaluate all child states in one batch so this remains
    # practical for large Monte Carlo simulations.
    # -----------------------------------------------------

    try_resolved = np.zeros(
        len(states),
        dtype=bool,
    )

    try_children = []

    try_entries = []


    for index, state in enumerate(
        states
    ):

        scorer = int(
            state.get(
                "_touchdown_original",
                0,
            )
        )

        if scorer == 0:

            continue


        if not touchdown_try_required(
            base,
            state,
        ):

            continue


        start = len(
            try_children
        )


        # Failed Try: zero additional points.
        try_children.append(
            touchdown_try_state(
                base,
                state,
                0.0,
            )
        )


        # Successful XP.
        try_children.append(
            touchdown_try_state(
                base,
                state,
                1.0,
            )
        )


        # Successful two-point conversion.
        try_children.append(
            touchdown_try_state(
                base,
                state,
                2.0,
            )
        )


        # Defense returns the Try for two points.
        try_children.append(
            touchdown_try_state(
                base,
                state,
                -2.0,
            )
        )


        try_entries.append(
            (
                index,
                scorer,
                start,
            )
        )


    if try_children:

        child_wp = (
            evaluate_post_play_states(
                base,
                try_children,
            )
        )


        for (
            index,
            scorer,
            start,
        ) in try_entries:

            miss_wp = float(
                child_wp[
                    start
                ]
            )

            xp_success_wp = float(
                child_wp[
                    start + 1
                ]
            )

            two_success_wp = float(
                child_wp[
                    start + 2
                ]
            )

            defensive_return_wp = float(
                child_wp[
                    start + 3
                ]
            )


            xp_value = (
                XP_SUCCESS_PROB
                *
                xp_success_wp

                +

                XP_DEFENSIVE_RETURN_PROB
                *
                defensive_return_wp

                +

                XP_NO_SCORE_PROB
                *
                miss_wp
            )


            two_value = (
                TWO_POINT_SUCCESS_PROB
                *
                two_success_wp

                +

                TWO_POINT_DEFENSIVE_RETURN_PROB
                *
                defensive_return_wp

                +

                TWO_POINT_NO_SCORE_PROB
                *
                miss_wp
            )


            if scorer > 0:

                wp[index] = max(
                    xp_value,
                    two_value,
                )

            else:

                wp[index] = min(
                    xp_value,
                    two_value,
                )


            try_resolved[
                index
            ] = True



    # -----------------------------------------------------
    # Regulation clock-kill terminal states.
    #
    # If the original offense is leading in Q4, still owns
    # the ball, and can exhaust the remaining clock with
    # ordinary kneel-down runoffs after accounting for the
    # opponent's remaining timeouts, treat the state as a
    # terminal win.
    #
    # This is intentionally conservative: it does not credit
    # any clock runoff before the next snap.
    # -----------------------------------------------------

    if base["qtr"] < 5:

        if int(base["qtr"]) != 4:

            return wp


        original_on_home_axis = bool(
            base["_wp_original_on_home_axis"]
        )


        for index, state in enumerate(
            states
        ):

            if (
                int(
                    round(
                        float(
                            state["qtr"]
                        )
                    )
                )
                !=
                4
            ):

                continue


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
                            base["_wp_original_on_home_axis"]
                        )
                    ),
                )
            )


            if not possession_original:

                continue


            post_diff = (
                original_score_diff_from_wp_state(
                    state,
                    original_on_home_axis,
                )
            )


            if post_diff <= 0.0:

                continue


            down = float(
                state.get(
                    "down",
                    np.nan,
                )
            )


            if not np.isfinite(
                down
            ):

                continue


            down = int(
                round(
                    down
                )
            )


            if (
                down < 1
                or
                down > 4
            ):

                continue


            if original_on_home_axis:

                opponent_timeouts = float(
                    state[
                        "away_timeouts_remaining"
                    ]
                )

            else:

                opponent_timeouts = float(
                    state[
                        "home_timeouts_remaining"
                    ]
                )


            opponent_timeouts = int(
                np.clip(
                    round(
                        opponent_timeouts
                    ),
                    0,
                    3,
                )
            )


            runoff_windows = max(
                0,
                (
                    4
                    -
                    down
                )
                -
                opponent_timeouts,
            )


            drainable_seconds = (
                40.0
                *
                runoff_windows
            )


            if drainable_seconds <= 0.0:

                continue


            remaining_seconds = float(
                state[
                    "game_seconds_remaining"
                ]
            )


            if (
                remaining_seconds
                <=
                drainable_seconds
            ):

                wp[index] = 1.0


        return wp


    phase = base["ot_phase"]

    ot_format = str(
        base.get(
            "ot_format",
            "REGULAR_SEASON",
        )
    ).strip().upper()

    regular_season_ot = (
        ot_format == "REGULAR_SEASON"
    )

    eps = 1e-9

    original_on_home_axis = bool(
        base["_wp_original_on_home_axis"]
    )


    for index, state in enumerate(
        states
    ):

        if try_resolved[index]:

            continue


        post_diff = (
            original_score_diff_from_wp_state(
                state,
                original_on_home_axis,
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
                        base["_wp_original_on_home_axis"]
                    )
                ),
            )
        )


        # -------------------------------------------------
        # Response-possession touchdown while trailing by 8.
        #
        # A touchdown earns six points and must be followed
        # by a Try. Historical GO donors can contain bundled
        # +6 or +7 scoring sequences, but in this game state
        # the offense must attempt a two-point conversion.
        #
        # Failure -> terminal loss.
        # Success -> tie, then either:
        #   * regular-season 0:00: tie (utility 0.5)
        #   * otherwise: tied sudden-death continuation
        opportunity_completed = bool(
            state.get(
                "_ot_opportunity_completed",
                False,
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
            regular_season_ot
            and
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

            elif opportunity_completed:

                if post_diff > eps:

                    wp[index] = 1.0

                elif post_diff < -eps:

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
                or
                opportunity_completed
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
                base["_wp_original_on_home_axis"]
            ),
            (
                base["site"]
                == "NEUTRAL"
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

        if fg_distance > 70:

            return (
                False,
                "FG distance beyond extended 70-yard support",
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

    "goal_to_go",

    "qtr",
    "game_seconds_remaining",

    "score_differential",

    "posteam_timeouts_remaining",
    "defteam_timeouts_remaining",

    "short_yardage",
    "inside_10",
    "inside_20",
    "final_two_minutes",

    "is_pass",

    "fake_punt",
    "fake_fg",
]


# Validation-calibrated normal-run adjustment.
#
# Derived only from 2023-2024 validation data using
# supported normal-run states with 0 < ydstogo <= 3.
NORMAL_GO_RUN_LOGIT_SHIFT = 0.2925
NORMAL_GO_RUN_CALIBRATION_MAX_YDSTOGO = 3.0


def go_conversion_probability(
    base,
    action,
    *,
    apply_calibration=True,
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

        "short_yardage":
            int(
                base["ydstogo"]
                <= 1
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
                    "game_seconds_remaining"
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


    probability = float(
        go_model
        .predict_proba(X)[0, 1]
    )


    if (
        apply_calibration
        and
        action
        ==
        "NORMAL_GO_RUN"
        and
        0.0
        <
        float(base["ydstogo"])
        <=
        NORMAL_GO_RUN_CALIBRATION_MAX_YDSTOGO
    ):

        probability = float(
            np.clip(
                probability,
                1e-9,
                1.0 - 1e-9,
            )
        )

        logit = (
            np.log(
                probability
                /
                (1.0 - probability)
            )
            +
            NORMAL_GO_RUN_LOGIT_SHIFT
        )

        probability = float(
            1.0
            /
            (
                1.0
                +
                np.exp(-logit)
            )
        )


    return probability


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

        is_touchdown = (
            safe_float(
                donor.get(
                    "touchdown",
                    0.0,
                ),
                0.0,
            )
            >=
            0.5
        )


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

            touchdown_original=(
                1
                if is_touchdown
                else 0
            ),
        )


    # -----------------------------------------------------
    # Defensive score
    # -----------------------------------------------------

    if transition == "opponent_scored":

        is_touchdown = (
            safe_float(
                donor.get(
                    "touchdown",
                    0.0,
                ),
                0.0,
            )
            >=
            0.5
        )


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

            touchdown_original=(
                -1
                if is_touchdown
                else 0
            ),
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


        # -------------------------------------------------
        # Successful goal-to-go plays must borrow from
        # successful goal-to-go transitions.
        #
        # In the development pool, every true goal-to-go
        # conversion results in an offensive score. Allowing
        # converted non-goal-to-go donors here can incorrectly
        # turn a successful fourth-and-goal into an ordinary
        # first-down state short of the end zone.
        # -------------------------------------------------

        if (
            bool(outcome)
            and
            int(
                base[
                    "goal_to_go"
                ]
            )
            ==
            1
        ):

            donor_pool = donor_pool[
                donor_pool[
                    "yardline_100"
                ]
                <=
                donor_pool[
                    "ydstogo"
                ]
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


        # -------------------------------------------------
        # Two-minute-warning clock context.
        #
        # Empirical seconds_to_state is materially shorter
        # in the minute before the Q2/Q4 two-minute warning.
        # Preserve the joint transition donor, but favor
        # donors from similar pre-warning clock context.
        #
        # Apply this only to normal GO decisions; fake-play
        # behavior remains unchanged.
        # -------------------------------------------------

        base_half_seconds = float(
            base[
                "half_seconds_remaining"
            ]
        )


        use_warning_clock_context = (

            action
            in (
                "NORMAL_GO_RUN",
                "NORMAL_GO_PASS",
            )

            and

            int(
                base[
                    "qtr"
                ]
            )
            in (2, 4)

            and

            base_half_seconds
            > 120.0

            and

            base_half_seconds
            <= 180.0

        )


        if use_warning_clock_context:

            donor_qtr = (
                donor_pool[
                    "qtr"
                ]
                .astype(int)
                .to_numpy()
            )


            donor_game_seconds = (
                donor_pool[
                    "game_seconds_remaining"
                ]
                .astype(float)
                .to_numpy()
            )


            donor_half_seconds = (
                np.where(

                    donor_qtr == 2,

                    (
                        donor_game_seconds
                        -
                        1800.0
                    ),

                    np.where(

                        donor_qtr == 4,

                        donor_game_seconds,

                        np.nan,

                    ),

                )
            )


            clock_gap = (
                np.abs(
                    donor_half_seconds
                    -
                    base_half_seconds
                )
                /
                60.0
            )


            donor_warning_context = (

                np.isin(
                    donor_qtr,
                    [2, 4],
                )

                &

                np.isfinite(
                    donor_half_seconds
                )

                &

                (
                    donor_half_seconds
                    > 120.0
                )

                &

                (
                    donor_half_seconds
                    <= 180.0
                )

            )


            # Non-warning-context donors remain available
            # as sparse-data fallback, but receive very
            # little sampling weight.
            clock_gap = np.where(
                donor_warning_context,
                clock_gap,
                5.0,
            )


            donor_pool[
                "_clock_context_gap"
            ] = clock_gap


        else:

            donor_pool[
                "_clock_context_gap"
            ] = 0.0


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

        if use_warning_clock_context:

            clock_weights = np.exp(
                -
                donor_pool[
                    "_clock_context_gap"
                ]
                .astype(float)
                .to_numpy()
            )


            weight_sum = float(
                np.sum(
                    clock_weights
                )
            )


            if (
                np.all(
                    np.isfinite(
                        clock_weights
                    )
                )
                and
                np.isfinite(
                    weight_sum
                )
                and
                weight_sum > 0.0
            ):

                clock_weights = (
                    clock_weights
                    /
                    weight_sum
                )


                donor_indices = rng.choice(
                    len(
                        donor_pool
                    ),
                    size=count,
                    replace=True,
                    p=clock_weights,
                )

            else:

                donor_indices = (
                    rng.integers(
                        0,
                        len(
                            donor_pool
                        ),
                        size=count,
                    )
                )

        else:

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


    # ---------------------------------------------------------
    # Extreme-distance field-goal tail.
    #
    # Development data through 2024 becomes extremely sparse
    # beyond 60 yards. A simple development-only logistic audit
    # of 58+ yard attempts produced a distance logit slope of
    # -0.3080364923753445.
    #
    # Do not use that audit as a hard replacement because doing
    # so would create a discontinuity near the edge of support.
    # Instead:
    #
    # 1. Anchor at this production model's own 60-yard make
    #    probability for the current roof / kicker context.
    # 2. Apply the frozen long-distance logit slope beyond 60.
    # 3. Never increase the production model's make probability.
    # 4. Preserve blocked and broken probabilities from their
    #    existing models; missed probability absorbs the
    #    remaining failure mass.
    #
    # FIELD_GOAL eligibility remains capped at 70 yards
    # elsewhere in the engine.
    # ---------------------------------------------------------

    if distance > 60.0:

        anchor_distance = 60.0

        anchor_make_row = (
            make_row.copy()
        )

        anchor_make_row[
            "fg_distance_estimate"
        ] = anchor_distance

        anchor_make_row[
            "fg_distance_sq"
        ] = (
            anchor_distance ** 2
        )

        anchor_make_clean = float(

            fg_bundle[
                "make_model"
            ]
            .predict_proba(

                anchor_make_row[
                    fg_bundle[
                        "make_features"
                    ]
                ]

            )[0, 1]

        )

        anchor_p_made = (
            p_clean
            *
            anchor_make_clean
        )

        eps = 1e-9

        anchor_p_made = float(
            np.clip(
                anchor_p_made,
                eps,
                1.0 - eps,
            )
        )

        tail_logit_slope = (
            -0.3080364923753445
        )

        anchor_logit = (
            np.log(
                anchor_p_made
                /
                (
                    1.0
                    -
                    anchor_p_made
                )
            )
        )

        tail_logit = (
            anchor_logit
            +
            tail_logit_slope
            *
            (
                distance
                -
                anchor_distance
            )
        )

        tail_p_made = float(
            1.0
            /
            (
                1.0
                +
                np.exp(
                    -tail_logit
                )
            )
        )

        probabilities[
            "made"
        ] = min(
            probabilities[
                "made"
            ],
            tail_p_made,
        )

        probabilities[
            "missed"
        ] = max(
            0.0,
            (
                1.0
                -
                probabilities[
                    "made"
                ]
                -
                probabilities[
                    "blocked"
                ]
                -
                probabilities[
                    "broken"
                ]
            ),
        )

        tail_total = sum(
            probabilities.values()
        )

        for key in probabilities:

            probabilities[key] /= (
                tail_total
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
# Rule 16 blocked-FG muff recovery
# =========================================================

FG_RULE16_MUFF_RECOVERY_PROB = (
    1.0
    /
    248.0
)

FG_RULE16_MUFF_ELAPSED_SECONDS = 7.0

FG_RULE16_MUFF_YARDLINE_RESIDUAL = 0.0


def blocked_fg_live_probabilities():

    p_muff = (
        FG_RULE16_MUFF_RECOVERY_PROB
    )

    remaining = (
        1.0
        -
        p_muff
    )

    baseline_score = (
        blocked_fg_score_probability()
    )

    p_score = (
        baseline_score
        *
        remaining
    )

    p_ordinary = (
        (
            1.0
            -
            baseline_score
        )
        *
        remaining
    )

    total = (
        p_score
        +
        p_muff
        +
        p_ordinary
    )

    if not np.isclose(
        total,
        1.0,
    ):

        raise RuntimeError(
            "Blocked-FG live probabilities "
            "do not sum to 1."
        )


    return {
        "opponent_scored":
            p_score,

        "kicking_team_muff_recovery":
            p_muff,

        "opponent_ball_no_score":
            p_ordinary,
    }


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


    blocked_live = (
        blocked_fg_live_probabilities()
    )

    p_block_score = (
        blocked_live[
            "opponent_scored"
        ]
    )

    p_block_muff = (
        blocked_live[
            "kicking_team_muff_recovery"
        ]
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

        live_roll = (
            rng.random()
        )

        if (
            live_roll
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
                        -6.0,

                    is_kickoff=True,

                    touchdown_original=-1,
                )

            )

            continue


        # -------------------------------------------------
        # RECEIVING-TEAM MUFF, KICKING TEAM RECOVERS
        #
        # 2025 CAR-NO empirical transition:
        #   recovery at the original fourth-down yardline,
        #   first-and-10,
        #   seven seconds elapsed.
        #
        # Under Rule 16, because the kick crossed the line
        # and the receiving team muffed it, that team has
        # completed its opportunity to possess even though
        # the original kicking team recovered.
        # -------------------------------------------------

        if (
            live_roll
            <
            (
                p_block_score
                +
                p_block_muff
            )
        ):

            recovery_yardline = (
                clip_yardline(
                    float(
                        base[
                            "yardline_100"
                        ]
                    )
                    +
                    FG_RULE16_MUFF_YARDLINE_RESIDUAL
                )
            )

            states.append(

                make_wp_state(

                    base,

                    possession_original=True,

                    yardline_100=
                        recovery_yardline,

                    down=1,

                    ydstogo=min(
                        10.0,
                        recovery_yardline,
                    ),

                    elapsed_seconds=
                        FG_RULE16_MUFF_ELAPSED_SECONDS,

                    ot_opportunity_completed=True,
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

    "site_advantage",
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


    donor_touchback = (
        safe_float(
            donor.get(
                "pbp_touchback",
                0.0,
            ),
            0.0,
        )
        >=
        0.5
    )


    if donor_touchback:

        receiving_yardline = 80.0

    else:

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


    regular_punt = (
        str(
            donor.get(
                "punt_branch",
                "",
            )
        )
        ==
        "REGULAR"
    )

    receiving_team_lost_ball = (
        safe_float(
            donor.get(
                "pbp_fumble_lost",
                0.0,
            ),
            0.0,
        )
        >=
        0.5
    )

    is_touchdown = (
        safe_float(
            donor.get(
                "touchdown",
                0.0,
            ),
            0.0,
        )
        >=
        0.5
    )


    receiving_team_safety = (
        safe_float(
            donor.get(
                "safety",
                0.0,
            ),
            0.0,
        )
        >=
        0.5
    )

    opportunity_completed = (
        regular_punt
        and
        transition
        in {
            "kicking_team_ball_no_score",
            "kicking_team_scored",
        }
        and
        (
            receiving_team_lost_ball
            or
            receiving_team_safety
        )
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

            possession_original=(
                False
                if receiving_team_safety
                else True
            ),

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

            touchdown_original=(
                -1
                if is_touchdown
                else 0
            ),
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

            possession_original=(
                True
                if receiving_team_safety
                else False
            ),

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

            touchdown_original=(
                1
                if is_touchdown
                else 0
            ),

            ot_opportunity_completed=
                opportunity_completed,
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

            ot_opportunity_completed=
                opportunity_completed,
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

        # Current offense is playing at home.
        "site":
            "HOME",

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