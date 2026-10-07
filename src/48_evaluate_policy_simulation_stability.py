import importlib.util
from pathlib import Path

import nflreadpy as nfl
import numpy as np
import pandas as pd

from sklearn.metrics import (
    log_loss,
    roc_auc_score,
)


# =========================================================
# Load frozen production decision engine.
# =========================================================

print("\nLoading frozen decision engine...")

spec = importlib.util.spec_from_file_location(
    "fourth_down_engine",
    "src/42_build_decision_engine.py",
)

engine = importlib.util.module_from_spec(spec)
spec.loader.exec_module(engine)

print("Decision engine loaded.")


# =========================================================
# Load 2023-2024 validation fourth downs.
# =========================================================

validation = pd.read_parquet(
    "data/fourth_down_modeling_split.parquet"
)

validation = validation[
    validation["split"] == "validation"
].copy()

print(
    f"\nValidation fourth downs: "
    f"{len(validation):,}"
)

print(
    validation["season"]
    .value_counts()
    .sort_index()
    .to_string()
)


# =========================================================
# Load roof information from raw 2023-2024 PBP.
# =========================================================

print("\nLoading 2023-2024 PBP...")

pbp = (
    nfl.load_pbp([2023, 2024])
    .sort([
        "game_id",
        "play_id",
    ])
    .to_pandas()
)

if "roof" in pbp.columns:

    game_roof = (
        pbp[
            [
                "game_id",
                "roof",
            ]
        ]
        .dropna(
            subset=["roof"]
        )
        .drop_duplicates(
            subset=["game_id"]
        )
        .set_index(
            "game_id"
        )["roof"]
    )

    if "roof" not in validation.columns:
        validation["roof"] = (
            validation["game_id"]
            .map(game_roof)
        )
    else:
        missing = validation["roof"].isna()

        validation.loc[
            missing,
            "roof",
        ] = (
            validation.loc[
                missing,
                "game_id",
            ]
            .map(game_roof)
        )


# =========================================================
# Load OT phase metadata.
# =========================================================

OT_STATES_PATH = Path(
    "data/overtime_states.parquet"
)

if not OT_STATES_PATH.exists():

    raise FileNotFoundError(
        "Missing data/overtime_states.parquet. "
        "Build it first with: "
        "python src/45_build_overtime_state_dataset.py"
    )


ot_states = pd.read_parquet(
    OT_STATES_PATH
)

ot_states = ot_states[
    ot_states["season"].isin(
        [2023, 2024]
    )
].copy()


OT_PHASE_MAP = {
    "OPENING_POSSESSION": "OPENING",
    "SECOND_POSSESSION": "RESPONSE",
    "SUDDEN_DEATH": "SUDDEN_DEATH",
}


OT_STATE_LOOKUP = {}

for _, row in ot_states.iterrows():

    phase = OT_PHASE_MAP.get(
        str(row["ot_phase"])
    )

    if phase is None:
        continue

    key = (
        str(row["game_id"]),
        int(
            round(
                float(row["play_id"])
            )
        ),
    )

    OT_STATE_LOOKUP[key] = {
        "ot_phase":
            phase,

        "ot_format":
            (
                "POSTSEASON"
                if str(
                    row["season_type"]
                ) == "POST"
                else
                "REGULAR_SEASON"
            ),
    }


# =========================================================
# State conversion.
# =========================================================

def valid_number(value):

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


def numeric_or(
    value,
    default,
):

    if valid_number(value):
        return float(value)

    return float(default)


def row_to_state(row):

    required = [
        "qtr",
        "game_seconds_remaining",
        "yardline_100",
        "ydstogo",
        "score_differential",
        "is_home",
    ]

    for column in required:

        if (
            column not in row.index
            or
            not valid_number(
                row[column]
            )
        ):
            return None

    state = {
        "qtr":
            int(row["qtr"]),

        "game_seconds_remaining":
            float(
                row[
                    "game_seconds_remaining"
                ]
            ),

        "yardline_100":
            float(
                row["yardline_100"]
            ),

        "ydstogo":
            float(
                row["ydstogo"]
            ),

        "score_differential":
            float(
                row[
                    "score_differential"
                ]
            ),

        "posteam_timeouts_remaining":
            numeric_or(
                row.get(
                    "posteam_timeouts_remaining",
                    3,
                ),
                3,
            ),

        "defteam_timeouts_remaining":
            numeric_or(
                row.get(
                    "defteam_timeouts_remaining",
                    3,
                ),
                3,
            ),

        "is_home":
            int(row["is_home"]),

        "spread_line":
            numeric_or(
                row.get(
                    "spread_line",
                    0,
                ),
                0,
            ),

        "total_line":
            numeric_or(
                row.get(
                    "total_line",
                    45,
                ),
                45,
            ),

        "roof":
            (
                str(
                    row.get(
                        "roof",
                        "outdoors",
                    )
                )
                if pd.notna(
                    row.get(
                        "roof",
                        np.nan,
                    )
                )
                else
                "outdoors"
            ),
    }

    if (
        "goal_to_go" in row.index
        and
        valid_number(
            row["goal_to_go"]
        )
    ):
        state["goal_to_go"] = int(
            row["goal_to_go"]
        )

    if state["qtr"] >= 5:

        key = (
            str(row["game_id"]),
            int(
                round(
                    float(row["play_id"])
                )
            ),
        )

        metadata = (
            OT_STATE_LOOKUP.get(key)
        )

        if metadata is None:
            raise RuntimeError(
                "Missing OT metadata for "
                f"{key[0]} play {key[1]}"
            )

        state["ot_phase"] = (
            metadata["ot_phase"]
        )

        state["ot_format"] = (
            metadata["ot_format"]
        )

        state["ot_period"] = max(
            1,
            state["qtr"] - 4,
        )

    return state


# =========================================================
# Validation-only policy simulation stability study.
#
# Development data: 2023-2024 validation only.
#
# Candidate adaptive rule being evaluated:
#
#   1. Start with 300 simulations/action.
#   2. If top-two gap < 1.96 combined MC SE,
#      rerun from scratch with 1,200 simulations/action.
#
# Nothing here changes production behavior.
# =========================================================

BASE_SIMULATIONS = 300
HIGH_SIMULATIONS = 1200

UNCERTAINTY_Z = 1.96

SAMPLE_PER_SEASON = 150

SEED_SAMPLE_2023 = 480023
SEED_SAMPLE_2024 = 480024

SEED_BASE_A = 481000
SEED_BASE_B = 482000

SEED_HIGH_A = 483000
SEED_HIGH_B = 484000


# =========================================================
# Build validation policy population.
# =========================================================

valid_indices = []


for index, row in validation.iterrows():

    state = row_to_state(row)

    if state is None:
        continue

    try:

        engine.normalize_state(
            state
        )

    except (
        TypeError,
        ValueError,
    ):

        continue

    valid_indices.append(
        index
    )


policy_df = (
    validation.loc[
        valid_indices
    ]
    .copy()
)


sample_parts = []


for season, seed in [
    (
        2023,
        SEED_SAMPLE_2023,
    ),
    (
        2024,
        SEED_SAMPLE_2024,
    ),
]:

    season_rows = (
        policy_df[
            policy_df["season"]
            ==
            season
        ]
    )

    n = min(
        SAMPLE_PER_SEASON,
        len(season_rows),
    )

    sample_parts.append(
        season_rows.sample(
            n=n,
            random_state=seed,
        )
    )


policy_sample = (
    pd.concat(
        sample_parts,
        ignore_index=True,
    )
    .reset_index(
        drop=True
    )
)


print(
    "\nVALIDATION POLICY SIMULATION STABILITY"
)

print(
    f"Eligible validation rows: "
    f"{len(policy_df):,}"
)

print(
    f"Study rows: "
    f"{len(policy_sample):,}"
)

print(
    policy_sample[
        "season"
    ]
    .value_counts()
    .sort_index()
    .to_string()
)


# =========================================================
# Recommendation summary helper.
# =========================================================

def recommendation_summary(
    state,
    n_simulations,
    seed,
):

    result = engine.recommend(

        state,

        n_simulations=
            n_simulations,

        seed=
            seed,
    )


    eligible = (
        result["results"][
            result["results"][
                "eligible"
            ]
        ]
        .sort_values(
            "expected_win_probability",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )


    if len(eligible) == 0:

        return {
            "action":
                None,

            "gap_pct":
                np.nan,

            "combined_se_pct":
                np.nan,

            "z_gap":
                np.nan,
        }


    best = eligible.iloc[0]


    if len(eligible) < 2:

        return {
            "action":
                best["action"],

            "gap_pct":
                np.nan,

            "combined_se_pct":
                np.nan,

            "z_gap":
                np.inf,
        }


    second = eligible.iloc[1]


    gap = float(
        best[
            "expected_win_probability"
        ]
        -
        second[
            "expected_win_probability"
        ]
    )


    best_se = float(
        best["mc_se"]
    )

    second_se = float(
        second["mc_se"]
    )


    if (
        np.isfinite(best_se)
        and
        np.isfinite(second_se)
    ):

        combined_se = float(
            np.sqrt(
                best_se ** 2
                +
                second_se ** 2
            )
        )

    else:

        combined_se = np.nan


    if (
        np.isfinite(combined_se)
        and
        combined_se > 0.0
    ):

        z_gap = float(
            gap
            /
            combined_se
        )

    elif gap > 0.0:

        z_gap = np.inf

    else:

        z_gap = np.nan


    return {
        "action":
            best["action"],

        "gap_pct":
            100.0 * gap,

        "combined_se_pct":
            (
                100.0 * combined_se
                if np.isfinite(
                    combined_se
                )
                else
                np.nan
            ),

        "z_gap":
            z_gap,
    }


# =========================================================
# Two independent realizations of:
#
#   A) fixed 300 simulations
#   B) candidate adaptive procedure
#
# Comparing A vs B seed realizations tells us how stable the
# numerical recommendation is.
# =========================================================

records = []


for i, row in (
    policy_sample.iterrows()
):

    if i % 25 == 0:

        print(
            f"Evaluating "
            f"{i:,}/"
            f"{len(policy_sample):,}..."
        )


    state = row_to_state(
        row
    )


    base_a = recommendation_summary(

        state,

        BASE_SIMULATIONS,

        SEED_BASE_A
        +
        i,
    )


    base_b = recommendation_summary(

        state,

        BASE_SIMULATIONS,

        SEED_BASE_B
        +
        i,
    )


    trigger_a = bool(
        np.isfinite(
            base_a["z_gap"]
        )
        and
        base_a["z_gap"]
        <
        UNCERTAINTY_Z
    )


    trigger_b = bool(
        np.isfinite(
            base_b["z_gap"]
        )
        and
        base_b["z_gap"]
        <
        UNCERTAINTY_Z
    )


    high_a = None
    high_b = None


    if trigger_a:

        high_a = (
            recommendation_summary(

                state,

                HIGH_SIMULATIONS,

                SEED_HIGH_A
                +
                i,
            )
        )


    if trigger_b:

        high_b = (
            recommendation_summary(

                state,

                HIGH_SIMULATIONS,

                SEED_HIGH_B
                +
                i,
            )
        )


    final_a = (
        high_a
        if high_a is not None
        else base_a
    )

    final_b = (
        high_b
        if high_b is not None
        else base_b
    )


    records.append({

        "season":
            int(
                row["season"]
            ),

        "game_id":
            row.get(
                "game_id",
                None,
            ),

        "play_id":
            row.get(
                "play_id",
                None,
            ),

        "base_a_action":
            base_a["action"],

        "base_b_action":
            base_b["action"],

        "base_a_gap_pct":
            base_a["gap_pct"],

        "base_b_gap_pct":
            base_b["gap_pct"],

        "base_a_combined_se_pct":
            base_a[
                "combined_se_pct"
            ],

        "base_b_combined_se_pct":
            base_b[
                "combined_se_pct"
            ],

        "base_a_z":
            base_a["z_gap"],

        "base_b_z":
            base_b["z_gap"],

        "trigger_a":
            trigger_a,

        "trigger_b":
            trigger_b,

        "high_a_action":
            (
                high_a["action"]
                if high_a is not None
                else
                None
            ),

        "high_b_action":
            (
                high_b["action"]
                if high_b is not None
                else
                None
            ),

        "high_a_z":
            (
                high_a["z_gap"]
                if high_a is not None
                else
                np.nan
            ),

        "high_b_z":
            (
                high_b["z_gap"]
                if high_b is not None
                else
                np.nan
            ),

        "final_a_action":
            final_a["action"],

        "final_b_action":
            final_b["action"],

        "final_a_z":
            final_a["z_gap"],

        "final_b_z":
            final_b["z_gap"],
    })


results = pd.DataFrame(
    records
)


results[
    "base_flip"
] = (
    results[
        "base_a_action"
    ]
    !=
    results[
        "base_b_action"
    ]
)


results[
    "adaptive_flip"
] = (
    results[
        "final_a_action"
    ]
    !=
    results[
        "final_b_action"
    ]
)


results[
    "final_a_unresolved"
] = (
    np.isfinite(
        results["final_a_z"]
    )
    &
    (
        results["final_a_z"]
        <
        UNCERTAINTY_Z
    )
)


results[
    "final_b_unresolved"
] = (
    np.isfinite(
        results["final_b_z"]
    )
    &
    (
        results["final_b_z"]
        <
        UNCERTAINTY_Z
    )
)


results[
    "both_final_resolved"
] = (
    ~results[
        "final_a_unresolved"
    ]
    &
    ~results[
        "final_b_unresolved"
    ]
)


results[
    "either_final_unresolved"
] = (
    results[
        "final_a_unresolved"
    ]
    |
    results[
        "final_b_unresolved"
    ]
)


results[
    "final_confidence_disagreement"
] = (
    results[
        "final_a_unresolved"
    ]
    !=
    results[
        "final_b_unresolved"
    ]
)


results[
    "either_trigger"
] = (
    results[
        "trigger_a"
    ]
    |
    results[
        "trigger_b"
    ]
)


results[
    "both_resolved"
] = (
    ~results[
        "trigger_a"
    ]
    &
    ~results[
        "trigger_b"
    ]
)


# =========================================================
# Summary.
# =========================================================

print(
    "\n300-SIMULATION BASELINE STABILITY"
)

print(
    f"Seed-to-seed recommendation flip rate: "
    f"{100.0 * results['base_flip'].mean():.2f}%"
)


resolved = (
    results[
        results[
            "both_resolved"
        ]
    ]
)

uncertain = (
    results[
        results[
            "either_trigger"
        ]
    ]
)


print(
    f"Both seeds resolved: "
    f"{len(resolved):,}"
)

if len(resolved) > 0:

    print(
        "Flip rate when both resolved: "
        f"{100.0 * resolved['base_flip'].mean():.2f}%"
    )


print(
    f"Either seed unresolved: "
    f"{len(uncertain):,}"
)

if len(uncertain) > 0:

    print(
        "Flip rate when either unresolved: "
        f"{100.0 * uncertain['base_flip'].mean():.2f}%"
    )


print(
    "\nCANDIDATE ADAPTIVE POLICY"
)

print(
    f"Initial simulations/action: "
    f"{BASE_SIMULATIONS:,}"
)

print(
    f"Rerun simulations/action:   "
    f"{HIGH_SIMULATIONS:,}"
)

print(
    f"Trigger threshold: "
    f"z < {UNCERTAINTY_Z:.2f}"
)


trigger_rate_a = float(
    results[
        "trigger_a"
    ].mean()
)

trigger_rate_b = float(
    results[
        "trigger_b"
    ].mean()
)


print(
    f"Trigger rate — seed A: "
    f"{100.0 * trigger_rate_a:.2f}%"
)

print(
    f"Trigger rate — seed B: "
    f"{100.0 * trigger_rate_b:.2f}%"
)


print(
    f"Adaptive seed-to-seed flip rate: "
    f"{100.0 * results['adaptive_flip'].mean():.2f}%"
)



final_resolved = (
    results[
        results[
            "both_final_resolved"
        ]
    ]
)

final_unresolved = (
    results[
        results[
            "either_final_unresolved"
        ]
    ]
)


print(
    f"Both adaptive seeds resolved: "
    f"{len(final_resolved):,}"
)

if len(final_resolved) > 0:

    print(
        "Flip rate when both adaptive "
        "seeds resolved: "
        f"{100.0 * final_resolved['adaptive_flip'].mean():.2f}%"
    )


print(
    f"Either adaptive seed unresolved: "
    f"{len(final_unresolved):,}"
)

if len(final_unresolved) > 0:

    print(
        "Flip rate when either adaptive "
        "seed unresolved: "
        f"{100.0 * final_unresolved['adaptive_flip'].mean():.2f}%"
    )


print(
    "Adaptive confidence-state disagreement: "
    f"{100.0 * results['final_confidence_disagreement'].mean():.2f}%"
)


average_trigger_rate = (
    0.5
    *
    (
        trigger_rate_a
        +
        trigger_rate_b
    )
)


average_simulations = (
    BASE_SIMULATIONS
    +
    average_trigger_rate
    *
    HIGH_SIMULATIONS
)


print(
    "Estimated average simulations/action: "
    f"{average_simulations:.1f}"
)

print(
    "Savings versus 1,200 everywhere: "
    f"{100.0 * (1.0 - average_simulations / HIGH_SIMULATIONS):.2f}%"
)


triggered_a = (
    results[
        results[
            "trigger_a"
        ]
    ]
)

triggered_b = (
    results[
        results[
            "trigger_b"
        ]
    ]
)


if len(triggered_a) > 0:

    print(
        "Still unresolved after 1,200 — seed A: "
        f"{100.0 * (triggered_a['high_a_z'] < UNCERTAINTY_Z).mean():.2f}%"
    )


if len(triggered_b) > 0:

    print(
        "Still unresolved after 1,200 — seed B: "
        f"{100.0 * (triggered_b['high_b_z'] < UNCERTAINTY_Z).mean():.2f}%"
    )


print(
    "\nIMPORTANT"
)

print(
    "This experiment uses only 2023-2024 validation data."
)

print(
    "2025 remains held out and is not used to select "
    "the adaptive simulation rule."
)


output_path = (
    "data/"
    "evaluation_validation_policy_"
    "simulation_stability.parquet"
)

results.to_parquet(
    output_path,
    index=False,
)

print(
    f"\nSaved: {output_path}"
)
