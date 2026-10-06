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
# Reporting helpers.
# =========================================================

def print_binary_metrics(
    name,
    actual,
    predicted,
):

    y = np.asarray(
        actual,
        dtype=float,
    )

    p = np.asarray(
        predicted,
        dtype=float,
    )

    print(f"\n{name}")
    print(f"Rows:        {len(y):,}")

    print(
        f"Logloss:     "
        f"{log_loss(y, p, labels=[0, 1]):.5f}"
    )

    print(
        f"Brier:       "
        f"{np.mean((p - y) ** 2):.5f}"
    )

    if len(np.unique(y)) > 1:
        print(
            f"AUC:         "
            f"{roc_auc_score(y, p):.5f}"
        )

    print(
        f"Actual rate: "
        f"{y.mean():.5f}"
    )

    print(
        f"Pred rate:   "
        f"{p.mean():.5f}"
    )


def print_calibration_table(
    name,
    actual,
    predicted,
    n_bins=10,
):

    calibration = pd.DataFrame({
        "actual":
            np.asarray(
                actual,
                dtype=float,
            ),

        "predicted":
            np.asarray(
                predicted,
                dtype=float,
            ),
    })

    calibration = calibration[
        np.isfinite(
            calibration["actual"]
        )
        &
        np.isfinite(
            calibration["predicted"]
        )
    ].copy()

    edges = np.linspace(
        0.0,
        1.0,
        n_bins + 1,
    )

    calibration[
        "probability_bin"
    ] = pd.cut(
        calibration["predicted"],
        bins=edges,
        include_lowest=True,
    )

    table = (
        calibration
        .groupby(
            "probability_bin",
            observed=True,
        )
        .agg(
            plays=(
                "actual",
                "size",
            ),
            mean_predicted=(
                "predicted",
                "mean",
            ),
            actual_rate=(
                "actual",
                "mean",
            ),
        )
        .reset_index()
    )

    table[
        "calibration_gap"
    ] = (
        table["actual_rate"]
        -
        table["mean_predicted"]
    )

    ece = (
        table["plays"]
        *
        table[
            "calibration_gap"
        ].abs()
    ).sum() / table["plays"].sum()

    print(
        f"\n{name} CALIBRATION"
    )

    print(
        table.to_string(
            index=False,
            formatters={
                "mean_predicted":
                    lambda x: f"{x:.3f}",

                "actual_rate":
                    lambda x: f"{x:.3f}",

                "calibration_gap":
                    lambda x: f"{x:+.3f}",
            },
        )
    )

    print(
        f"Weighted absolute "
        f"calibration error: "
        f"{ece:.4f}"
    )


# =========================================================
# GO conversion validation.
# =========================================================

go_test = validation[
    validation["action"].isin(
        engine.GO_ACTIONS.keys()
    )
    &
    validation["converted"].notna()
].copy()


go_records = []


for _, row in go_test.iterrows():

    state = row_to_state(row)

    if state is None:
        continue

    base = engine.normalize_state(
        state
    )

    probability = (
        engine
        .go_conversion_probability(
            base,
            row["action"],
        )
    )

    go_records.append({
        "action":
            row["action"],

        "actual":
            int(row["converted"]),

        "predicted":
            probability,

        "season":
            int(row["season"]),

        "ydstogo":
            float(row["ydstogo"]),

        "yardline_100":
            float(row["yardline_100"]),
    })


go_eval = pd.DataFrame(
    go_records
)


print_binary_metrics(
    "2023-2024 VALIDATION — GO CONVERSION",
    go_eval["actual"],
    go_eval["predicted"],
)


print("\nGO BY ACTION")

print(
    go_eval
    .groupby("action")
    .agg(
        plays=(
            "actual",
            "size",
        ),
        actual_conversion=(
            "actual",
            "mean",
        ),
        predicted_conversion=(
            "predicted",
            "mean",
        ),
    )
    .sort_index()
    .to_string()
)


print_calibration_table(
    "2023-2024 VALIDATION — GO CONVERSION",
    go_eval["actual"],
    go_eval["predicted"],
)




# =========================================================
# Normal-run calibration stability.
# =========================================================

run_eval = go_eval[
    go_eval["action"]
    ==
    "NORMAL_GO_RUN"
].copy()


print(
    "\nNORMAL GO RUN — BY VALIDATION SEASON"
)

print(
    run_eval
    .groupby("season")
    .agg(
        plays=(
            "actual",
            "size",
        ),
        actual_conversion=(
            "actual",
            "mean",
        ),
        predicted_conversion=(
            "predicted",
            "mean",
        ),
    )
    .assign(
        gap=lambda x:
            x["actual_conversion"]
            -
            x["predicted_conversion"]
    )
    .to_string()
)


run_eval["yards_to_go_bucket"] = pd.cut(
    run_eval["ydstogo"],
    bins=[
        0.0,
        1.0,
        2.0,
        3.0,
    ],
    include_lowest=True,
    labels=[
        "1 yard",
        "2 yards",
        "3 yards",
    ],
)


print(
    "\nNORMAL GO RUN — BY YARDS TO GO"
)

print(
    run_eval
    .groupby(
        "yards_to_go_bucket",
        observed=True,
    )
    .agg(
        plays=(
            "actual",
            "size",
        ),
        actual_conversion=(
            "actual",
            "mean",
        ),
        predicted_conversion=(
            "predicted",
            "mean",
        ),
    )
    .assign(
        gap=lambda x:
            x["actual_conversion"]
            -
            x["predicted_conversion"]
    )
    .to_string()
)




# =========================================================
# Cross-year run-only calibration.
#
# Fit a single log-odds shift on one validation season
# and evaluate it on the other. This tests whether the
# observed run underprediction is stable out of season.
#
# Only use the production-supported normal-run range.
# =========================================================

supported_run_eval = run_eval[
    (run_eval["ydstogo"] > 0)
    &
    (run_eval["ydstogo"] <= 3)
].copy()


def apply_logit_shift(
    probability,
    delta,
):

    p = np.clip(
        np.asarray(
            probability,
            dtype=float,
        ),
        1e-6,
        1.0 - 1e-6,
    )

    logit = np.log(
        p / (1.0 - p)
    )

    shifted = (
        logit
        +
        float(delta)
    )

    return (
        1.0
        /
        (
            1.0
            +
            np.exp(-shifted)
        )
    )


def binary_logloss(
    actual,
    predicted,
):

    y = np.asarray(
        actual,
        dtype=float,
    )

    p = np.clip(
        np.asarray(
            predicted,
            dtype=float,
        ),
        1e-9,
        1.0 - 1e-9,
    )

    return float(
        -np.mean(
            y * np.log(p)
            +
            (1.0 - y)
            *
            np.log(1.0 - p)
        )
    )


def binary_brier(
    actual,
    predicted,
):

    y = np.asarray(
        actual,
        dtype=float,
    )

    p = np.asarray(
        predicted,
        dtype=float,
    )

    return float(
        np.mean(
            (p - y) ** 2
        )
    )


def fit_logit_shift(
    actual,
    predicted,
):

    # Dense deterministic grid.
    # A single intercept shift preserves ranking.
    grid = np.linspace(
        -1.0,
        1.0,
        4001,
    )

    losses = []

    for delta in grid:

        adjusted = (
            apply_logit_shift(
                predicted,
                delta,
            )
        )

        losses.append(
            binary_logloss(
                actual,
                adjusted,
            )
        )

    best_index = int(
        np.argmin(losses)
    )

    return float(
        grid[best_index]
    )


def evaluate_cross_year(
    fit_year,
    test_year,
):

    fit_frame = (
        supported_run_eval[
            supported_run_eval["season"]
            ==
            fit_year
        ]
    )

    test_frame = (
        supported_run_eval[
            supported_run_eval["season"]
            ==
            test_year
        ]
    )

    delta = fit_logit_shift(
        fit_frame["actual"],
        fit_frame["predicted"],
    )

    raw = (
        test_frame["predicted"]
        .to_numpy()
    )

    adjusted = (
        apply_logit_shift(
            raw,
            delta,
        )
    )

    actual = (
        test_frame["actual"]
        .to_numpy()
    )

    print(
        f"\nFIT {fit_year} -> TEST {test_year}"
    )

    print(
        f"Fit rows:      "
        f"{len(fit_frame):,}"
    )

    print(
        f"Test rows:     "
        f"{len(test_frame):,}"
    )

    print(
        f"Logit shift:   "
        f"{delta:+.4f}"
    )

    print(
        f"Actual rate:   "
        f"{actual.mean():.5f}"
    )

    print(
        f"Raw pred:      "
        f"{raw.mean():.5f}"
    )

    print(
        f"Adjusted pred: "
        f"{adjusted.mean():.5f}"
    )

    print(
        f"Logloss:       "
        f"{binary_logloss(actual, raw):.5f}"
        f" -> "
        f"{binary_logloss(actual, adjusted):.5f}"
    )

    print(
        f"Brier:         "
        f"{binary_brier(actual, raw):.5f}"
        f" -> "
        f"{binary_brier(actual, adjusted):.5f}"
    )

    return delta


print(
    "\nNORMAL GO RUN — CROSS-YEAR CALIBRATION"
)

delta_2023 = evaluate_cross_year(
    2023,
    2024,
)

delta_2024 = evaluate_cross_year(
    2024,
    2023,
)


combined_delta = fit_logit_shift(
    supported_run_eval["actual"],
    supported_run_eval["predicted"],
)


print(
    "\nCOMBINED 2023-2024 CANDIDATE"
)

print(
    f"Supported rows: "
    f"{len(supported_run_eval):,}"
)

print(
    f"Candidate logit shift: "
    f"{combined_delta:+.4f}"
)

print(
    "This candidate is diagnostic only; "
    "production is unchanged."
)


# =========================================================
# Field-goal validation.
# =========================================================

def actual_fg_made(row):

    value = str(
        row.get(
            "field_goal_result",
            "",
        )
    ).lower()

    if "made" in value:
        return 1

    if (
        "miss" in value
        or
        "block" in value
    ):
        return 0

    if (
        str(
            row.get(
                "execution_status",
                "",
            )
        ).upper()
        ==
        "BROKEN"
    ):
        return 0

    return None


fg_test = validation[
    validation["action"]
    ==
    "FIELD_GOAL"
].copy()


fg_records = []


for _, row in fg_test.iterrows():

    actual = actual_fg_made(
        row
    )

    if actual is None:
        continue

    state = row_to_state(row)

    if state is None:
        continue

    base = engine.normalize_state(
        state
    )

    probabilities = (
        engine
        .field_goal_probabilities(
            base
        )
    )

    fg_records.append({
        "actual":
            int(actual),

        "predicted":
            probabilities["made"],

        "distance":
            (
                base["yardline_100"]
                +
                18.0
            ),
    })


fg_eval = pd.DataFrame(
    fg_records
)


print_binary_metrics(
    "2023-2024 VALIDATION — FIELD GOAL",
    fg_eval["actual"],
    fg_eval["predicted"],
)


print_calibration_table(
    "2023-2024 VALIDATION — FIELD GOAL",
    fg_eval["actual"],
    fg_eval["predicted"],
)


print(
    "\nIMPORTANT:"
)

print(
    "2023-2024 is validation data and may be used "
    "to guide model-development decisions."
)

print(
    "2025 remains the held-out benchmark and must "
    "not be used for tuning."
)
