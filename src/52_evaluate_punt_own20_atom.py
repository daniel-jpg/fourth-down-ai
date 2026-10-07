import importlib.util

import numpy as np
import pandas as pd


# =========================================================
# Validation-only candidate punt sampler audit.
#
# Candidate:
#
# If a sampled training donor's actual resulting field
# position was the receiving team's own 20 (yardline_100=80),
# preserve that discrete state exactly.
#
# Otherwise retain the current production rule:
#
#     current prediction + donor OOF residual
#
# 2025 is not used.
# =========================================================


SIMULATIONS_PER_ROW = 1000
RANDOM_SEED = 520052

OWN20_YARDLINE = 80.0
OWN20_TOLERANCE = 0.5


evaluation = pd.read_parquet(
    "data/"
    "evaluation_validation_punt_distribution.parquet"
)


spec = importlib.util.spec_from_file_location(
    "decision_engine",
    "src/42_build_decision_engine.py",
)

engine = importlib.util.module_from_spec(
    spec
)

spec.loader.exec_module(
    engine
)


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

        nearest = available_buckets[
            np.argmin(
                np.abs(
                    available_buckets
                    -
                    start_bucket
                )
            )
        ]

        candidate = pool[
            pool[
                "start_yardline_bucket"
            ]
            ==
            nearest
        ]


    return candidate


def clip_draws(
    values,
):

    return np.fromiter(
        (
            engine.clip_yardline(
                value
            )
            for value in values
        ),
        dtype=float,
        count=len(values),
    )


rng = np.random.default_rng(
    RANDOM_SEED
)


baseline_all = []
candidate_all = []

records = []


for i, row in evaluation.iterrows():

    donors = donor_candidates(
        row[
            "start_yardline_100"
        ]
    )


    donor_indices = rng.integers(
        0,
        len(donors),
        size=
            SIMULATIONS_PER_ROW,
    )


    sampled = (
        donors.iloc[
            donor_indices
        ]
        .reset_index(
            drop=True
        )
    )


    residual = (
        sampled[
            "yardline_residual"
        ]
        .to_numpy(
            dtype=float
        )
    )


    donor_final = (
        sampled[
            "state_yardline_100"
        ]
        .to_numpy(
            dtype=float
        )
    )


    prediction = float(
        row[
            "point_prediction"
        ]
    )


    baseline_draws = clip_draws(
        prediction
        +
        residual
    )


    donor_own20 = np.isclose(
        donor_final,
        OWN20_YARDLINE,
        atol=
            OWN20_TOLERANCE,
    )


    candidate_draws = (
        baseline_draws.copy()
    )


    candidate_draws[
        donor_own20
    ] = OWN20_YARDLINE


    baseline_all.append(
        baseline_draws
    )

    candidate_all.append(
        candidate_draws
    )


    observed = float(
        row[
            "observed_yardline_100"
        ]
    )


    baseline_p10 = float(
        np.quantile(
            baseline_draws,
            0.10,
        )
    )

    baseline_p90 = float(
        np.quantile(
            baseline_draws,
            0.90,
        )
    )


    candidate_p10 = float(
        np.quantile(
            candidate_draws,
            0.10,
        )
    )

    candidate_p90 = float(
        np.quantile(
            candidate_draws,
            0.90,
        )
    )


    records.append({

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
            float(
                row[
                    "start_yardline_100"
                ]
            ),

        "field_bucket":
            row[
                "field_bucket"
            ],

        "observed_yardline_100":
            observed,

        "baseline_mean":
            float(
                np.mean(
                    baseline_draws
                )
            ),

        "candidate_mean":
            float(
                np.mean(
                    candidate_draws
                )
            ),

        "baseline_10_90_covered":
            bool(
                observed
                >=
                baseline_p10
                and
                observed
                <=
                baseline_p90
            ),

        "candidate_10_90_covered":
            bool(
                observed
                >=
                candidate_p10
                and
                observed
                <=
                candidate_p90
            ),

        "sampled_own20_donor_rate":
            float(
                np.mean(
                    donor_own20
                )
            ),
    })


results = pd.DataFrame(
    records
)


baseline_all = np.concatenate(
    baseline_all
)

candidate_all = np.concatenate(
    candidate_all
)

observed_all = (
    evaluation[
        "observed_yardline_100"
    ]
    .to_numpy(
        dtype=float
    )
)


def stats(
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

        "own20":
            float(
                np.mean(
                    np.isclose(
                        values,
                        OWN20_YARDLINE,
                        atol=
                            OWN20_TOLERANCE,
                    )
                )
            ),

        "inside20":
            float(
                np.mean(
                    values > 80.0
                )
            ),

        "inside10":
            float(
                np.mean(
                    values > 90.0
                )
            ),

        "inside5":
            float(
                np.mean(
                    values > 95.0
                )
            ),
    }


observed_stats = stats(
    observed_all
)

baseline_stats = stats(
    baseline_all
)

candidate_stats = stats(
    candidate_all
)


print(
    "\nPUNT OWN-20 ATOM CANDIDATE"
)

print(
    f"Validation rows: "
    f"{len(evaluation):,}"
)

print(
    f"Simulations per row: "
    f"{SIMULATIONS_PER_ROW:,}"
)


print(
    "\nOVERALL DISTRIBUTION"
)

print(
    f"{'Metric':<22}"
    f"{'Observed':>12}"
    f"{'Current':>12}"
    f"{'Candidate':>12}"
)


for key, label in [
    (
        "mean",
        "Mean yardline",
    ),
    (
        "median",
        "Median",
    ),
    (
        "p10",
        "P10",
    ),
    (
        "p90",
        "P90",
    ),
]:

    print(
        f"{label:<22}"
        f"{observed_stats[key]:>12.3f}"
        f"{baseline_stats[key]:>12.3f}"
        f"{candidate_stats[key]:>12.3f}"
    )


for key, label in [
    (
        "own20",
        "Own-20 atom",
    ),
    (
        "inside20",
        "Inside own 20",
    ),
    (
        "inside10",
        "Inside own 10",
    ),
    (
        "inside5",
        "Inside own 5",
    ),
]:

    print(
        f"{label:<22}"
        f"{100 * observed_stats[key]:>11.2f}%"
        f"{100 * baseline_stats[key]:>11.2f}%"
        f"{100 * candidate_stats[key]:>11.2f}%"
    )


print(
    "\n80% INTERVAL COVERAGE"
)

print(
    "Current:   "
    f"{100 * results['baseline_10_90_covered'].mean():.2f}%"
)

print(
    "Candidate: "
    f"{100 * results['candidate_10_90_covered'].mean():.2f}%"
)


# =========================================================
# Starting-field-position comparison.
# =========================================================

evaluation = evaluation.copy()

evaluation[
    "start_bucket"
] = (
    np.floor(
        evaluation[
            "start_yardline_100"
        ]
        /
        10.0
    )
    *
    10
).astype(int)


results[
    "start_bucket"
] = (
    np.floor(
        results[
            "start_yardline_100"
        ]
        /
        10.0
    )
    *
    10
).astype(int)


bucket_rows = []


for bucket in sorted(
    evaluation[
        "start_bucket"
    ].unique()
):

    obs = evaluation[
        evaluation[
            "start_bucket"
        ]
        ==
        bucket
    ]


    result_rows = results[
        results[
            "start_bucket"
        ]
        ==
        bucket
    ]


    # Reconstruct draw subsets using the same row order.
    row_mask = (
        evaluation[
            "start_bucket"
        ]
        .to_numpy()
        ==
        bucket
    )


    baseline_bucket = np.concatenate(
        [
            draws
            for draws, keep
            in zip(
                np.array_split(
                    baseline_all,
                    len(evaluation),
                ),
                row_mask,
            )
            if keep
        ]
    )


    candidate_bucket = np.concatenate(
        [
            draws
            for draws, keep
            in zip(
                np.array_split(
                    candidate_all,
                    len(evaluation),
                ),
                row_mask,
            )
            if keep
        ]
    )


    observed_bucket = (
        obs[
            "observed_yardline_100"
        ]
        .to_numpy(
            dtype=float
        )
    )


    bucket_rows.append({

        "start_bucket":
            bucket,

        "n":
            len(
                observed_bucket
            ),

        "obs_own20":
            stats(
                observed_bucket
            )[
                "own20"
            ],

        "current_own20":
            stats(
                baseline_bucket
            )[
                "own20"
            ],

        "candidate_own20":
            stats(
                candidate_bucket
            )[
                "own20"
            ],

        "obs_inside20":
            stats(
                observed_bucket
            )[
                "inside20"
            ],

        "current_inside20":
            stats(
                baseline_bucket
            )[
                "inside20"
            ],

        "candidate_inside20":
            stats(
                candidate_bucket
            )[
                "inside20"
            ],

        "sampled_donor_own20":
            float(
                result_rows[
                    "sampled_own20_donor_rate"
                ]
                .mean()
            ),
    })


bucket_table = pd.DataFrame(
    bucket_rows
)


for column in [
    "obs_own20",
    "current_own20",
    "candidate_own20",
    "obs_inside20",
    "current_inside20",
    "candidate_inside20",
    "sampled_donor_own20",
]:

    bucket_table[
        column
    ] *= 100.0


print(
    "\nBY PRODUCTION START BUCKET"
)

print(
    bucket_table.to_string(
        index=False,
        formatters={

            "obs_own20":
                "{:.2f}%".format,

            "current_own20":
                "{:.2f}%".format,

            "candidate_own20":
                "{:.2f}%".format,

            "obs_inside20":
                "{:.2f}%".format,

            "current_inside20":
                "{:.2f}%".format,

            "candidate_inside20":
                "{:.2f}%".format,

            "sampled_donor_own20":
                "{:.2f}%".format,
        },
    )
)


print(
    "\nIMPORTANT"
)

print(
    "This candidate uses only the frozen 2014-2022 "
    "training donor pool and 2023-2024 validation states."
)

print(
    "2025 remains held out."
)

print(
    "No production behavior is changed by this script."
)


output_path = (
    "data/"
    "evaluation_validation_punt_own20_atom.parquet"
)


results.to_parquet(
    output_path,
    index=False,
)


print(
    f"\nSaved: {output_path}"
)
