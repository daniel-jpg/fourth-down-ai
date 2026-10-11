"""Frozen-model-compatible constrained win probability development candidate.

The production WP model uses 14 ordered input features and 3 output classes:
away win=0, tie=1, home win=2.

For regulation scrimmage states, the candidate learns the conditional home-win
probability from a binary HistGradientBoostingClassifier constrained by:
  * home score margin must never make home win LESS likely;
  * forward progress for the current offense must never penalize that offense;
  * home timeouts must not hurt home; away timeouts must not help home.

Field-position sign is transformed into HOME-team field advantage, so one
positive monotonic constraint is correct regardless of who possesses the ball.

Tie odds come from the frozen, three-class model to retain the game's tie
outcome in the production API; that baseline tie model has NOT been retrained.
Unresolved kickoffs and overtime must use the original frozen model, because
the constrained candidate only trains on REGULATION non-kickoff states.
Never auto-promote a candidate based on constraints alone: compare Brier,
calibration, directionality, and decision outputs out of time.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier


REQUIRED_FEATURES = (
    "qtr", "yardline_100", "is_home_posteam", "is_kickoff",
    "home_score_differential", "home_timeouts_remaining",
    "away_timeouts_remaining",
)


def monotonic_constraints(features: list[str]) -> list[int]:
    """Validate the exact feature contract and specify football invariants."""
    if len(features) != len(set(features)):
        raise ValueError("Duplicate win-probability features.")
    missing = set(REQUIRED_FEATURES).difference(features)
    if missing:
        raise ValueError(f"Missing required WP features: {sorted(missing)}")
    constrained = {
        "yardline_100": 1,  # REPLACED by home-field-advantage at prediction
        "home_score_differential": 1,
        "home_timeouts_remaining": 1,
        "away_timeouts_remaining": -1,
    }
    return [constrained.get(name, 0) for name in features]


def home_advantage_matrix(X, features: list[str]) -> np.ndarray:
    """Preserve model inputs except field position in HOME-team perspective.

    Home possesses: reaching opponent end zone means (100-yardline_100) rises.
    Away possesses: moving away from home end zone means yardline_100 rises.
    Neither transformation uses labels or the realized outcome.
    """
    arr = np.asarray(X, dtype=float)
    monotonic_constraints(features)
    if arr.ndim != 2 or arr.shape[1] != len(features):
        raise ValueError("WP feature matrix must match the ordered feature list.")
    yi = features.index("yardline_100")
    pi = features.index("is_home_posteam")
    own = arr[:, yi]
    pos = arr[:, pi]
    if not (np.isfinite(pos).all() and np.isin(pos, (0.0, 1.0)).all()):
        raise ValueError("Possession axis must be defined for each state.")
    out = arr.copy()
    out[:, yi] = np.where(pos == 1.0, 100.0-own, own)
    return out


def new_binary_model(features: list[str]) -> HistGradientBoostingClassifier:
    """FIXED model specification: no tuning against 2025+ outcomes."""
    return HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=220,
        max_leaf_nodes=31,
        min_samples_leaf=100,
        l2_regularization=2.0,
        early_stopping=False,
        monotonic_cst=monotonic_constraints(features),
        random_state=42,
    )


def fit_binary_wp(X, final_home_margin, features: list[str], game_id):
    """Fit on complete, non-tied games, weighting games rather than plays.

    Ties are excluded from binary training, and handled separately by the
    original frozen multi-class model during inference.
    """
    matrix = home_advantage_matrix(X, features)
    result = np.asarray(final_home_margin, dtype=float)
    groups = np.asarray(game_id)
    if result.shape != (len(matrix),) or groups.shape != (len(matrix),):
        raise ValueError("Outcome and game IDs must match feature rows.")
    if not np.isfinite(result).all() or len(matrix) == 0:
        raise ValueError("Need nonempty completed games with finite outcomes.")

    valid = (result != 0)
    if not valid.any():
        raise ValueError("No non-tied training games.")
    labels = (result[valid] > 0).astype(int)
    if len(np.unique(labels)) != 2:
        raise ValueError("Training set requires home and away winners.")

    ids = pd_unique_inverse(groups[valid])
    counts = np.bincount(ids)
    weights = 1.0 / counts[ids]
    weights *= len(weights) / weights.sum()

    model = new_binary_model(features)
    model.fit(matrix[valid], labels, sample_weight=weights)
    return model


def pd_unique_inverse(items: np.ndarray) -> np.ndarray:
    """Game-ID grouping without importing pandas in the inference module."""
    _, inverse = np.unique(items.astype(str), return_inverse=True)
    return inverse


class ConstrainedRegulationWP:
    """3-class classifier-compatible, pickleable candidate for development."""

    def __init__(self, binary_model, frozen_model, features: list[str]):
        self.binary_model = binary_model
        self.frozen_model = frozen_model
        self.features = tuple(features)
        monotonic_constraints(features)
        self.classes_ = np.array([0, 1, 2], dtype=int)

    def predict_proba(self, X) -> np.ndarray:
        arr = np.asarray(X, dtype=float)
        transformed = home_advantage_matrix(arr, list(self.features))
        frozen = np.asarray(self.frozen_model.predict_proba(arr), dtype=float)
        frozen_classes = np.asarray(self.frozen_model.classes_, dtype=int)
        frozen_ix = {int(c): i for i, c in enumerate(frozen_classes)}
        if 0 not in frozen_ix or 2 not in frozen_ix:
            raise ValueError("Frozen WP model lacks home and away win classes.")
        original = np.column_stack([
            frozen[:, frozen_ix[0]],
            frozen[:, frozen_ix[1]] if 1 in frozen_ix else np.zeros(len(arr)),
            frozen[:, frozen_ix[2]],
        ])

        # No extrapolation to kickoff-pending or overtime states.
        q = arr[:, self.features.index("qtr")]
        kick = arr[:, self.features.index("is_kickoff")]
        scrimmage = (
            np.isfinite(q) & (q >= 1) & (q <= 4)
            & np.isfinite(kick) & (kick == 0)
        )
        if not scrimmage.any():
            return original

        pred = self.binary_model.predict_proba(transformed[scrimmage])
        binary_class = {int(v): j for j, v in enumerate(self.binary_model.classes_)}
        if 1 not in binary_class:
            raise ValueError("Binary WP model lacks home-win class.")
        p_home_conditional = np.clip(pred[:, binary_class[1]], 0, 1)
        p_tie = np.clip(original[scrimmage, 1], 0, 1)
        p_home = (1-p_tie)*p_home_conditional
        p_away = (1-p_tie)*(1-p_home_conditional)

        result = original.copy()
        result[scrimmage, :] = np.column_stack([p_away, p_tie, p_home])
        if (not np.isfinite(result).all()
                or not np.allclose(result.sum(axis=1), 1.0, atol=1e-6)
                or ((result < -1e-9) | (result > 1+1e-9)).any()):
            raise ValueError("Constrained candidate returned invalid WP probabilities.")
        return result
