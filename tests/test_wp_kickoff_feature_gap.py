from pathlib import Path
import runpy

import numpy as np
import pandas as pd
import pytest

script = Path(__file__).resolve().parents[1] / "src/62_audit_kickoff_feature_gap.py"
audit = runpy.run_path(str(script))
variants = audit["kickoff_variants"]
predict = audit["predicted_kickoffs"]
contrib = audit["feature_contributions"]


def native():
    return pd.DataFrame({
        "down": [1.0, np.nan, 1.0],
        "ydstogo": [10.0, np.nan, 8.0],
        "yardline_100": [65.0, 60.0, np.nan],
        "goal_to_go": [0.0, 1.0, np.nan],
        "is_home_posteam": [0.0, 1.0, 0.0],
    })


class StubWP:
    classes_ = np.array([0, 2])

    def predict_proba(self, X):
        down = np.isfinite(X[:, 0]).astype(float)
        distance = np.isfinite(X[:, 1]).astype(float)
        yard = np.where(np.isfinite(X[:, 2]), X[:, 2], 0.)
        g2g = np.nan_to_num(X[:, 3], nan=0.)
        # Intentional interaction: one-field effects need not be additive.
        p = .25 + .08*down + .06*distance + .001*yard + .04*g2g + .03*down*distance
        return np.column_stack([1-p, p])


FIELDS = ["down", "ydstogo", "yardline_100", "goal_to_go"]


def test_engine_variant_replaces_only_four_kickoff_fields():
    source = native()
    frames = variants(source)
    engine = frames["engine"]
    assert engine["down"].isna().all()
    assert engine["ydstogo"].isna().all()
    assert engine["yardline_100"].isna().all()
    assert engine["goal_to_go"].eq(0.).all()
    pd.testing.assert_series_equal(engine["is_home_posteam"], source["is_home_posteam"])
    pd.testing.assert_frame_equal(frames["native"], source)


def test_only_and_restore_variants_are_isolated():
    source = native()
    frames = variants(source)
    for f in FIELDS:
        only = frames[f"only_{f}"]
        restore = frames[f"restore_{f}"]
        for other in FIELDS:
            pd.testing.assert_series_equal(
                only[other],
                frames["engine"][other] if other == f else source[other],
            )
            pd.testing.assert_series_equal(
                restore[other],
                source[other] if other == f else frames["engine"][other],
            )
    pd.testing.assert_frame_equal(frames["step_goal_to_go"], frames["engine"])


def test_sequential_attribution_sums_exact_change_even_with_interactions():
    frames = variants(native())
    p = predict(StubWP(), FIELDS, frames)
    for mask in (np.ones(3, bool), np.array([True, False, True])):
        effects = contrib(p, mask)
        overall = 100*np.mean((p["engine"]-p["native"])[mask])
        assert sum(effects["sequential_change_pp"]) == pytest.approx(overall)
        assert effects["feature"].tolist() == FIELDS


def test_rejects_bad_mask_or_missing_features():
    p = predict(StubWP(), FIELDS, variants(native()))
    with pytest.raises(ValueError, match="nonempty mask"):
        contrib(p, np.zeros(3, dtype=bool))
    with pytest.raises(ValueError, match="Missing native"):
        variants(native().drop(columns="goal_to_go"))
    with pytest.raises(ValueError, match="Missing WP"):
        predict(StubWP(), FIELDS + ["absent"], variants(native()))
