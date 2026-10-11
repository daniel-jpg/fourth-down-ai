"""Tests for frozen 2026 W6+ kickoff continuation evaluation."""

import hashlib
import importlib.util
from pathlib import Path
import runpy

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / "src/66_forward_2026_continuation_wp.py"
spec = importlib.util.spec_from_file_location("prospective_2026", path)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)

helper61 = runpy.run_path(str(ROOT / "src/61_audit_observed_kickoff_wp.py"))
helper65 = runpy.run_path(str(ROOT / "src/65_crossfit_2025_continuation_wp.py"))
FIELDS = helper61["FIELDS"]


def play(game, play_id, typ, week, time_left, *, own=None, kick_spot=35,
         receiver="SEA", score_for=20, score_against=17, touchback=0,
         result=7, desc="", season=2026):
    return dict(
        season=season, season_type="REG", week=week,
        game_id=game, play_id=play_id, play_type=typ,
        game_seconds_remaining=float(time_left),
        half_seconds_remaining=float(time_left), qtr=4,
        down=np.nan if own is None else 1.,
        ydstogo=0. if own is None else 10.,
        yardline_100=float(kick_spot if own is None else 100-own),
        goal_to_go=0., posteam_score=score_for,
        defteam_score=score_against,
        posteam_timeouts_remaining=2, defteam_timeouts_remaining=1,
        posteam=receiver, home_team="SEA", touchback=touchback,
        desc=desc, result=result,
    )


def sample():
    return pd.DataFrame([
        play("g5", 1, "kickoff", 5, 290, touchback=1),
        play("g5", 2, "run", 5, 290, own=35),
        play("g6", 1, "kickoff", 6, 299, touchback=1),
        play("g6", 2, "run", 6, 299, own=35),
        play("g7", 1, "kickoff", 7, 245, receiver="LAR"),
        play("g7", 2, "pass", 7, 239, receiver="LAR", own=28),
        play("g8", 1, "kickoff", 8, 200, desc="onside kick"),
        play("g8", 2, "run", 8, 194, own=40),
        play("g9", 1, "kickoff", 9, 180, touchback=1),
        play("g9", 2, "run", 9, 180, own=35, score_for=27),
        play("g10", 1, "kickoff", 10, 160, result=0),
        play("g10", 2, "run", 10, 154, own=28, result=0),
        play("g11", 1, "kickoff", 11, 180, kick_spot=50),
        play("g11", 2, "pass", 11, 174, own=30),
        play("g12", 1, "kickoff", 12, 35, touchback=1),
        play("g12", 2, "run", 12, 35, own=35),
    ])


class FakeWP:
    classes_ = np.array([0, 2])

    def predict_proba(self, matrix):
        p = np.clip(
            0.3
            + 0.0004*np.nan_to_num(
                matrix[:, FEATURES.index("yardline_100")], nan=20.)
            + 0.00008*matrix[:, FEATURES.index("game_seconds_remaining")]
            + 0.01*matrix[:, FEATURES.index("is_kickoff")],
            0.01, 0.99,
        )
        return np.column_stack([1-p, p])


FEATURES = [
    "game_seconds_remaining", "half_seconds_remaining", "qtr", "down",
    "ydstogo", "yardline_100", "goal_to_go", "home_score_differential",
    "home_timeouts_remaining", "away_timeouts_remaining", "is_home_posteam",
    "is_kickoff", "is_overtime", "final_five_minutes",
]


def donors():
    return pd.DataFrame({
        "own": [35., 28., 20.],
        "elapsed": [0., 6., 0.],
        "touchback": [True, False, True],
    })


def test_2026_week6_plus_pairs_and_exception_filters():
    result = audit.prospective_2026_pairs(sample(), FIELDS)
    assert result["game_id"].tolist() == ["g6", "g7"]
    assert result["week"].tolist() == [6, 7]
    assert result["play_id_snap"].tolist() == [2, 2]
    with pytest.raises(ValueError, match="2026 only"):
        audit.prospective_2026_pairs(sample().assign(season=2025), FIELDS)


def test_2026_without_future_weeks_returns_empty_not_w1_to_w5():
    pbp = sample().loc[lambda x: x["week"].eq(5)].reset_index(drop=True)
    result = audit.prospective_2026_pairs(pbp, FIELDS)
    assert result.empty


def test_forecast_never_uses_actual_field_position_or_final_result():
    pairs = audit.prospective_2026_pairs(sample(), FIELDS)
    first = audit.prospective_predictions(
        pairs, donors(), FakeWP(), FEATURES, helper61, helper65
    )
    changed = pairs.copy()
    changed["yardline_100_snap"] = 56.
    changed["result_kick"] = -7
    changed["result_snap"] = -7
    second = audit.prospective_predictions(
        changed, donors(), FakeWP(), FEATURES, helper61, helper65
    )
    np.testing.assert_allclose(first["continuation"], second["continuation"])
    np.testing.assert_allclose(first["engine_pending"], second["engine_pending"])
    np.testing.assert_allclose(first["native_pending"], second["native_pending"])
    assert not np.array_equal(
        first["oracle_first_snap"], second["oracle_first_snap"]
    )
    assert not np.array_equal(
        first["actual_home_win"], second["actual_home_win"]
    )


def test_frozen_2025_donors_enforce_year_and_ordinary_outcomes():
    rows = pd.DataFrame([
        dict(
            season=2025, week=(j % 18)+1,
            touchback=1 if j % 2 == 0 else 0,
            yardline_100_snap=65. if j % 2 == 0 else 71.,
            game_seconds_remaining_kick=250.,
            game_seconds_remaining_snap=250. if j % 2 == 0 else 244.,
        )
        for j in range(60)
    ])
    stub64 = {"ordinary_2025_q4_pairs": lambda pbp: rows}
    stub65 = {"with_kickoff_weeks": lambda paired, pbp: paired}
    got = audit.frozen_2025_donors(
        pd.DataFrame({"season": [2025]}), stub64, stub65
    )
    assert len(got) == 60
    assert got["touchback"].sum() == 30
    with pytest.raises(ValueError, match="2025"):
        audit.frozen_2025_donors(
            pd.DataFrame({"season": [2026]}), stub64, stub65
        )


def test_pinned_git_blob_detection(tmp_path, monkeypatch):
    root = tmp_path
    (root / "models").mkdir()
    content = b"frozen model"
    file = root / "models" / "wp.bin"
    file.write_bytes(content)
    sha = hashlib.sha1(f"blob {len(content)}\0".encode()+content).hexdigest()
    monkeypatch.setattr(audit, "PINNED_GIT_BLOBS", {"models/wp.bin": sha})
    assert audit.git_blob_sha(file) == sha
    audit.enforce_frozen_artifacts(root)
    file.write_bytes(b"different")
    with pytest.raises(RuntimeError, match="Frozen WP artifact changed"):
        audit.enforce_frozen_artifacts(root)


def test_predeclared_gate_does_not_score_small_samples():
    scores = pd.DataFrame({
        "game_id": ["g6", "g7"],
    })
    assert "NOT READY" in audit.frozen_gate(scores, {})
