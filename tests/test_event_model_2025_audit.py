"""Observed fourth-down event audit must never imply counterfactual policy validity."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "src/69_audit_2025_event_models.py"
spec = importlib.util.spec_from_file_location("audit2025events", SCRIPT)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def state_row(**kwargs):
    row = {
        "season": 2025, "split": "test", "game_id": "2025_FOO_BAR",
        "action": "NORMAL_GO_PASS", "qtr": 4,
        "game_seconds_remaining": 299.,
        "yardline_100": 6., "ydstogo": 4.,
        "score_differential": -4., "is_home": 0,
        "is_neutral_site": 0,
        "posteam_timeouts_remaining": 3.,
        "defteam_timeouts_remaining": 3.,
        "goal_to_go": 0., "roof": "outdoors",
        "converted": 1., "field_goal_result": None,
    }
    row.update(kwargs)
    return pd.Series(row)


def test_rams_fourth_down_features_contain_only_preplay_data():
    original = state_row()
    result = audit.to_preplay_state(original)
    assert result["site"] == "AWAY"
    assert result["ydstogo"] == 4.
    assert result["yardline_100"] == 6.
    assert result["game_seconds_remaining"] == 299.
    assert "converted" not in result
    assert "field_goal_result" not in result

    changed = original.copy()
    changed["converted"] = 0
    changed["field_goal_result"] = "made"
    assert audit.to_preplay_state(original) == audit.to_preplay_state(changed)
    neutral = audit.to_preplay_state(
        state_row(is_home=0, is_neutral_site=1)
    )
    assert neutral["site"] == "NEUTRAL"


def test_labels_match_frozen_2025_benchmark_and_do_not_invent_unknowns():
    assert audit.fg_label(state_row(field_goal_result="made")) == 1
    assert audit.fg_label(state_row(field_goal_result="missed")) == 0
    assert audit.fg_label(state_row(field_goal_result="blocked")) == 0
    assert audit.fg_label(
        state_row(field_goal_result=None, execution_status="BROKEN")
    ) == 0
    assert audit.fg_label(
        state_row(field_goal_result=None)
    ) is None


def test_chronological_filters_fail_closed_on_leakage():
    raw = pd.DataFrame([
        state_row(season=2023, split="validation", game_id="g23"),
        state_row(season=2024, split="validation", game_id="g24"),
        state_row(season=2025, split="test", game_id="g25"),
        state_row(season=2022, split="train", game_id="g22"),
    ])
    out = audit.validate_historical_rows(raw)
    assert set(out["season"]) == {2023, 2024, 2025}
    assert not out["season"].eq(2022).any()
    with pytest.raises(RuntimeError, match="split mismatch"):
        audit.validate_historical_rows(raw.assign(split="test"))
    with pytest.raises(ValueError, match="Missing"):
        audit.validate_historical_rows(raw.drop(columns=["converted"]))


class Dummy:
    @staticmethod
    def normalize_state(state):
        assert "converted" not in state
        return state

    @staticmethod
    def go_conversion_probability(state, action):
        assert action in ("NORMAL_GO_RUN", "NORMAL_GO_PASS")
        return .4 if action == "NORMAL_GO_PASS" else .7

    @staticmethod
    def field_goal_probabilities(state):
        return {"made": .97}


def test_chosen_action_only_is_evaluated_and_unknown_labels_skipped():
    rows = pd.DataFrame([
        state_row(game_id="g1", action="NORMAL_GO_PASS", converted=1),
        state_row(game_id="g2", action="FIELD_GOAL",
                  field_goal_result="made", converted=np.nan),
        state_row(game_id="g3", action="NORMAL_GO_PASS", converted=np.nan),
    ])
    preds, excluded = audit.predict_observed_events(
        rows, {
            "normalize_state": Dummy.normalize_state,
            "go_conversion_probability": Dummy.go_conversion_probability,
            "field_goal_probabilities": Dummy.field_goal_probabilities,
        },
    )
    assert len(preds) == 2
    assert preds["predicted"].to_list() == [.4, .97]
    assert preds["actual"].to_list() == [1, 1]
    assert excluded["missing/invalid actual outcome"] == 1


def test_bootstrap_clusters_games_and_can_have_small_support():
    a = np.array([1., 0., 0., 1., 1.])
    p = np.array([.2, .6, .6, .4, .7])
    groups = np.array(["g1", "g1", "g1", "g2", "g2"])
    ci = audit._bootstrap_gap(a, p, groups, draws=300)
    assert ci == audit._bootstrap_gap(a, p, groups, draws=300)
    assert ci[0] < ci[1]
    table = pd.DataFrame({"game_id": groups, "actual": a, "predicted": p})
    stats = audit.summarize_subgroup("SMALL", table)
    assert stats["status"] == "LOW SUPPORT"
    assert stats["n"] == 5
    assert stats["gap_actual_minus_predicted"] == pytest.approx(
        np.mean(a-p)
    )


def test_prespecified_groups_focus_on_24yd_fg_vs_4th_and_4_pass():
    rows = pd.DataFrame([
        dict(game_id="pass", action="NORMAL_GO_PASS", qtr=4,
             seconds=299., ydstogo=4., yardline=6., fg_distance=24.,
             actual=0, predicted=.42),
        dict(game_id="fg", action="FIELD_GOAL", qtr=4,
             seconds=299., ydstogo=4., yardline=6., fg_distance=24.,
             actual=1, predicted=.97),
        dict(game_id="run", action="NORMAL_GO_RUN", qtr=3,
             seconds=400., ydstogo=2., yardline=54., fg_distance=72.,
             actual=1, predicted=.7),
    ])
    results = {x["group"]: x for x in audit.report_predeclared_groups(rows)}
    assert results["GO PASS 4TH&4 OPP 1-10"]["n"] == 1
    assert results["FG 18-29 Q4 LAST FIVE"]["n"] == 1
    assert results["GO RUN SHORT <=3"]["n"] == 1
    assert results["FG 50+ YARDS"]["n"] == 0


def test_no_future_2026_or_model_retraining_in_this_audit():
    source = SCRIPT.read_text()
    assert "SEASONS = (2023, 2024, 2025)" in source
    assert "nfl.load_pbp(list(SEASONS))" in source
    assert "nfl.load_pbp([2026])" not in source
    assert "model.fit(" not in source
    assert "joblib.dump(" not in source
