"""Exercise the deployed app through Streamlit's headless AppTest API.

Fixed engine responses test UI wiring without Monte Carlo assertions. One
smoke test also loads the real frozen engine. No future evaluation is run.
AppTest does not execute the custom football field's browser JavaScript.
"""
from copy import deepcopy
import importlib.abc
import importlib.util
from pathlib import Path
import socket
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from src import kickoff_q4_reliability_guard as guard
from src import streamlit_kickoff_preview as preview


ROOT = Path(__file__).resolve().parents[1]
EXPLAINER = "Why this recommendation? Immediate action odds"
NORMAL = {
    "NORMAL_GO_PASS": ("GO — PASS / DROPBACK", .64, "conversion=0.438"),
    "NORMAL_GO_RUN": ("GO — RUN", .50, "conversion=0.486"),
    "FIELD_GOAL": ("FIELD GOAL", .40, "make=0.730, miss=0.263, block=0.007"),
    "PUNT": ("PUNT", .20, "regular=0.982, blocked=0.0050, broken=0.0130"),
}


class FixedEngine:
    """Only the recommend boundary is replaced; the app itself runs as-is."""

    def __init__(self):
        self.unavailable = set()
        self.close = False
        self.bad_detail = False
        self.wp_overrides = {}
        self.calls = []
        self.outputs = []

    def result(self, *, experimental=False):
        rows = []
        for action, (label, wp, detail) in NORMAL.items():
            wp = self.wp_overrides.get(action, wp)
            eligible = action not in self.unavailable
            if self.close and action == "NORMAL_GO_RUN":
                wp = .639
            if experimental and action == "FIELD_GOAL":
                wp = .88
            if self.bad_detail and action == "NORMAL_GO_PASS":
                detail = "conversion=nan"
            rows.append({
                "action": action,
                "decision": label,
                "eligible": eligible,
                "expected_win_probability": wp if eligible else np.nan,
                "mc_se": .004 if eligible else np.nan,
                "detail": detail if eligible else "Outside frozen model support",
            })
        rows.append({
            "action": "FAKE_PUNT_RUN", "decision": "FAKE PUNT — RUN",
            "eligible": False, "expected_win_probability": .99,
            "mc_se": .001, "detail": "experimental only",
        })
        return {"pre_decision_wp": .33, "results": pd.DataFrame(rows)}

    def recommend(self, state, *, n_simulations, seed):
        self.calls.append((deepcopy(state), n_simulations, seed))
        result = self.result()
        # Retain the returned object to detect accidental mutation of its rows.
        self.outputs.append(result)
        return result


def widget(at, kind, label):
    matches = [item for item in getattr(at, kind) if item.label == label]
    assert len(matches) == 1, (kind, label, matches)
    return matches[0]


def run(at):
    at.run()
    assert not at.exception, [item.message for item in at.exception]
    return at


def analyze(at):
    widget(at, "button", "Analyze Fourth Down").click()
    return run(at)


def metrics(at):
    return {item.label: item.value for item in at.metric}


def values(at, kind):
    return "\n".join(item.value for item in getattr(at, kind))


@pytest.fixture(autouse=True)
def offline_ui(monkeypatch):
    """Fail closed if any UI case attempts a live network connection."""
    connection = Mock(side_effect=AssertionError("UI must use frozen local artifacts"))
    monkeypatch.setattr(socket.socket, "connect", connection)
    yield
    connection.assert_not_called()


@pytest.fixture
def ui(monkeypatch):
    monkeypatch.chdir(ROOT)
    st.cache_resource.clear()
    engine = FixedEngine()
    original_spec = importlib.util.spec_from_file_location

    class Loader(importlib.abc.Loader):
        def create_module(self, spec):
            return None

        def exec_module(self, module):
            module.recommend = engine.recommend

    def engine_spec(name, location, *args, **kwargs):
        if name == "fourth_down_engine":
            assert Path(location).name == "42_build_decision_engine.py"
            return importlib.util.spec_from_loader(name, Loader())
        return original_spec(name, location, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "spec_from_file_location", engine_spec)
    probe = Mock(return_value={"applicable": True, "material": False})
    monkeypatch.setattr(guard, "inspect_made_fg_kickoff_encoding", probe)
    previews = {}
    for name in (
        "preview_recommendation", "preview_observed_2025_recommendation",
        "preview_observed_2025_resolved_recommendation",
        "preview_native_kickoff_feature_candidate", "preview_observed_2025_with_onside",
    ):
        fn = Mock(side_effect=lambda *args, **kwargs: engine.result(experimental=True))
        monkeypatch.setattr(preview, name, fn)
        previews[name] = fn
    at = run(AppTest.from_file(str(ROOT / "app.py"), default_timeout=15))
    yield SimpleNamespace(at=at, engine=engine, probe=probe, previews=previews)
    # Do not leak the fake cached module to real-engine tests or another session.
    st.cache_resource.clear()


def test_initial_page_is_idle_and_has_no_kickoff_research_panel(ui):
    at = ui.at
    assert not any("kickoff comparison" in e.label.lower() for e in at.expander)
    assert not any("experimental" in c.label.lower() for c in at.checkbox)
    assert not widget(at, "button", "Analyze Fourth Down").disabled
    assert not at.metric
    assert not ui.engine.calls
    assert not ui.probe.called
    assert not any(fn.called for fn in ui.previews.values())


@pytest.mark.parametrize("quarter,seconds", [(1, 3078.), (3, 1278.), (4, 378.)])
def test_regulation_inputs_reach_engine_and_render_probabilities(ui, quarter, seconds):
    at = ui.at
    widget(at, "radio", "Quarter").set_value(quarter)
    widget(at, "text_input", "Time remaining (MM:SS)").set_value("06:18")
    widget(at, "number_input", "Yards to go").set_value(2)
    widget(at, "number_input", "Offense score").set_value(10)
    widget(at, "number_input", "Defense score").set_value(14)
    widget(at, "radio", "Offense timeouts").set_value(1)
    widget(at, "radio", "Defense timeouts").set_value(2)
    widget(at, "radio", "Game site").set_value("Neutral")
    widget(at, "selectbox", "Stadium / roof").set_value("closed")
    run(at)
    analyze(at)
    assert ui.engine.calls == [({
        "qtr": quarter, "game_seconds_remaining": seconds,
        "yardline_100": 30, "ydstogo": 2., "score_differential": -4.,
        "posteam_timeouts_remaining": 1., "defteam_timeouts_remaining": 2.,
        "site": "NEUTRAL", "roof": "closed",
    }, 300, 42)]
    assert metrics(at) == {
        "Recommendation": "GO FOR IT", "Win probability if chosen": "64.00%",
        "Advantage over FIELD GOAL": "+24.00 pp", "PASS / DROPBACK": "64.00%",
        "RUN": "50.00%", "FIELD GOAL": "40.00%", "PUNT": "20.00%",
    }
    assert "before the fourth-down decision: 33.00%" in values(at, "caption")


def test_immediate_odds_and_detailed_output_match_engine_response(ui):
    analyze(ui.at)
    expander = widget(ui.at, "expander", EXPLAINER)
    assert expander.proto.expanded is False
    odds = expander.dataframe[0].value
    assert odds["Play"].tolist() == [entry[0] for entry in NORMAL.values()]
    assert odds["Estimated game WP"].tolist() == ["64.00%", "50.00%", "40.00%", "20.00%"]
    assert odds["Immediate modeled outcome"].tolist() == [
        "Convert: 43.8%", "Convert: 48.6%",
        "Make: 73.0%  |  Miss: 26.3%  |  Block: 0.7%",
        "Regular: 98.2%  |  Blocked: 0.5%  |  Broken play: 1.3%",
    ]
    assert odds["MC SE (pp)"].tolist() == ["0.40"] * 4
    assert "NOT the same" in values(expander, "caption")
    assert "median kicker context" in values(expander, "caption")
    detail = widget(ui.at, "expander", "Show detailed model output").dataframe[0].value
    assert detail["Expected WP"].tolist() == [64., 50., 40., 20.]
    assert detail["Model Detail"].tolist() == ui.engine.outputs[0]["results"].iloc[:4]["detail"].tolist()
    assert len(ui.engine.calls) == 1  # Explanation must not trigger extra simulations.


@pytest.mark.parametrize("unavailable", [
    {"NORMAL_GO_RUN"}, {"FIELD_GOAL"}, {"PUNT"},
    {"NORMAL_GO_RUN", "FIELD_GOAL", "PUNT"},
])
def test_unavailable_actions_have_no_invented_probabilities(ui, unavailable):
    ui.engine.unavailable = unavailable
    analyze(ui.at)
    labels = {"NORMAL_GO_RUN": "RUN", "FIELD_GOAL": "FIELD GOAL", "PUNT": "PUNT"}
    for action in unavailable:
        assert metrics(ui.at)[labels[action]] == "Unavailable"
    assert "Some normal actions are unavailable" in values(ui.at, "warning")
    assert "Outside frozen model support" in values(ui.at, "markdown")
    odds = widget(ui.at, "expander", EXPLAINER).dataframe[0].value
    assert len(odds) == 4 - len(unavailable)
    for action in unavailable:
        assert NORMAL[action][0] not in odds["Play"].tolist()
    assert not any("FAKE" in label for label in odds["Play"])
    assert metrics(ui.at)["Recommendation"] == "GO FOR IT"
    if "FIELD_GOAL" in unavailable:
        ui.probe.assert_not_called()


def test_invalid_explanation_fails_closed_but_official_result_remains(ui):
    ui.engine.bad_detail = True
    analyze(ui.at)
    expander = widget(ui.at, "expander", EXPLAINER)
    assert not expander.dataframe
    assert "could not be verified" in values(expander, "caption")
    assert metrics(ui.at)["Win probability if chosen"] == "64.00%"
    assert len(ui.engine.calls) == 1


def test_close_action_fixture_exercises_adaptive_budget_and_playcall_message(ui):
    ui.engine.close = True
    analyze(ui.at)
    assert [(n, seed) for _, n, seed in ui.engine.calls] == [(300, 42), (1200, 43)]
    assert "Slight lean: PASS / DROPBACK" in values(ui.at, "info")
    assert metrics(ui.at)["Recommendation"] == "GO FOR IT"


@pytest.mark.parametrize("clock", ["oops", "07:60", "16:00", "-1:00"])
def test_invalid_clocks_block_analysis(ui, clock):
    widget(ui.at, "text_input", "Time remaining (MM:SS)").set_value(clock)
    run(ui.at)
    assert widget(ui.at, "button", "Analyze Fourth Down").disabled
    assert "Enter the clock as MM:SS" in values(ui.at, "error")
    assert not ui.engine.calls


def test_impossible_yards_to_go_block_analysis(ui):
    widget(ui.at, "number_input", "Yards to go").set_value(31)
    run(ui.at)
    assert widget(ui.at, "button", "Analyze Fourth Down").disabled
    assert "cannot exceed the distance" in values(ui.at, "error")
    assert not ui.engine.calls


def test_field_component_initial_state_reaches_engine(ui):
    # Seed before the component registers. AppTest cannot send browser JS events.
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=15)
    at.session_state["football_field"] = {"yardline_100": 6}
    run(at)
    widget(at, "number_input", "Yards to go").set_value(4)
    run(at)
    analyze(at)
    assert ui.engine.calls[0][0]["yardline_100"] == 6
    assert "opponent 6" in values(at, "markdown")


@pytest.mark.parametrize("quarter", [1, 2, 3, 4])
def test_regulation_clock_cannot_exceed_fifteen_minutes(ui, quarter):
    widget(ui.at, "radio", "Quarter").set_value(quarter)
    widget(ui.at, "text_input", "Time remaining (MM:SS)").set_value("15:01")
    run(ui.at)
    assert widget(ui.at, "button", "Analyze Fourth Down").disabled
    assert "Regulation quarter clock cannot exceed 15:00" in values(ui.at, "error")
    widget(ui.at, "text_input", "Time remaining (MM:SS)").set_value("15:00")
    run(ui.at)
    if quarter == 2:
        widget(ui.at, "radio", "Second-half kickoff receiver").set_value("Current offense")
        run(ui.at)
    assert not widget(ui.at, "button", "Analyze Fourth Down").disabled
    assert not ui.engine.calls


@pytest.mark.parametrize("receiver,expected", [("Current offense", "OFFENSE"), ("Current defense", "DEFENSE")])
def test_second_quarter_requires_known_halftime_receiver(ui, receiver, expected):
    widget(ui.at, "radio", "Quarter").set_value(2)
    run(ui.at)
    assert widget(ui.at, "button", "Analyze Fourth Down").disabled
    widget(ui.at, "radio", "Second-half kickoff receiver").set_value(receiver)
    run(ui.at)
    analyze(ui.at)
    assert ui.engine.calls[0][0]["second_half_receiver"] == expected
    assert ui.engine.calls[0][0]["game_seconds_remaining"] == 2258.
    ui.probe.assert_not_called()


def test_q4_material_warning_is_automatic_and_does_not_change_official_results(ui):
    ui.probe.return_value = {
        "applicable": True, "material": True,
        "legacy_conditional_made_fg_wp": .76, "native_conditional_made_fg_wp": .50,
        "absolute_difference_pp": 26., "assumed_fg_execution_seconds": 5.,
    }
    analyze(ui.at)
    assert "Model reliability warning" in values(ui.at, "warning")
    assert "conditional post-FG win probabilities" in values(ui.at, "warning")
    assert "frozen-model outputs are unchanged" in values(ui.at, "warning")
    assert metrics(ui.at)["Win probability if chosen"] == "64.00%"
    assert not any("experimental" in c.label.lower() for c in ui.at.checkbox)
    ui.probe.assert_called_once_with(ui.engine.calls[0][0])
    assert not any(fn.called for fn in ui.previews.values())


@pytest.mark.parametrize("quarter,clock", [(3, "07:38"), (4, "00:45"), (4, "00:00")])
def test_warning_probe_skips_out_of_scope_periods_and_clocks(ui, quarter, clock):
    widget(ui.at, "radio", "Quarter").set_value(quarter)
    widget(ui.at, "text_input", "Time remaining (MM:SS)").set_value(clock)
    run(ui.at)
    analyze(ui.at)
    ui.probe.assert_not_called()
    assert "Model reliability warning" not in values(ui.at, "warning")


def test_warning_probe_failure_has_visible_fallback_and_preserves_result(ui):
    ui.probe.side_effect = RuntimeError("unavailable fixture")
    analyze(ui.at)
    assert "reliability check is unavailable" in values(ui.at, "caption")
    assert metrics(ui.at)["FIELD GOAL"] == "40.00%"


def test_q4_459_trailing_four_fg_sensitivity_is_not_a_confident_pick(ui):
    # Real historical reproducibility state: 4th-and-4 at opponent 6,
    # trailing 27-31 with 4:59 and both teams holding three timeouts.
    # Fixed estimates mirror the observed live official model output;
    # the reliability probe is mocked to avoid a second live engine.
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=15)
    at.session_state["football_field"] = {"yardline_100": 6}
    ui.engine.wp_overrides = {
        "NORMAL_GO_PASS": .3545,
        "FIELD_GOAL": .4157,
    }
    ui.engine.unavailable = {"NORMAL_GO_RUN", "PUNT"}
    ui.probe.return_value = {
        "applicable": True, "material": True,
        "legacy_conditional_made_fg_wp": .4244,
        "native_conditional_made_fg_wp": .3743,
        "absolute_difference_pp": 5.01,
        "assumed_fg_execution_seconds": 5.,
    }
    run(at)
    widget(at, "text_input", "Time remaining (MM:SS)").set_value("04:59")
    widget(at, "number_input", "Yards to go").set_value(4)
    widget(at, "number_input", "Offense score").set_value(27)
    widget(at, "number_input", "Defense score").set_value(31)
    run(at)
    analyze(at)

    observed_state = ui.engine.calls[0][0]
    assert observed_state["game_seconds_remaining"] == 299.
    assert observed_state["yardline_100"] == 6
    assert observed_state["ydstogo"] == 4.
    assert observed_state["score_differential"] == -4.
    assert observed_state["site"] == "AWAY"
    assert ui.probe.call_count == 1
    assert len(ui.engine.calls) == 1
    assert metrics(at)["Recommendation"] == "INCONCLUSIVE"
    assert metrics(at)["Frozen-model FG WP"] == "41.57%"
    assert metrics(at)["Unverified FG lead vs GO FOR IT"] == "+6.12 pp"
    assert metrics(at)["FIELD GOAL"] == "41.57%"
    assert metrics(at)["PASS / DROPBACK"] == "35.45%"
    assert "No dependable pick" in values(at, "info")
    assert "Model reliability warning" in values(at, "warning")
    assert "separate experimental panel" not in values(at, "warning")
    assert "2025 kickoff comparison" not in [w.label for w in at.expander]
    assert not any(fn.called for fn in ui.previews.values())


def test_q4_nonmaterial_fg_leader_retains_original_recommendation(ui):
    ui.engine.wp_overrides = {"NORMAL_GO_PASS": .35, "FIELD_GOAL": .41}
    ui.engine.unavailable = {"NORMAL_GO_RUN"}
    ui.probe.return_value = {"applicable": True, "material": False}
    analyze(ui.at)
    assert metrics(ui.at)["Recommendation"] == "FIELD GOAL"
    assert metrics(ui.at)["Win probability if chosen"] == "41.00%"
    assert metrics(ui.at)["Advantage over GO FOR IT"] == "+6.00 pp"
    assert "Frozen-model FG WP" not in metrics(ui.at)
    assert "Model reliability warning" not in values(ui.at, "warning")


def test_q4_fg_leader_probe_failure_fails_closed_without_changing_wp(ui):
    ui.engine.wp_overrides = {"NORMAL_GO_PASS": .35, "FIELD_GOAL": .41}
    ui.engine.unavailable = {"NORMAL_GO_RUN"}
    ui.probe.side_effect = RuntimeError("probe unavailable")
    analyze(ui.at)
    assert metrics(ui.at)["Recommendation"] == "INCONCLUSIVE"
    assert metrics(ui.at)["Frozen-model FG WP"] == "41.00%"
    assert metrics(ui.at)["FIELD GOAL"] == "41.00%"
    assert "reliability check is unavailable" in values(ui.at, "caption")
    assert "No dependable pick" in values(ui.at, "info")
    assert len(ui.engine.calls) == 1


def test_public_q4_analysis_uses_only_official_model(ui):
    at = ui.at
    assert not any("kickoff comparison" in e.label.lower() for e in at.expander)
    assert not any("experimental" in c.label.lower() for c in at.checkbox)
    analyze(at)
    assert metrics(at)["Recommendation"] == "GO FOR IT"
    assert metrics(at)["FIELD GOAL"] == "40.00%"
    assert len(ui.engine.calls) == 1
    assert not any(fn.called for fn in ui.previews.values())
    assert not any("Experimental strategy" in name for name in metrics(at))


@pytest.mark.parametrize("ot_format,clock,timeouts", [
    ("REGULAR_SEASON", "09:00", 2), ("POSTSEASON", "14:00", 3),
])
def test_overtime_controls_state_and_game_value_labels(ui, ot_format, clock, timeouts):
    at = ui.at
    widget(at, "number_input", "Offense score").set_value(27)
    widget(at, "radio", "Quarter").set_value(5)
    run(at)
    widget(at, "radio", "Overtime format").set_value(ot_format)
    widget(at, "text_input", "Time remaining (MM:SS)").set_value(clock)
    run(at)
    assert len(widget(at, "radio", "Offense timeouts").options) == timeouts + 1
    assert not any("experimental" in item.label.lower() for item in at.checkbox)
    analyze(at)
    state = ui.engine.calls[0][0]
    assert state["ot_format"] == ot_format and state["ot_phase"] == "OPENING"
    assert state["ot_period"] == 1
    assert state["game_seconds_remaining"] == (540 if timeouts == 2 else 840)
    assert "Game value if chosen" in metrics(at)
    assert "Win probability if chosen" not in metrics(at)
    assert "Show game-value chart" in [item.label for item in at.expander]
    assert "0.5 for a tie" in values(widget(at, "expander", "Model notes"), "markdown")
    assert "before the fourth-down" not in values(at, "caption")
    odds = widget(at, "expander", EXPLAINER)
    assert "Estimated game value" in odds.dataframe[0].value.columns
    assert "Estimated game WP" not in odds.dataframe[0].value.columns
    assert "overtime game value" in values(odds, "caption")
    ui.probe.assert_not_called()


@pytest.mark.parametrize("phase,offense,clock,error", [
    ("OPENING", 24, "07:38", "opening OT possession"),
    ("RESPONSE", 28, "07:38", "responding offense cannot already be leading"),
    ("SUDDEN_DEATH", 24, "07:38", "Sudden death can only"),
    ("OPENING", 27, "10:01", "cannot exceed 10:00"),
    ("OPENING", 27, "00:00", "clock has expired"),
])
def test_invalid_overtime_states_disable_analysis(ui, phase, offense, clock, error):
    at = ui.at
    widget(at, "radio", "Quarter").set_value(5)
    widget(at, "number_input", "Offense score").set_value(offense)
    widget(at, "text_input", "Time remaining (MM:SS)").set_value(clock)
    run(at)
    widget(at, "radio", "OT possession").set_value(phase)
    run(at)
    assert widget(at, "button", "Analyze Fourth Down").disabled
    assert error in values(at, "error")
    assert not ui.engine.calls


@pytest.mark.parametrize("phase,offense", [("RESPONSE", 24), ("SUDDEN_DEATH", 27)])
def test_valid_response_and_sudden_death_states_reach_engine(ui, phase, offense):
    widget(ui.at, "radio", "Quarter").set_value(5)
    widget(ui.at, "number_input", "Offense score").set_value(offense)
    run(ui.at)
    widget(ui.at, "radio", "OT possession").set_value(phase)
    run(ui.at)
    analyze(ui.at)
    assert ui.engine.calls[0][0]["ot_phase"] == phase


@pytest.mark.parametrize("quarter,site", [(3, "Away"), (4, "Home"), (5, "Neutral")])
def test_real_frozen_engine_loads_and_renders_without_live_data(monkeypatch, quarter, site):
    """No mocks of recommend(), no exact Monte Carlo winner assertions."""
    monkeypatch.chdir(ROOT)
    st.cache_resource.clear()
    try:
        at = run(AppTest.from_file(str(ROOT / "app.py"), default_timeout=60))
        widget(at, "radio", "Quarter").set_value(quarter)
        widget(at, "radio", "Game site").set_value(site)
        if quarter == 5:
            widget(at, "number_input", "Offense score").set_value(27)
        run(at)
        analyze(at)
        assert metrics(at)["Recommendation"] in {"GO FOR IT", "FIELD GOAL", "PUNT"}
        odds = widget(at, "expander", EXPLAINER).dataframe[0].value
        assert not odds.empty
        label_map = {"GO — PASS / DROPBACK": "PASS / DROPBACK", "GO — RUN": "RUN",
                     "FIELD GOAL": "FIELD GOAL", "PUNT": "PUNT"}
        value_column = "Estimated game value" if quarter == 5 else "Estimated game WP"
        for _, row in odds.iterrows():
            assert row[value_column] == metrics(at)[label_map[row["Play"]]]
            assert 0 <= float(row[value_column].rstrip("%")) <= 100
        if quarter == 4:
            assert "Model reliability warning" in values(at, "warning")
    finally:
        st.cache_resource.clear()
