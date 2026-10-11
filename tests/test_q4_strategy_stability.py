"""The Q4 stability grid is fixed before results and cannot mutate production."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT=Path(__file__).resolve().parents[1]
SCRIPT=ROOT/"src/78_audit_q4_strategy_stability.py"
spec=importlib.util.spec_from_file_location("q4_local_stability_78",SCRIPT)
mod=importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def frame(*, fg=.33, run=.42, passing=.40, punt=.20, se=.002,
          eligible=True):
    return {"results":pd.DataFrame([
        {
            "action":action,
            "eligible":available,
            "expected_win_probability":prob,
            "mc_se":se,
        }
        for action,prob,available in (
            ("FIELD_GOAL",fg,eligible),
            ("NORMAL_GO_RUN",run,True),
            ("NORMAL_GO_PASS",passing,True),
            ("PUNT",punt,True),
        )
    ])}


def test_prespecified_grid_changes_one_dimension_at_a_time():
    cases=mod.build_states()
    assert len(cases)==14
    assert len({name for name,_ in cases})==14
    assert cases[0]==("reference",mod.BASE)
    for label,state in cases:
        delta={key for key,val in state.items() if mod.BASE[key]!=val}
        assert len(delta)==(0 if label=="reference" else 1)
        assert state["qtr"]==4
        assert 0<state["ydstogo"]<=state["yardline_100"]
        assert state["game_seconds_remaining"]>0
    assert not any("2026" in label for label,_ in cases)


def test_snapshot_ranks_go_over_fg_even_when_run_pass_is_too_close():
    details=mod.snapshot(frame(fg=.3244,run=.4157,passing=.4082,se=.004))
    assert details["supported"]
    assert details["top_strategy"]=="GO FOR IT"
    assert details["go_best_action"]=="NORMAL_GO_RUN"
    assert details["go_minus_fg_pp"]==pytest.approx(9.13)
    assert not details["go_vs_fg_near_tie_mc_only"]
    assert details["run_vs_pass"]["near_tie_mc_only"]


def test_ineligible_fg_is_reported_not_fabricated():
    vals=mod.snapshot(frame(eligible=False))
    assert not vals["supported"]
    assert "FG" in vals["reason"]
    assert vals["eligible_actions"] == [
        "NORMAL_GO_PASS", "NORMAL_GO_RUN", "PUNT"
    ]


def test_report_counts_scenario_sign_changes_and_separate_mc_ties():
    data=[]
    for label,o,e in [
        ("reference",.07,.09),("distance",-.01,.01),("field",.005,-.02)
    ]:
        original=mod.snapshot(frame(fg=.31,run=.31+o,
                                    passing=.30,se=.002))
        experimental=mod.snapshot(frame(fg=.32,run=.32+e,
                                        passing=.30,se=.002))
        data.append({
            "name":label,
            "official":original,
            "expanded_2025":experimental,
        })
    summary=mod.summarize(data)
    assert summary["prespecified_states"]==3
    assert summary["paired_supported"]==3
    assert summary["official"]["go_above_fg_point_count"]==2
    assert summary["expanded_2025"]["go_above_fg_point_count"]==2
    assert summary["point_margin_sign_changed_between_models"]==2
    assert summary["strategy_leader_changed_between_models"]==2


def test_invalid_simulated_probability_or_missing_actions_fails_closed():
    bad=frame()
    bad["results"].loc[bad["results"]["action"].eq("FIELD_GOAL"),
                       "mc_se"]=np.nan
    with pytest.raises(ValueError,match="Invalid WP or MC uncertainty"):
        mod.snapshot(bad)
    dup=frame()
    dup["results"]=pd.concat([dup["results"],
                             dup["results"].iloc[[0]]],ignore_index=True)
    with pytest.raises(ValueError,match="Duplicate"):
        mod.snapshot(dup)


def test_audit_cannot_load_2026_or_write_live_engine():
    code=SCRIPT.read_text()
    assert "preview_observed_2025_resolved_recommendation(" in code
    assert "isolated_engine_for_preview()" in code
    assert "OUTPUT=ROOT/\"artifacts/" in code
    assert "load_pbp([2026])" not in code
    assert "joblib.dump(" not in code
    assert "app.py" not in code
