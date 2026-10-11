"""Presentation-only immediate action explanation regression tests.

No real 2026 outcomes, additional simulations in the Streamlit UI,
or model modifications are required by these explainers.
"""
from pathlib import Path
import ast

import numpy as np
import pandas as pd
import pytest

from src.fourth_down_action_explainer import (
    NORMAL_ACTIONS,
    action_assumptions_table,
    parse_engine_detail,
    formatted_outcomes,
)
from src.streamlit_kickoff_preview import isolated_engine_for_preview

ROOT=Path(__file__).resolve().parents[1]


def example_results():
    return pd.DataFrame([
        {
            "action":"NORMAL_GO_PASS",
            "eligible":True,
            "expected_win_probability":.4144,
            "mc_se":.004,
            "detail":"conversion=0.438",
        },
        {
            "action":"NORMAL_GO_RUN",
            "eligible":True,
            "expected_win_probability":.4202,
            "mc_se":.003,
            "detail":"conversion=0.486",
        },
        {
            "action":"FIELD_GOAL",
            "eligible":True,
            "expected_win_probability":.3408,
            "mc_se":.003,
            "detail":"make=0.730, miss=0.263, block=0.007",
        },
        {
            "action":"PUNT",
            "eligible":True,
            "expected_win_probability":.2205,
            "mc_se":.003,
            "detail":"regular=0.982, blocked=0.0050, broken=0.0130",
        },
        {
            "action":"FAKE_FIELD_GOAL_RUN",
            "eligible":False,
            "expected_win_probability":np.nan,
            "mc_se":np.nan,
            "detail":"experimental only",
        },
    ])


@pytest.mark.parametrize(("action","detail","expected"),[
    ("NORMAL_GO_PASS","conversion=0.400",{"conversion":.4}),
    ("NORMAL_GO_RUN","conversion=0.615",{"conversion":.615}),
    ("FIELD_GOAL","make=0.730, miss=0.263, block=0.007",
     {"make":.730,"miss":.263,"block":.007}),
    ("PUNT","regular=0.982, blocked=0.0050, broken=0.0130",
     {"regular":.982,"blocked":.005,"broken":.013}),
])
def test_exact_frozen_action_details_can_be_understood(action,detail,expected):
    assert parse_engine_detail(action,detail)==expected


@pytest.mark.parametrize(("action","detail"),[
    ("NORMAL_GO_RUN","conversion=nan"),
    ("NORMAL_GO_PASS","conversion=-0.1"),
    ("NORMAL_GO_PASS","conversion=1.2"),
    ("NORMAL_GO_PASS","success=0.5"),
    ("FIELD_GOAL","make=0.7, miss=0.2, block=0.2"),
    ("FIELD_GOAL","make=0.7, miss=0.3"),
    ("FIELD_GOAL","make=0.7, miss=0.3, block=0.0, extra=0.0"),
    ("PUNT","regular=0.9, blocked=0.2, broken=0.0"),
    ("PUNT","regular=0.9, regular=0.1, broken=0.0"),
    ("FAKE_FG","make=1.0"),
])
def test_invalid_details_fail_closed_without_fabricating_outcomes(action,detail):
    with pytest.raises(ValueError):
        parse_engine_detail(action,detail)


def test_explicitly_separate_immediate_outcome_from_game_wp():
    result=action_assumptions_table(example_results())
    assert result.columns.tolist()==[
        "Play","Immediate modeled outcome","Estimated game WP","MC SE (pp)",
    ]
    assert len(result)==4
    assert result["Play"].tolist()==[
        "GO — PASS / DROPBACK", "GO — RUN", "FIELD GOAL", "PUNT",
    ]
    assert result.iloc[0]["Immediate modeled outcome"]=="Convert: 43.8%"
    assert result.iloc[0]["Estimated game WP"]=="41.44%"
    assert result.iloc[0]["MC SE (pp)"]=="0.40"
    assert "Make: 73.0%" in result.iloc[2]["Immediate modeled outcome"]
    assert "Block: 0.7%" in result.iloc[2]["Immediate modeled outcome"]
    assert "Broken play: 1.3%" in result.iloc[3]["Immediate modeled outcome"]


def test_missing_eligibility_support_never_invents_percentages():
    source=example_results()
    source.loc[source["action"].eq("NORMAL_GO_RUN"),"eligible"]=False
    source.loc[source["action"].eq("FIELD_GOAL"),"eligible"]=False
    result=action_assumptions_table(source)
    assert len(result)==2
    assert "GO — RUN" not in set(result["Play"])
    assert "FIELD GOAL" not in set(result["Play"])


def test_duplicate_or_missing_normal_action_details_fail_closed():
    raw=example_results()
    duplicated=pd.concat([raw,raw.iloc[[0]]],ignore_index=True)
    with pytest.raises(ValueError,match="Duplicate"):
        action_assumptions_table(duplicated)
    raw.loc[raw["action"].eq("FIELD_GOAL"),"detail"]="make=none"
    with pytest.raises(ValueError,match="Unrecognized"):
        action_assumptions_table(raw)
    with pytest.raises(ValueError,match="Missing frozen"):
        action_assumptions_table(raw.drop(columns=["detail"]))


def test_real_frozen_engine_detail_matches_explainer_without_recomputation():
    private=isolated_engine_for_preview()
    normal={
        "qtr":4,
        "game_seconds_remaining":458.,
        "yardline_100":30.,
        "ydstogo":3.,
        "score_differential":-3.,
        "posteam_timeouts_remaining":3.,
        "defteam_timeouts_remaining":3.,
        "site":"AWAY",
        "roof":"outdoors",
    }
    before=private["recommend"](normal,n_simulations=220,seed=42)
    table=action_assumptions_table(before["results"])
    after=private["recommend"](normal,n_simulations=220,seed=42)
    pd.testing.assert_frame_equal(
        before["results"],after["results"],check_exact=True
    )
    assert not table.empty
    eligible=set(
        before["results"].loc[before["results"]["eligible"],"action"]
    )
    actual={a for a in eligible if a in NORMAL_ACTIONS}
    assert len(table)==len(actual)
    assert all(text.endswith("%") for text in table["Estimated game WP"])


def test_app_is_read_only_and_outcome_explainer_is_accessible():
    code=(ROOT/"app.py").read_text()
    ast.parse(code)
    assert "Why this recommendation? Immediate action odds" in code
    assert 'action_assumptions_table(result["results"])' in code
    assert code.count("result = adaptive_recommend(")==1
    assert "Monte Carlo SE is simulation" in code
    assert "development-data" in code
    assert code.index("Why this recommendation? Immediate action odds") > code.index(
        "Compare Actions"
    )
    src=(ROOT/"src/fourth_down_action_explainer.py").read_text()
    assert "load_pbp([2026])" not in src
    assert "joblib.dump(" not in src
    assert "evaluate_post_play_states(" not in src
