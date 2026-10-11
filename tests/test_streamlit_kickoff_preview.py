"""Experimental web preview must remain isolated from production recommendations."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import streamlit_kickoff_preview as preview


def recommendation(fg, go, *, fg_se=0.002, go_se=0.002):
    return {
        "results": pd.DataFrame([
            {
                "action": "FIELD_GOAL", "eligible": True,
                "expected_win_probability": fg, "mc_se": fg_se,
            },
            {
                "action": "NORMAL_GO_PASS", "eligible": True,
                "expected_win_probability": go, "mc_se": go_se,
            },
            {
                "action": "NORMAL_GO_RUN", "eligible": False,
                "expected_win_probability": np.nan, "mc_se": np.nan,
            },
            {
                "action": "FAKE_FG_PASS", "eligible": False,
                "expected_win_probability": np.nan, "mc_se": np.nan,
            },
        ]),
    }


def fake_donors():
    return pd.DataFrame([
        {
            "season": 2024,
            "game_id": f"g{j}", "quarter": 4,
            "transition_type": "touchback",
            "own_yardline": 30., "elapsed_seconds": 0.,
        }
        for j in range(40)
    ] + [
        {
            "season": 2024,
            "game_id": f"r{j}", "quarter": 4,
            "transition_type": "return",
            "own_yardline": 29., "elapsed_seconds": 6.,
        }
        for j in range(40)
    ])


def fake_engine():
    """Dedicated globals, as if imported into an isolated module."""
    ns = {"np": np, "pd": pd}
    exec(
        "def evaluate_post_play_states(base, states):\n"
        "    return np.array([s['yardline_100'] for s in states], dtype=float)\n"
        "def recommend(state, n_simulations, seed):\n"
        "    rows = [{\n"
        "       'qtr': state['qtr'], '_score_change_original': 3.,\n"
        "       'is_kickoff': 1., 'yardline_100': np.nan,\n"
        "       'down': np.nan, 'ydstogo': np.nan, 'goal_to_go': 0.,\n"
        "       'game_seconds_remaining': 200.,\n"
        "       'half_seconds_remaining': 200., 'final_five_minutes': 1.,\n"
        "    }]\n"
        "    pos = evaluate_post_play_states({'qtr': 4}, rows)[0]\n"
        "    return {'results': pd.DataFrame([\n"
        "        {'action':'FIELD_GOAL','eligible':True,\n"
        "         'expected_win_probability': .25+.001*pos,\n"
        "         'mc_se':.002},\n"
        "        {'action':'NORMAL_GO_PASS','eligible':True,\n"
        "         'expected_win_probability':.30,'mc_se':.002},\n"
        "    ])}\n",
        ns,
    )
    return ns


def test_preview_scenario_never_mutates_cached_production_engine(
    monkeypatch, tmp_path,
):
    pool_path = tmp_path / "2024.parquet"
    fake_donors().to_parquet(pool_path)
    private = fake_engine()
    production = fake_engine()
    native = production["evaluate_post_play_states"]
    isolated = private["evaluate_post_play_states"]
    monkeypatch.setattr(preview, "isolated_engine_for_preview", lambda: private)

    result = preview.preview_recommendation(
        {"qtr": 4}, donor_path=pool_path, touchback_rate=1.,
        simulations=200,
    )
    assert result["results"].iloc[0]["expected_win_probability"] == pytest.approx(.315)
    # The private monkeypatch is restored on exit, and the cached production
    # module was never passed into the context manager.
    assert private["evaluate_post_play_states"] is isolated
    assert native.__globals__["evaluate_post_play_states"] is native


def test_preview_excludes_untouched_overtime_and_bad_rates(tmp_path):
    dummy = tmp_path / "donors.parquet"
    dummy.write_bytes(b"not an actual parquet file")
    with pytest.raises(ValueError, match="Q4 only"):
        preview.preview_recommendation({"qtr": 5}, donor_path=dummy)
    with pytest.raises(ValueError, match="Touchback rate"):
        preview.preview_recommendation({"qtr": 4}, donor_path=dummy, touchback_rate=1.1)
    with pytest.raises(ValueError, match="simulations"):
        preview.preview_recommendation({"qtr": 4}, donor_path=dummy, simulations=100)


def test_side_by_side_values_and_experimental_only_filter():
    current = recommendation(.4157, .3545)
    alternative = recommendation(.3412, .3451)
    table = preview.action_comparison(current, alternative)
    assert table["Action"].tolist() == ["PASS / DROPBACK", "FIELD GOAL"]
    fg = table.loc[table["Action"].eq("FIELD GOAL")].iloc[0]
    assert fg["Production WP (%)"] == pytest.approx(41.57)
    assert fg["2025+ scenario WP (%)"] == pytest.approx(34.12)
    assert fg["Scenario minus production (pp)"] == pytest.approx(-7.45)
    with pytest.raises(ValueError, match="Eligibility differs"):
        preview.action_comparison(
            current,
            {"results": alternative["results"].loc[
                ~alternative["results"]["action"].eq("NORMAL_GO_PASS")
            ]},
        )


def test_preview_small_edge_explicitly_near_tie():
    r = preview.preview_rank_with_uncertainty(recommendation(.3412, .3451))
    assert r["leader"] == "PASS / DROPBACK"
    assert r["runner_up"] == "FIELD GOAL"
    assert r["gap_pp"] == pytest.approx(.39)
    assert r["near_tie"]

    r = preview.preview_rank_with_uncertainty(recommendation(
        .30, .37, fg_se=.001, go_se=.001
    ))
    assert r["leader"] == "PASS / DROPBACK"
    assert not r["near_tie"]


def test_production_app_stays_syntactically_valid():
    app = Path(__file__).resolve().parents[1] / "app.py"
    compile(app.read_text(), str(app), "exec")


def test_private_engine_loads_independent_globals_in_existing_repository():
    from src.streamlit_kickoff_preview import isolated_engine_for_preview
    # Existing joblib artifacts and models are loaded privately; this test
    # detects import path and API incompatibilities before Streamlit deploy.
    module = isolated_engine_for_preview()
    assert callable(module["recommend"])
    assert callable(module["evaluate_post_play_states"])
    assert module["recommend"].__globals__ is module



def observed_2025_donors():
    """Enough independent games to exercise the frozen artifact integrity gate."""
    rows = []
    for i in range(160):
        rows.append({
            "season": 2025, "game_id": f"G{i}", "week": (i % 18)+1,
            "kick_play_id": float(100+i), "quarter": 4,
            "transition_type": "touchback" if i % 4 == 0 else "return",
            "own_yardline": 35. if i % 4 == 0 else 29.,
            "elapsed_seconds": 0. if i % 4 == 0 else 6.,
        })
        rows.append({
            "season": 2025, "game_id": f"G{i}", "week": (i % 18)+1,
            "kick_play_id": float(300+i), "quarter": 4,
            "transition_type": "return",
            "own_yardline": 30., "elapsed_seconds": 7.,
        })
    return pd.DataFrame(rows)


def test_observed_2025_preview_really_uses_2025_donors_not_old_assumptions(
    monkeypatch, tmp_path,
):
    frozen = tmp_path / "frozen_2025.csv"
    observed_2025_donors().to_csv(frozen, index=False)
    private = fake_engine()
    original = private["evaluate_post_play_states"]
    production = fake_engine()
    prod_original = production["evaluate_post_play_states"]
    monkeypatch.setattr(preview, "isolated_engine_for_preview", lambda: private)
    result = preview.preview_observed_2025_recommendation(
        {"qtr": 4}, donor_path=frozen, simulations=200,
    )
    value = result["results"].loc[
        result["results"]["action"].eq("FIELD_GOAL"),
        "expected_win_probability",
    ].iloc[0]
    assert 0.30 < value < 0.34
    assert original.__globals__["evaluate_post_play_states"] is original
    assert prod_original.__globals__["evaluate_post_play_states"] is prod_original

    summary = preview.observed_2025_pool_summary(frozen)
    assert summary["kickoffs"] == 320
    assert summary["games"] == 160
    assert summary["touchback_rate"] == pytest.approx(.125)
    with pytest.raises(ValueError, match="Q4"):
        preview.preview_observed_2025_recommendation(
            {"qtr": 3}, donor_path=frozen
        )
    with pytest.raises(FileNotFoundError):
        preview.preview_observed_2025_recommendation(
            {"qtr": 4}, donor_path=tmp_path / "missing.csv"
        )



def test_official_recommendation_is_flagged_when_empirical_scenario_disagrees():
    baseline = recommendation(.4157, .3545)
    empirical = recommendation(.3411, .3449, fg_se=.0008, go_se=.0042)
    status = preview.recommendation_sensitivity(baseline, empirical)
    assert status["production_leader"] == "FIELD GOAL"
    assert status["scenario_point_leader"] == "PASS / DROPBACK"
    assert status["reversed"]
    assert status["scenario_near_tie"]
    assert status["model_sensitive"]


def test_agreement_with_clear_margin_is_not_flagged_as_sensitive():
    baseline = recommendation(.42, .35, fg_se=.001, go_se=.001)
    scenario = recommendation(.40, .33, fg_se=.001, go_se=.001)
    status = preview.recommendation_sensitivity(baseline, scenario)
    assert status["production_leader"] == "FIELD GOAL"
    assert not status["reversed"]
    assert not status["scenario_near_tie"]
    assert not status["model_sensitive"]


def test_model_sensitivity_is_triggered_even_without_point_rank_flipping():
    baseline = recommendation(.44, .38)
    scenario = recommendation(.355, .352, fg_se=.003, go_se=.003)
    status = preview.recommendation_sensitivity(baseline, scenario)
    assert not status["reversed"]
    assert status["scenario_near_tie"]
    assert status["model_sensitive"]
