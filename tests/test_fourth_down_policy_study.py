"""Synthetic-only regression for the sealed 2027 fourth-down study.

No nflverse access, no 2026 Weeks 6-18 access, no engine/model refits.
"""
import json
from pathlib import Path

import pytest

from src.fourth_down_policy_study import (
    POLICY_COMMIT,
    evaluate,
)

MANIFEST = {
    "mode": "synthetic",
    "policy_commit": POLICY_COMMIT,
    "feature_policy": "pre_snap_only",
    "holdout_season": 2027,
    "nuisance_training_end_season": 2025,
}


def record(
    game="g1", play="p1", *,
    observed="FIELD_GOAL", policy="FIELD_GOAL", y=1,
    e=0.5, mu=0.6, q_observed=0.6, q_policy=0.6,
    quarter=4,
):
    return {
        "game_id": game,
        "play_id": play,
        "season": 2027,
        "week": 6,
        "quarter": quarter,
        "observed_action": observed,
        "policy_action": policy,
        "eligible_actions": [
            "NORMAL_GO_PASS", "FIELD_GOAL", "PUNT",
        ],
        "offense_win": y,
        "q_wp_observed": q_observed,
        "q_wp_policy": q_policy,
        "mu_policy": mu,
        "propensity_policy": e,
        "feature_asof": "pre_snap",
    }


def test_known_two_row_dr_and_factual_brier():
    rows = [
        record("a", "1", y=1, q_observed=.6, e=.5, mu=.6),
        record("b", "1", observed="PUNT", y=0, q_observed=.2,
               e=.5, mu=.3),
    ]
    out = evaluate(rows, MANIFEST, bootstrap_draws=100)
    cal = out["factual_action_calibration"]
    ope = out["observational_one_step_policy_comparison"]
    assert cal["brier"] == pytest.approx(.10)
    assert cal["factual_supported_decisions"] == 2
    assert cal["by_observed_action"]["FIELD_GOAL"]["decisions"] == 1
    assert cal["by_observed_action"]["PUNT"]["decisions"] == 1
    # Matched: .6 + (1-.6)/.5 = 1.4; unmatched: .3
    # DR policy .85, coaches .50, difference .35
    assert ope["dr_one_step_policy_value"] == pytest.approx(.85)
    assert ope["coaches_observed_win_rate_on_supported"] == pytest.approx(.50)
    assert ope["dr_minus_observed_coaches_value"] == pytest.approx(.35)
    assert ope["matched_importance_weight_ess"] == pytest.approx(1.)
    assert not ope["sufficient_overlap_to_report_exploratory_comparison"]
    assert "NOT causal" in ope["interpretation"]
    assert len(ope["dr_difference_95pct_game_cluster_ci"]) == 2


def test_bootstrap_is_deterministic_and_game_clustered():
    # 10 plays but only two independent games: cannot claim 10 games.
    rows = []
    for i in range(5):
        rows.extend([
            record("home_win", f"{i}", y=1, q_observed=.8),
            record("away_win", f"{i}", observed="PUNT", y=0, q_observed=.2),
        ])
    a = evaluate(rows, MANIFEST, bootstrap_draws=120, seed=2027)
    b = evaluate(rows, MANIFEST, bootstrap_draws=120, seed=2027)
    assert a == b
    assert a["n_decisions"] == 10
    assert a["n_games"] == 2
    assert a["factual_action_calibration"]["distinct_games"] == 2
    assert a["observational_one_step_policy_comparison"]["distinct_supported_games"] == 2
    assert not a["observational_one_step_policy_comparison"]["sufficient_overlap_to_report_exploratory_comparison"]


def test_support_is_selected_only_by_pre_snap_propensity_not_logged_action():
    rows = [
        record("g1", e=.049, observed="FIELD_GOAL"),
        record("g2", e=.051, observed="PUNT", y=0),
        record("g3", e=.95, observed="FIELD_GOAL"),
        record("g4", e=.951, observed="PUNT", y=0),
    ]
    ope = evaluate(rows, MANIFEST, bootstrap_draws=30)["observational_one_step_policy_comparison"]
    assert ope["supported_decisions"] == 2
    assert ope["distinct_supported_games"] == 2
    assert ope["disagreements_with_coaches"] == 1
    assert ope["support_coverage"] == .5


def test_unsupported_observed_choice_has_no_factual_calibration_but_is_counted():
    r = record(observed="NORMAL_GO_RUN", q_observed=None)
    out = evaluate([r], MANIFEST, bootstrap_draws=20)
    assert out["factual_action_calibration"]["factual_supported_decisions"] == 0
    assert out["factual_action_calibration"]["factual_unsupported_decisions"] == 1
    assert out["observational_one_step_policy_comparison"]["supported_decisions"] == 1


@pytest.mark.parametrize(
    "field,value,part",
    [
        ("season", 2026, "season"),
        ("season", 2025, "season"),
        ("quarter", 5, "quarter"),
        ("feature_asof", "post_snap", "pre-snap"),
        ("policy_action", "FAKE_FG_RUN", "normal GO"),
        ("observed_action", "FAKE_PUNT_RUN", "normal GO"),
        ("offense_win", .5, "non-tied"),
        ("offense_win", True, "non-tied"),
        ("q_wp_observed", 1.5, "[0, 1]"),
        ("q_wp_policy", -0.01, "[0, 1]"),
        ("q_wp_policy", float("nan"), "finite"),
        ("mu_policy", None, "numeric"),
        ("propensity_policy", float("inf"), "finite"),
        ("feature_asof", "post_game_final", "pre-snap"),
    ],
)
def test_fail_closed_on_unsafe_or_invalid_records(field, value, part):
    r = record()
    r[field] = value
    with pytest.raises(ValueError, match=part):
        evaluate([r], MANIFEST, bootstrap_draws=20)


def test_duplicate_decision_fails_closed():
    with pytest.raises(ValueError, match="Duplicate"):
        evaluate([record(), record()], MANIFEST, bootstrap_draws=20)


def test_policy_must_be_pre_snap_eligible():
    r = record()
    r["eligible_actions"] = ["PUNT"]
    with pytest.raises(ValueError, match="eligible"):
        evaluate([r], MANIFEST, bootstrap_draws=20)


def test_missing_factual_wp_is_not_allowed_for_supported_action():
    with pytest.raises(ValueError, match="q_wp_observed"):
        evaluate([record(q_observed=None)], MANIFEST, bootstrap_draws=20)


@pytest.mark.parametrize(
    "key,value,expected",
    [
        ("policy_commit", "HEAD", "frozen"),
        ("feature_policy", "post_game", "pre_snap_only"),
        ("holdout_season", 2026, "2027"),
        ("nuisance_training_end_season", 2026, "exclude all 2026"),
        ("mode", "unsealed", "synthetic or prospective"),
    ],
)
def test_manifest_cannot_peek_or_refit(key, value, expected):
    m = {**MANIFEST, key: value}
    with pytest.raises(ValueError, match=expected):
        evaluate([record()], m, bootstrap_draws=20)


def test_prospective_guard_requires_complete_regular_season_and_provenance():
    m = {**MANIFEST, "mode": "prospective"}
    with pytest.raises(ValueError, match="completed 2027"):
        evaluate([record()], m, bootstrap_draws=20)
    m["final_2027_regular_season_complete"] = True
    with pytest.raises(ValueError, match="provenance"):
        evaluate([record()], m, bootstrap_draws=20)
    m["nuisance_provenance"] = "frozen model training source: <=2025"
    with pytest.raises(ValueError, match="100 distinct games"):
        evaluate([record()], m, bootstrap_draws=20)


def test_large_synthetic_support_pass_is_not_a_causal_policy_claim():
    # Satisfies sample/ESS/discordance; still returns the explicit caveat.
    rows = []
    for i in range(240):
        rows.append(record(
            game=f"synthetic-{i}", play="1",
            observed="FIELD_GOAL" if i % 2 else "PUNT",
            policy="FIELD_GOAL",
            y=int(i % 3 == 0),
            mu=.5,
            e=.5,
            q_observed=.5,
        ))
    out = evaluate(rows, MANIFEST, bootstrap_draws=30)
    stat = out["observational_one_step_policy_comparison"]
    assert stat["sufficient_overlap_to_report_exploratory_comparison"]
    assert stat["distinct_supported_games"] == 240
    assert stat["matched_importance_weight_ess"] == pytest.approx(120)
    assert "cannot identify" in out["warning"]
    assert out["mode"] == "synthetic"



def test_prespecified_quarter_site_closecall_and_deep_own_groups():
    own5 = record("own5", "1", quarter=1,
                  observed="NORMAL_GO_PASS", policy="FIELD_GOAL",
                  y=1, q_observed=.6)
    own5.update(site="AWAY", close_call=False,
                yardline_100=95., ydstogo=3.)
    q4 = record("q4", "1", quarter=4,
                observed="FIELD_GOAL", y=0, q_observed=.4)
    q4.update(site="HOME", close_call=True,
              yardline_100=6., ydstogo=4.)
    unsupported = record("unsupported", "1", quarter=4,
                         observed="NORMAL_GO_RUN", y=1, q_observed=None)
    unsupported.update(site="NEUTRAL", close_call=False,
                       yardline_100=95., ydstogo=3.)
    out = evaluate([own5, q4, unsupported], MANIFEST, bootstrap_draws=20)
    s = out["prespecified_descriptive_subgroups"]
    assert "NOT causal" in s["classification"] or "causal" in s["classification"]
    assert s["q4_vs_q1_to_q3"]["Q1_TO_Q3"]["decisions"] == 1
    assert s["q4_vs_q1_to_q3"]["Q4"]["decisions"] == 2
    assert s["offense_site"]["HOME"]["factual_supported_decisions"] == 1
    assert s["offense_site"]["AWAY"]["decisions"] == 1
    assert s["offense_site"]["NEUTRAL"]["factual_unsupported_decisions"] == 1
    assert s["close_call"]["YES"]["decisions"] == 1
    assert s["close_call"]["NO"]["decisions"] == 2
    own = s["deep_own_short_yardage"]["OWN_1_TO_10_4TH_2_OR_3"]
    assert own["decisions"] == 2
    assert own["factual_supported_decisions"] == 1
    assert own["factual_unsupported_decisions"] == 1
    assert own["factual_brier"] == pytest.approx(.16)
    assert own["mean_factual_predicted_wp"] == pytest.approx(.6)
    assert own["factual_observed_win_rate"] == pytest.approx(1.)
    assert s["deep_own_short_yardage"]["OTHER"]["decisions"] == 1


def test_missing_optional_synthetic_group_metadata_is_explicit_not_imputed():
    out = evaluate([record()], MANIFEST, bootstrap_draws=20)
    s = out["prespecified_descriptive_subgroups"]
    assert s["offense_site"]["NOT_RECORDED"]["decisions"] == 1
    assert s["close_call"]["NOT_RECORDED"]["decisions"] == 1
    assert s["deep_own_short_yardage"]["NOT_RECORDED"]["decisions"] == 1


@pytest.mark.parametrize("field,value,expected", [
    ("site", "N", "site"),
    ("close_call", "yes", "close_call"),
    ("yardline_100", False, "yardline_100"),
    ("yardline_100", 100, "yardline_100"),
    ("ydstogo", float("inf"), "ydstogo"),
    ("ydstogo", -1, "ydstogo"),
])
def test_prespecified_metadata_validation_fails_closed(field, value, expected):
    obj = record()
    obj.update(site="HOME", close_call=True, yardline_100=6, ydstogo=4)
    obj[field] = value
    with pytest.raises(ValueError, match=expected):
        evaluate([obj], MANIFEST, bootstrap_draws=20)


def test_prospective_requires_prespecified_metadata_without_peeking():
    obj = {**MANIFEST, "mode": "prospective",
           "final_2027_regular_season_complete": True,
           "nuisance_provenance": "synthetic nuisance metadata"}
    records = [record(f"future-game-{n}", "1") for n in range(100)]
    with pytest.raises(ValueError, match="requires complete pre-snap"):
        evaluate(records, obj, bootstrap_draws=20)



def test_cli_synthetic_fixture_roundtrip(tmp_path):
    import subprocess
    import sys
    root = Path(__file__).resolve().parents[1]
    fixture = tmp_path / "sealed.json"
    dest = tmp_path / "out.json"
    fixture.write_text(json.dumps({
        "manifest": MANIFEST,
        "records": [record(), record("g2", "2", observed="PUNT", y=0)],
    }))
    subprocess.run(
        [sys.executable, "-m", "src.fourth_down_policy_study",
         str(fixture), "--output", str(dest)],
        cwd=root, check=True,
    )
    result = json.loads(dest.read_text())
    assert result["mode"] == "synthetic"
    assert result["n_decisions"] == 2
    assert result["season"] == 2027


def test_module_has_no_live_2026_data_loader_or_network_side_effects():
    content = (
        Path(__file__).resolve().parents[1]
        / "src" / "fourth_down_policy_study.py"
    ).read_text()
    assert "import nflverse" not in content.lower()
    assert "requests.get" not in content.lower()
    assert "urllib.request" not in content.lower()
    assert "import nfl_data_py" not in content.lower()
