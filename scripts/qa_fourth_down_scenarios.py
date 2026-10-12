"""Real-engine QA matrix for fourth-down situations (no future outcome data).

This exercises the actual frozen 4th-down estimator. Q4 uses the same
native kickoff input correction, 1,200 draws and seed as the public app.
All other periods use the frozen engine without any research scenario.
It never fits models, downloads data, or runs Steps 66/83.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from src.streamlit_kickoff_preview import (
    isolated_engine_for_preview,
    isolated_engine_from_loaded_engine,
    preview_native_kickoff_feature_candidate,
)

NORMAL = {
    "NORMAL_GO_PASS": "GO FOR IT",
    "NORMAL_GO_RUN": "GO FOR IT",
    "FIELD_GOAL": "FIELD GOAL",
    "PUNT": "PUNT",
}


@dataclass(frozen=True)
class Case:
    name: str
    qtr: int
    secs: int
    ydstogo: float
    yardline: float
    score: float
    site: str
    offense_to: int = 3
    defense_to: int = 3
    roof: str = "outdoors"
    receiver: str | None = None
    ot_format: str | None = None
    ot_phase: str | None = None
    ot_period: int = 1
    expected: str | None = None
    ineligible: tuple[str, ...] = ()

    def state(self) -> dict[str, Any]:
        game_seconds = (
            self.secs if self.qtr >= 5 else
            (4 - self.qtr) * 900 + self.secs
        )
        state = {
            "qtr": self.qtr,
            "game_seconds_remaining": float(game_seconds),
            "yardline_100": float(self.yardline),
            "ydstogo": float(self.ydstogo),
            "score_differential": float(self.score),
            "posteam_timeouts_remaining": float(self.offense_to),
            "defteam_timeouts_remaining": float(self.defense_to),
            "site": self.site,
            "roof": self.roof,
        }
        if self.receiver:
            state["second_half_receiver"] = self.receiver
        if self.qtr >= 5:
            state.update({
                "ot_format": self.ot_format,
                "ot_phase": self.ot_phase,
                "ot_period": self.ot_period,
            })
        return state


CASES = [
    Case("Q1 early tied short go", 1, 600, 1, 38, 0, "HOME"),
    Case("Q1 backed up own 5", 1, 330, 3, 95, 0, "NEUTRAL"),
    Case("Q2 0:25 down 3 at opp22", 2, 25, 7, 22, -3, "AWAY",
         receiver="DEFENSE", ineligible=("NORMAL_GO_RUN",)),
    Case("Q3 6:15 up 7 from own45", 3, 375, 5, 55, 7, "HOME",
         roof="closed", offense_to=2, defense_to=3,
         ineligible=("NORMAL_GO_RUN", "FIELD_GOAL")),
    Case("Q4 user: down4 at opp6", 4, 299, 4, 6, -4, "AWAY",
         expected="FIELD GOAL", ineligible=("NORMAL_GO_RUN", "PUNT")),
    Case("Q4 user: down3 at opp30", 4, 458, 3, 30, -3, "AWAY",
         expected="GO FOR IT"),
    Case("Q4 final 10s down2", 4, 10, 10, 25, -2, "HOME",
         offense_to=1, defense_to=1,
         expected="FIELD GOAL", ineligible=("NORMAL_GO_RUN",)),
    Case("Q4 1:30 up7 own35", 4, 90, 10, 65, 7, "HOME",
         offense_to=2, defense_to=0, expected="PUNT",
         ineligible=("NORMAL_GO_RUN", "FIELD_GOAL")),
    Case("Q4 0:45 down4 opp1", 4, 45, 1, 1, -4, "AWAY",
         offense_to=2, defense_to=2, expected="GO FOR IT",
         ineligible=("PUNT",)),
    Case("Q4 tied 4min opp19", 4, 240, 5, 19, 0, "NEUTRAL",
         offense_to=2, defense_to=2, ineligible=("NORMAL_GO_RUN",)),
    Case("Q4 home-site sensitivity opp6", 4, 299, 4, 6, -4, "HOME",
         ineligible=("NORMAL_GO_RUN", "PUNT")),
    Case("OT regular tied opening", 5, 458, 3, 30, 0, "HOME",
         offense_to=2, defense_to=2,
         ot_format="REGULAR_SEASON", ot_phase="OPENING"),
    Case("OT regular tied response", 5, 300, 5, 35, 0, "HOME",
         offense_to=2, defense_to=2, expected="FIELD GOAL",
         ot_format="REGULAR_SEASON", ot_phase="RESPONSE",
         ineligible=("NORMAL_GO_RUN",)),
    Case("OT playoffs opening", 5, 720, 2, 25, 0, "AWAY",
         ot_format="POSTSEASON", ot_phase="OPENING"),
    Case("OT playoffs sudden death period2", 5, 480, 3, 35, 0, "HOME",
         ot_format="POSTSEASON", ot_phase="SUDDEN_DEATH", ot_period=2),
]


def check_result(case: Case, result: dict) -> dict[str, Any]:
    rows = result["results"]
    assert not rows["action"].duplicated().any(), case.name
    assert set(rows["action"]).issuperset(NORMAL), case.name
    eligible = rows.loc[rows["eligible"]].copy()
    assert len(eligible) >= 2, f"{case.name}: fewer than two ranked actions"
    assert set(eligible["action"]).issubset(NORMAL), (
        f"{case.name}: unsupported fake action was ranked"
    )
    assert eligible["expected_win_probability"].between(0, 1).all(), case.name
    assert eligible["mc_se"].notna().all(), case.name
    assert (eligible["mc_se"] >= 0).all(), case.name
    assert not eligible["detail"].isna().any(), case.name
    for action in case.ineligible:
        row = rows.loc[rows["action"].eq(action)].iloc[0]
        assert not bool(row["eligible"]), f"{case.name}: {action} leaked"
        assert math.isnan(float(row["expected_win_probability"])), (
            f"{case.name}: ineligible {action} displays a win probability"
        )
    for _, row in rows.loc[~rows["eligible"]].iterrows():
        assert math.isnan(float(row["expected_win_probability"])), (
            f"{case.name}: an unavailable action has fabricated WP"
        )

    eligible.sort_values("expected_win_probability", ascending=False,
                         inplace=True)
    chosen = eligible.iloc[0]
    assert result["recommendation"] == chosen["decision"], (
        f"{case.name}: engine recommendation inconsistent with action values"
    )
    if len(eligible) > 1:
        edge = 100 * float(
            eligible.iloc[0]["expected_win_probability"]
            - eligible.iloc[1]["expected_win_probability"]
        )
        assert math.isclose(
            edge, float(result["edge_over_second_best_pct"]),
            rel_tol=1e-8, abs_tol=1e-8
        ), f"{case.name}: engine action-level edge not reproducible"
    assert 0 <= float(result["pre_decision_wp"]) <= 1, case.name

    # Mirror Streamlit's strategy choice: PASS/RUN are one GO strategy.
    strategy_wp: dict[str, float] = {}
    action_wp: dict[str, float] = {}
    for _, row in eligible.iterrows():
        name = str(row["action"])
        value = float(row["expected_win_probability"])
        action_wp[name] = value
        strat = NORMAL[name]
        strategy_wp[strat] = max(strategy_wp.get(strat, -1.0), value)
    ranked = sorted(strategy_wp.items(), key=lambda kv: kv[1], reverse=True)
    best, runner = ranked[:2]
    strategy_edge = 100 * (best[1] - runner[1])
    assert strategy_edge >= -1e-10, case.name
    if case.expected:
        assert best[0] == case.expected, (
            f"{case.name}: expected {case.expected}, got {best[0]}"
        )

    if case.name == "Q4 user: down4 at opp6":
        assert round(100 * action_wp["FIELD_GOAL"], 2) == 37.00, (
            "User's corrected FG WP changed unexpectedly"
        )
        assert round(100 * action_wp["NORMAL_GO_PASS"], 2) == 34.34, (
            "User's corrected PASS WP changed unexpectedly"
        )
        assert round(strategy_edge, 2) == 2.66, (
            "User's corrected displayed edge changed unexpectedly"
        )
    if case.name == "Q4 user: down3 at opp30":
        assert "NORMAL_GO_RUN" in action_wp and "NORMAL_GO_PASS" in action_wp
        assert best[0] == "GO FOR IT"
        assert math.isclose(
            strategy_edge,
            100 * (max(action_wp["NORMAL_GO_RUN"], action_wp["NORMAL_GO_PASS"])
                   - action_wp["FIELD_GOAL"]),
            abs_tol=1e-10,
        ), "User's GO vs FG gap is not consistent"

    return {
        "case": case.name,
        "quarter": f"OT-{case.ot_format}" if case.qtr == 5 else f"Q{case.qtr}",
        "winner": best[0],
        "wp": round(100 * best[1], 2),
        "second": runner[0],
        "second_wp": round(100 * runner[1], 2),
        "edge_pp": round(strategy_edge, 2),
        "eligible": ",".join(str(a) for a in eligible["action"]),
    }


def main() -> None:
    engine = isolated_engine_for_preview()
    baseline_recommend = engine["recommend"]
    baseline_evaluate = engine["evaluate_post_play_states"]
    rows = []
    print("Scenario QA: 15 real-engine states; no future play outcomes / model refit", flush=True)
    for case in CASES:
        state = case.state()
        if case.qtr == 4:
            # Separate per-request functions; loaded model data shared read-only.
            private = isolated_engine_from_loaded_engine(engine)
            unmodified = private["evaluate_post_play_states"]
            result = preview_native_kickoff_feature_candidate(
                state, simulations=1200, action_seed=43,
                isolated_engine=private,
            )
            assert private["evaluate_post_play_states"] is unmodified, (
                f"{case.name}: temporary Q4 patch leaked"
            )
        else:
            result = engine["recommend"](state, n_simulations=300, seed=42)
        assert engine["recommend"] is baseline_recommend
        assert engine["evaluate_post_play_states"] is baseline_evaluate
        entry = check_result(case, result)
        rows.append(entry)
        print(
            f"{entry['case']:<36} {entry['quarter']:<18} "
            f"{entry['winner']:<12} {entry['wp']:>6.2f}% "
            f"vs {entry['second']:<12} {entry['second_wp']:>6.2f}% "
            f"edge {entry['edge_pp']:+6.2f} pp",
            flush=True,
        )
    assert len(rows) == len(CASES) and len(rows) == 15
    assert {r["quarter"] for r in rows}.issuperset(
        {"Q1", "Q2", "Q3", "Q4", "OT-REGULAR_SEASON", "OT-POSTSEASON"}
    )
    assert {r["winner"] for r in rows}.issuperset(
        {"GO FOR IT", "FIELD GOAL", "PUNT"}
    ), "Test states do not exercise all three normal strategies"
    print(
        "PASS: 15 states; all normal-action rankings, probabilities, "
        "eligibility exclusions, overtime variants and user anchors consistent.",
        flush=True,
    )


if __name__ == "__main__":
    main()
