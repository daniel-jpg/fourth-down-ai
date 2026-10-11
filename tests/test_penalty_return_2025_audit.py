"""No penalized kickoff is silently treated as an ordinary return donor."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "src/76_freeze_penalty_return_continuation.py"
spec = importlib.util.spec_from_file_location("audit_2025_special_76", SCRIPT)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def sample_2025_exceptions():
    rows = []
    donors = []

    def event(game, play, kind, *, own35=35., desc="",
              penalty=0, td=0, team="SEA", sec=290.,
              qtr=4, home_score=31, away_score=30, down=np.nan,
              yd=np.nan):
        return {
            "season":2025, "season_type":"REG", "game_id":game,
            "play_id":float(play), "play_type":kind, "qtr":qtr,
            "posteam":team, "week":1,
            "yardline_100":own35,
            "game_seconds_remaining":sec,
            "home_score":home_score, "away_score":away_score,
            "down":down, "desc":desc, "penalty":penalty,
            "return_touchdown":td,
            "posteam_score":31, "defteam_score":30,
        }

    # 488 ordinary donors, each uniquely mapped to a frozen kickoff.
    for i in range(488):
        g = f"ordinary-{i}"
        rows.append(event(g,100,"kickoff",desc="Touchback"))
        donors.append({"game_id":g,"kick_play_id":100.})

    # 52 penalty/re-kick category (no onside).
    for i in range(52):
        g=f"pen-{i}"
        rows.append(event(g,100,"kickoff",penalty=1,desc="Holding penalty"))
        if i < 32:
            rows.append(event(g,110,"run",team="SEA",sec=286.,
                              own35=65.,down=1,desc="Clean first snap"))
        elif i < 40:
            # Rekick from shifted spot, excluded from standard-35 universe.
            rows.append(event(g,105,"kickoff",own35=40.,sec=289.,
                              desc="Repeat kickoff after penalty"))
        elif i < 44:
            rows.append(event(g,110,"pass",team="LA",own35=60.,
                              sec=287.,down=1,desc="Unexpected possessor"))
        elif i < 47:
            rows.append(event(g,110,"run",team="SEA",own35=55.,
                              sec=287.,down=1,home_score=38,
                              desc="Scoring intervened"))
        # Else, no first snap and no re-kick.

    rows.append(event("td",100,"kickoff",td=1,desc="Return touchdown"))
    rows.append(event("unmatched-1",100,"kickoff",desc="No first possession"))
    rows.append(event("unmatched-2",100,"kickoff",desc="No first possession"))
    return pd.DataFrame(rows), pd.DataFrame(donors)


def test_exhaustive_step73_special_denominator_and_first_snap_reconciliation():
    raw, donors = sample_2025_exceptions()
    filtered = audit.flagged_2025(raw, donors, audit.coverage_helper())
    assert len(filtered) == 55
    assert filtered["outcome_group"].value_counts().to_dict() == {
        "penalty_or_rekick": 52,
        "other_unmatched":2,
        "kickoff_return_td":1,
    }
    mapped = audit.next_scrimmage_and_rekick(raw, filtered)
    summary = audit.summarize(filtered, mapped)
    assert summary["special_total"] == 55
    assert summary["verified_penalty_first_snap"] == 32
    assert summary["verified_penalty_games"] == 32
    reasons = summary["outcome_reason_counts"]
    assert reasons["verified_penalty_first_snap"] == 32
    assert reasons["re_kick_before_first_snap"] == 8
    assert reasons["unexpected_possession_change"] == 4
    assert reasons["score_changed_before_first_snap"] == 3
    assert reasons["not_verified"] == 5
    assert reasons["return_touchdown_requires_post_score"] == 1
    assert reasons["unknown_nonordinary_kickoff"] == 2
    assert sum(reasons.values()) == 55
    assert summary["no_2026_data"]
    assert summary["not_a_future_penalty_probability"]


def test_first_snap_never_crosses_next_kick_or_other_game():
    raw, donors = sample_2025_exceptions()
    f = audit.flagged_2025(raw, donors, audit.coverage_helper())
    rows = audit.next_scrimmage_and_rekick(raw, f)
    repeat = rows.loc[rows["game_id"].eq("pen-32")].iloc[0]
    assert repeat["outcome_reason"] == "re_kick_before_first_snap"
    assert not bool(repeat["verified_penalty_first_snap"])
    assert pd.isna(repeat["play_id_snap"])
    ordinary = rows.loc[rows["game_id"].eq("pen-0")].iloc[0]
    assert ordinary["posteam_snap"] == "SEA"
    assert ordinary["next_receiving_own_yardline"] == 35.


def test_invalid_2026_schema_or_missing_events_fail_closed():
    raw, donors = sample_2025_exceptions()
    helper = audit.coverage_helper()
    with pytest.raises(ValueError, match="Only 2025"):
        audit.flagged_2025(raw.assign(season=2026), donors, helper)
    with pytest.raises(ValueError, match="Missing PBP"):
        audit.flagged_2025(raw.drop(columns=["home_score"]), donors, helper)
    flagged = audit.flagged_2025(raw, donors, helper)
    with pytest.raises(ValueError, match="Duplicate PBP"):
        audit.next_scrimmage_and_rekick(
            pd.concat([raw, raw.iloc[[0]]]), flagged
        )


def test_2025_penalty_td_report_preserves_coverage_without_false_model_claims():
    source=SCRIPT.read_text()
    assert "nfl.load_pbp([2025])" in source
    assert "return_touchdown_requires_post_score" in source
    assert "re_kick_before_first_snap" in source
    assert "nfl.load_pbp([2026])" not in source
    assert "joblib.dump(" not in source
    assert "models/win_probability_model.joblib" not in source
