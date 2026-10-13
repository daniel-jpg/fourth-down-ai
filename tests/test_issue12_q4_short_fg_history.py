"""Synthetic selection tests: only eligible historical first kickoffs are counted."""
import pandas as pd
from scripts.audit_issue12_q4_short_fg_history import (
    YEARS, select_made_fg_kickoffs, summarize,
)


def test_no_protected_seasons_in_historical_audit():
    assert YEARS == tuple(range(2014, 2026))
    assert max(YEARS) == 2025


def test_precise_prior_made_fg_and_first_kickoff_same_game():
    # Synthetic records: one made FG followed by ordinary / onside kickoff;
    # another game has a miss, which must NOT count.
    records = []
    def add(game, play, typ, post, fg, clk, yard=35., desc="", ps=0, ds=0):
        records.append(dict(
            season=2025, season_type="REG", game_id=game, play_id=play,
            play_type=typ, qtr=4, posteam=post, home_team="H",
            away_team="A", field_goal_result=fg,
            game_seconds_remaining=clk, yardline_100=yard,
            posteam_score=ps, defteam_score=ds, desc=desc,
        ))
    add("2025_01_H_A", 1, "field_goal", "A", "made", 45, 1, ps=14, ds=17)
    add("2025_01_H_A", 2, "kickoff", "H", "", 40, 35, ps=17, ds=16,
        desc="Onside kick")
    add("2025_01_H_A", 3, "kickoff", "H", "", 40, 35, ps=17, ds=16)
    add("2025_02_H_A", 1, "field_goal", "A", "missed", 35, 1)
    add("2025_02_H_A", 2, "kickoff", "H", "", 30, 35, ps=17, ds=16)
    picked = select_made_fg_kickoffs(pd.DataFrame(records))
    assert len(picked) == 1
    assert picked.iloc[0]["clock"] == 40
    assert picked.iloc[0]["receiver_lead_after_fg"] == 1
    assert picked.iloc[0]["declared_onside_text"]
    out = summarize(picked)
    assert out["2023_2025_examined_development"]["0_to_40s"][
        "receiving_team_leads_exactly_one"
    ]["plays"] == 1
