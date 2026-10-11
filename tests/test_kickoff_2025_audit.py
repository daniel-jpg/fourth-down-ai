import runpy

import numpy as np
import pandas as pd
import pytest

from src.kickoff_transition import build_2024_pool

namespace = runpy.run_path("src/55_audit_2025_kickoffs.py")
audit = namespace["audit_kickoff_pairs"]
metrics = namespace["metrics"]
check_parity = namespace["_q4_counts_match_frozen_2024"]
wilson_interval = namespace["wilson_interval"]


def row(year, game, play, kind, clock, *, team=None, own=None,
        down=None, touchback=None, desc="", quarter=4):
    return {
        "season": year, "game_id": game, "play_id": play, "play_type": kind,
        "qtr": quarter, "game_seconds_remaining": clock,
        "posteam": team, "yardline_100": (100-own if own is not None else np.nan),
        "down": down, "touchback": touchback, "desc": desc,
    }


def sample_2025():
    return pd.DataFrame([
        row(2025, "g1", 10, "kickoff", 300, team="SEA", touchback=1),
        row(2025, "g1", 11, "run", 300, team="SEA", own=35, down=1),
        row(2025, "g1", 20, "kickoff", 250, team="LAR", touchback=0),
        row(2025, "g1", 21, "pass", 244, team="LAR", own=29, down=1),
        row(2025, "g1", 22, "pass", 226, team="LAR", own=39, down=2),
        row(2025, "g1", 30, "kickoff", 200, team="SEA", touchback=1),
        row(2025, "g1", 31, "pass", 200, team="SEA", own=20, down=1),
        row(2025, "g1", 40, "kickoff", 100, team="LAR", touchback=0, desc="onside kick"),
        row(2025, "g1", 41, "run", 97, team="SEA", own=51, down=1),
        row(2025, "g1", 50, "kickoff", 70, team="LAR", touchback=0),
        row(2025, "g1", 51, "run", 66, team="SEA", own=42, down=1),
    ])


def sample_2024():
    return pd.DataFrame([
        row(2024, "g2", 10, "kickoff", 400, team="SEA", touchback=1),
        row(2024, "g2", 11, "run", 400, team="SEA", own=30, down=1),
        row(2024, "g2", 20, "kickoff", 350, team="LAR", touchback=0),
        row(2024, "g2", 21, "pass", 344, team="LAR", own=28, down=1),
    ])


def test_holdout_2025_classifies_and_retains_unmatched():
    result = audit(sample_2025(), 2025)
    assert result["status"].tolist() == [
        "touchback", "return", "touchback", "onside", "other/unmatched"
    ]
    assert result.loc[0, "own"] == 35
    assert result.loc[1, "own"] == 29
    assert result.loc[1, "elapsed"] == 6
    assert result.loc[2, "own"] == 20
    summary = metrics(result)
    assert summary["kickoffs"] == 5
    assert summary["ordinary"] == 3
    assert summary["touchbacks"] == 2
    assert summary["returns"] == 1
    assert summary["other_unmatched"] == 1
    assert summary["onside"] == 1
    assert summary["tb_35"] == 1
    assert summary["tb_20"] == 1
    assert summary["return_elapsed_median"] == 6


def test_2024_parity_matches_frozen_builder_exactly():
    pbp = sample_2024()
    audited = audit(pbp, 2024)
    check_parity(pbp, audited)
    assert len(build_2024_pool(pbp)) == 2


def test_rejects_mixed_seasons():
    pbp = pd.concat([sample_2024(), sample_2025()], ignore_index=True)
    with pytest.raises(ValueError, match="one requested season"):
        audit(pbp, 2025)


def test_2024_does_not_mistake_own35_for_touchback():
    pbp = sample_2024()
    pbp.loc[pbp["play_id"].eq(11), "yardline_100"] = 65
    audited = audit(pbp, 2024)
    assert audited.loc[0, "status"] == "other/unmatched"


def test_wilson_handles_no_observations():
    low, high = wilson_interval(0, 0)
    assert np.isnan(low) and np.isnan(high)
    low, high = wilson_interval(50, 100)
    assert low < 0.5 < high
