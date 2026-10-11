"""Q4 specialist evaluation must preserve chronology and production."""

import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

PATH=Path(__file__).resolve().parents[1]/"src/71_evaluate_q4_specialist.py"
spec=importlib.util.spec_from_file_location("step71_fixed_q4",PATH)
mod=importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def small_history():
    rows=[]
    for year in (2023,2024,2025):
        for i in range(20):
            rows.append({
                "season":year,"game_id":f"{year}_{i//4}",
                "qtr":4 if i%2 else 3,
                "game_seconds_remaining":240 if i%3 else 340,
                "final_home_margin":7 if i%2 else -4,
                "home_score_differential":-2,
                "is_home_posteam":float(i%2),
                "yardline_100":70,
            })
    return pd.DataFrame(rows)


def test_last_five_uses_only_q4_final_five_and_excludes_2026():
    raw=small_history()
    selected=mod.last_five(raw)
    assert len(selected)>0
    assert selected["qtr"].eq(4).all()
    assert selected["game_seconds_remaining"].le(300).all()
    with pytest.raises(ValueError,match="Only completed"):
        mod.last_five(raw.assign(season=2026))


def test_chronology_and_game_isolation():
    data=mod.last_five(small_history())
    train,valid=mod.test_split(data,2023,2024)
    assert train["season"].max()==2023
    assert valid["season"].min()==2024
    assert not set(train["game_id"])&set(valid["game_id"])
    with pytest.raises(ValueError,match="fixed"):
        mod.test_split(data,2024,2024)


def test_grouped_brier_calibration_and_cluster_interval():
    frame=pd.DataFrame({"game_id":["g1","g1","g2","g3"],
                        "final_home_margin":[7,-3,7,-3]})
    original=np.array([.5,.5,.5,.5])
    candidate=np.array([.7,.3,.7,.3])
    def ci(diff,game_ids):
        assert len(diff)==len(game_ids)==4
        return float(diff.min()),float(diff.max())
    result=mod.subgroup_metrics("test",frame,original,candidate,ci)
    assert result["n"]==4
    assert result["games"]==3
    assert result["improvement"]>0
    assert result["ci95"][0]>0
    with pytest.raises(ValueError,match="invalid"):
        mod.subgroup_metrics("test",frame,np.array([.4]),candidate,ci)


def test_script_never_downloads_2026_or_writes_production():
    code=PATH.read_text()
    assert "step.load_seasons()" in code
    assert "((2023,2024),(2024,2025))" in code
    assert "nfl.load_pbp([2026])" not in code
    assert "joblib.dump(" not in code
    assert "2026" in code
