"""Step 69: audited 2023-25 event-model calibration for actual fourth downs.

Unlike fourth-down policy "win-probability improvement", successful conversion
and field-goal make are actually OBSERVABLE on the action that was taken. This
script assesses ONLY predictive calibration of the existing frozen model on
observed actions; it cannot validate a counterfactual FG-vs-GO policy.

Fixed groups selected for football relevance before running this report:
- Q4 last five minutes;
- GO passes on 4th-and-4, opponent 1-10, and both combined;
- GO run/pass within their supported yardage contexts;
- FG distances 18-29, 30-39, 40-49, >=50 yards.

Models, feature contracts and outputs are unchanged. No training, parameter
selection or promotion of a model. 2025 has already been examined by prior
development work and MUST NOT be called an untouched holdout.

Only 2023-2025 data are accessed; 2026 outcomes remain entirely unopened.
Run via GitHub Actions, no local action required.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys

import nflreadpy as nfl
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/fourth_down_modeling_split.parquet"
ENGINE = ROOT / "src/42_build_decision_engine.py"
REPORT = ROOT / "artifacts/2025_event_model_diagnostic.json"
SEASONS = (2023, 2024, 2025)
ACTIONS = ("NORMAL_GO_PASS", "NORMAL_GO_RUN", "FIELD_GOAL")
MIN_LARGE_SAMPLE = 30

NEEDED = {
    "season", "split", "game_id", "action", "converted", "field_goal_result",
    "qtr", "game_seconds_remaining", "yardline_100", "ydstogo",
    "score_differential", "is_home",
}


def load_engine(path: Path = ENGINE) -> dict:
    """Isolate production global objects; this report does not change them."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing frozen engine: {path}")
    spec = importlib.util.spec_from_file_location("event_model_2025_audit", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load the frozen decision engine.")
    module = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(module)
    return vars(module)


def fg_label(row: pd.Series) -> int | None:
    """Use the existing frozen 2025 benchmark's label semantics."""
    value = str(row.get("field_goal_result", "")).lower()
    if "made" in value:
        return 1
    if "miss" in value or "block" in value:
        return 0
    for key in ("fg_made", "field_goal_made"):
        if key in row and pd.notna(row[key]):
            return int(bool(row[key]))
    if str(row.get("execution_status", "")).upper() == "BROKEN":
        return 0
    return None


def to_preplay_state(row: pd.Series) -> dict:
    """Before-action information only. No realized score, result or gain."""
    required = [
        "qtr", "game_seconds_remaining", "yardline_100", "ydstogo",
        "score_differential", "is_home",
    ]
    for column in required:
        if column not in row or not pd.notna(row[column]):
            raise ValueError(f"Invalid preplay state column: {column}")
        if not np.isfinite(float(row[column])):
            raise ValueError(f"Nonfinite preplay state column: {column}")
    q = int(row["qtr"])
    if q < 1 or q > 4:
        raise ValueError("Regulation only; OT is excluded.")
    is_home = int(row["is_home"])
    neutral = int(row.get("is_neutral_site", 0) or 0)
    if is_home not in (0, 1) or neutral not in (0, 1):
        raise ValueError("Invalid home/neutral flag.")
    site = "NEUTRAL" if neutral else ("HOME" if is_home else "AWAY")
    state = {
        "qtr": q,
        "game_seconds_remaining": float(row["game_seconds_remaining"]),
        "yardline_100": float(row["yardline_100"]),
        "ydstogo": float(row["ydstogo"]),
        "score_differential": float(row["score_differential"]),
        "posteam_timeouts_remaining": numeric_default(
            row.get("posteam_timeouts_remaining"), 3
        ),
        "defteam_timeouts_remaining": numeric_default(
            row.get("defteam_timeouts_remaining"), 3
        ),
        "site": site,
        "roof": str(row.get("roof", "outdoors") or "outdoors"),
    }
    if "goal_to_go" in row and pd.notna(row["goal_to_go"]):
        state["goal_to_go"] = float(row["goal_to_go"])
    return state


def numeric_default(value, fallback: float) -> float:
    try:
        result = float(value)
        return result if np.isfinite(result) else float(fallback)
    except (TypeError, ValueError):
        return float(fallback)


def validate_historical_rows(dataset: pd.DataFrame) -> pd.DataFrame:
    missing = NEEDED.difference(dataset.columns)
    if missing:
        raise ValueError(f"Missing historical model fields: {sorted(missing)}")
    selected = dataset.loc[
        dataset["season"].isin(SEASONS)
        & dataset["action"].isin(ACTIONS)
    ].copy()
    if selected.empty:
        raise ValueError("No completed-era historical actions available.")
    expected = np.where(
        selected["season"].eq(2025), "test", "validation",
    )
    if not np.array_equal(selected["split"].to_numpy(str), expected):
        raise RuntimeError("Chronological train/validation/test split mismatch.")
    if not set(selected["season"].unique()).issubset(SEASONS):
        raise RuntimeError("Unexpected seasons in event-model diagnostic.")
    if selected["game_id"].isna().any():
        raise RuntimeError("Missing game IDs prevent game-cluster uncertainty.")
    return selected.reset_index(drop=True)


def enrich_roof(rows: pd.DataFrame) -> pd.DataFrame:
    """Roof is observable pre-kickoff and must not be inferred from labels."""
    pbp = nfl.load_pbp(list(SEASONS))
    if "roof" not in pbp.columns:
        raise RuntimeError("NFL PBP roof metadata not available.")
    meta = (pbp.select(["game_id", "roof"]).to_pandas()
            .dropna(subset=["roof"])
            .drop_duplicates(subset=["game_id"])
            .set_index("game_id")["roof"])
    copied = rows.copy()
    copied["roof"] = copied["game_id"].map(meta).fillna("outdoors")
    return copied


def predict_observed_events(rows: pd.DataFrame, engine: dict) -> tuple[pd.DataFrame, dict]:
    """No unchosen action is predicted, and labels never enter model features."""
    records = []
    reasons: dict[str, int] = {}
    for _, row in rows.iterrows():
        try:
            action = str(row["action"])
            if action == "FIELD_GOAL":
                actual = fg_label(row)
            else:
                actual = (int(float(row["converted"]))
                          if pd.notna(row["converted"]) else None)
            if actual is None or actual not in (0, 1):
                reasons["missing/invalid actual outcome"] = (
                    reasons.get("missing/invalid actual outcome", 0)+1
                )
                continue
            state = engine["normalize_state"](to_preplay_state(row))
            predicted = (
                float(engine["field_goal_probabilities"](state)["made"])
                if action == "FIELD_GOAL"
                else float(engine["go_conversion_probability"](state, action))
            )
            if not np.isfinite(predicted) or not 0 <= predicted <= 1:
                raise ValueError("Model prediction is nonfinite or not a probability.")
        except (ValueError, KeyError, TypeError, OverflowError) as exc:
            kind = f"invalid preplay state: {type(exc).__name__}"
            reasons[kind] = reasons.get(kind, 0)+1
            continue
        records.append({
            "game_id": str(row["game_id"]),
            "season": int(row["season"]),
            "action": action,
            "qtr": int(state["qtr"]),
            "seconds": float(state["game_seconds_remaining"]),
            "yardline": float(state["yardline_100"]),
            "ydstogo": float(state["ydstogo"]),
            "fg_distance": float(state["yardline_100"])+18,
            "site": str(state["site"]),
            "actual": actual,
            "predicted": predicted,
        })
    if not records:
        raise RuntimeError("No valid observed outcome predictions.")
    table = pd.DataFrame.from_records(records)
    if table["game_id"].isna().any() or not table["predicted"].between(0, 1).all():
        raise RuntimeError("Event-model predictions are invalid.")
    return table, reasons


def _bootstrap_gap(y: np.ndarray, p: np.ndarray, games: np.ndarray,
                   seed: int = 69, draws: int = 1600) -> tuple[float, float]:
    """95% game-cluster interval for actual-minus-predicted probability."""
    if len(y) != len(p) or len(y) != len(games) or len(y) == 0 or draws < 100:
        raise ValueError("Bootstrap inputs must align and be nonempty.")
    ids, inverse = np.unique(games.astype(str), return_inverse=True)
    delta = np.asarray(y, dtype=float)-np.asarray(p, dtype=float)
    sums = np.bincount(inverse, weights=delta, minlength=len(ids))
    counts = np.bincount(inverse, minlength=len(ids))
    rng = np.random.default_rng(seed)
    chunks = rng.integers(0, len(ids), size=(draws, len(ids)))
    mean = sums[chunks].sum(axis=1)/counts[chunks].sum(axis=1)
    return tuple(float(x) for x in np.quantile(mean, [.025, .975]))


def summarize_subgroup(name: str, subset: pd.DataFrame) -> dict:
    n = len(subset)
    result: dict = {"group": name, "n": n, "games": int(subset["game_id"].nunique())}
    if not n:
        result["status"] = "NO DATA"
        return result
    y = subset["actual"].to_numpy(float)
    p = subset["predicted"].to_numpy(float)
    lo, hi = _bootstrap_gap(y, p, subset["game_id"].to_numpy())
    eps = 1e-9
    pred = np.clip(p, eps, 1-eps)
    result.update({
        "actual": float(y.mean()), "predicted": float(p.mean()),
        "gap_actual_minus_predicted": float((y-p).mean()),
        "gap_ci95": [lo, hi],
        "brier": float(np.mean((y-p)**2)),
        "logloss": float(np.mean(-y*np.log(pred)-(1-y)*np.log(1-pred))),
        "status": ("LOW SUPPORT" if n < MIN_LARGE_SAMPLE else
                   ("GAP CI EXCLUDES ZERO" if lo > 0 or hi < 0
                    else "INCONCLUSIVE GAP")),
    })
    return result


def report_predeclared_groups(frame: pd.DataFrame) -> list[dict]:
    """Prespecified descriptive groups; no subgroup-driven parameter fitting."""
    q4 = frame["qtr"].eq(4)
    late = q4 & frame["seconds"].le(300)
    pass_go = frame["action"].eq("NORMAL_GO_PASS")
    run_go = frame["action"].eq("NORMAL_GO_RUN")
    fg = frame["action"].eq("FIELD_GOAL")
    near_goal = frame["yardline"].between(1, 10)
    at_four = frame["ydstogo"].between(3.5, 4.5)
    masks = [
        ("ALL GO PASS", pass_go),
        ("GO PASS Q4 LAST FIVE", pass_go & late),
        ("GO PASS FOURTH AND FOUR", pass_go & at_four),
        ("GO PASS OPPONENT 1-10", pass_go & near_goal),
        ("GO PASS 4TH&4 OPP 1-10", pass_go & at_four & near_goal),
        ("ALL GO RUN", run_go),
        ("GO RUN Q4 LAST FIVE", run_go & late),
        ("GO RUN SHORT <=3", run_go & frame["ydstogo"].le(3)),
        ("ALL FIELD GOALS", fg),
        ("FG Q4 LAST FIVE", fg & late),
        ("FG 18-29 YARDS", fg & frame["fg_distance"].between(18, 29)),
        ("FG 30-39 YARDS", fg & frame["fg_distance"].between(30, 39)),
        ("FG 40-49 YARDS", fg & frame["fg_distance"].between(40, 49)),
        ("FG 50+ YARDS", fg & frame["fg_distance"].ge(50)),
        ("FG 18-29 Q4 LAST FIVE", fg & late
         & frame["fg_distance"].between(18, 29)),
    ]
    return [summarize_subgroup(name, frame.loc[mask]) for name, mask in masks]


def main() -> None:
    os.chdir(ROOT)
    print("=== STEP 69: OBSERVED EVENT-MODEL CALIBRATION ===", flush=True)
    print("2023/24 DEVELOPMENT and 2025 EXPLORATORY; no 2026 outcomes.")
    print("Only the historical CHOSEN action is scored; NOT policy/counterfactual validation.")
    source = pd.read_parquet(DATA)
    data = enrich_roof(validate_historical_rows(source))
    engine = load_engine()
    predicted, invalid = predict_observed_events(data, engine)
    print(f"Eligible input rows {len(data):,}; scored {len(predicted):,}; "
          f"excluded {sum(invalid.values()):,}. Exclusions: {invalid}")
    if len(predicted) < len(data)*.90:
        raise RuntimeError("More than 10% of event states were unscorable.")
    report = {"source": "frozen historical fourth_down_modeling_split.parquet",
              "period": "2023-24 development; 2025 previously inspected",
              "models": "unchanged production component models",
              "counterfactual_validity": "NOT ESTABLISHED",
              "no_2026_used": True,
              "eligibility": {
                  "input_rows": len(data), "scored": len(predicted),
                  "excluded": invalid,
              },
              "groups": {}}
    for label, years in [
        ("2023-24 DEVELOPMENT", (2023, 2024)),
        ("2025 EXPLORATORY", (2025,)),
    ]:
        sub = predicted.loc[predicted["season"].isin(years)]
        groups = report_predeclared_groups(sub)
        report["groups"][label] = groups
        print(f"\n== {label}: {len(sub):,} actual selected fourth downs ==")
        for item in groups:
            if item["n"] == 0:
                print(f"  {item['group']:<31} n=0")
                continue
            c = item["gap_ci95"]
            print(f"  {item['group']:<31} n={item['n']:>5} "
                  f"games={item['games']:>3} "
                  f"actual={item['actual']:.1%} predicted={item['predicted']:.1%} "
                  f"bias={item['gap_actual_minus_predicted']:+.1%} "
                  f"95%game=[{c[0]:+.1%},{c[1]:+.1%}] "
                  f"Brier={item['brier']:.4f} {item['status']}")
    REPORT.parent.mkdir(exist_ok=True, parents=True)
    REPORT.write_text(json.dumps(report, indent=2))
    print(f"\nREPORT: {REPORT.relative_to(ROOT)}", flush=True)
    print("NO AUTOMATIC PROMOTION. These are observed-action calibration diagnostics only.")


if __name__ == "__main__":
    main()
