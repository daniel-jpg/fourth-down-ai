"""Step 83: one-time native kickoff challenger AFTER the frozen Step 66 gate.

This companion is designed to run automatically ONCE only after the original,
unchanged 2026 W6-18 Step 66 forward validation has published its result at
>=100 distinct eligible games. Its only purpose is to score the PRE-EXISTING
2023-25 research candidate (native normal-35 kickoff inputs) against the frozen
synthetic kickoff baseline, including HOME/AWAY kicker subgroups.

DO NOT run or fetch any 2026 outcomes before the original Step 66 result
exists. Unit tests use entirely synthetic mock 2026 rows; no real 2026 NFL
data is needed to develop or test this companion.

Primary: paired Brier gain = Brier(synthetic) - Brier(native), positive better.
Uncertainty: percentile 95% GAME-cluster bootstrap, 2000 draws, seed 83.
Secondary: paired log-loss gain; site- and clock-specific descriptive scores.
The explicit research screen requires overall lower CI > 0, at least 20 games
per kicking-team site, and nonnegative gains in BOTH home and away-kicking
subgroups. This screen NEVER authorizes promoting a production model.
No 2025 donors or 2026 outcomes are used for fitting or choosing thresholds.

The cohort is the SAME selected 2026 W6-18 ordinary Q4 kickoff/first-snap
cohort as Step 66. That means it EXCLUDES penalty-affected, onside, return-TD,
unmatched, and score-changing kicks; it is not the broad 2023-25 cohort.
It is a precommitted observational forecast reliability comparison, not
a causal fourth-down GO-vs-FG evaluation.
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import runpy
import sys

import nflreadpy as nfl
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ORIGINAL_GATE_PATH = ROOT / "reports/2026_forward_kickoff_wp_final.txt"
CHALLENGER_REPORT_PATH = ROOT / "reports/2026_native_kickoff_challenger_final.json"
STEP66_PATH = ROOT / "src/66_forward_2026_continuation_wp.py"
STEP61_PATH = ROOT / "src/61_audit_observed_kickoff_wp.py"

# Independently freeze the existing one-time evaluation, input helper and
# native encoding before prospective outcomes are scored.
PINNED_SOURCE_GIT_BLOBS = {
    "src/66_forward_2026_continuation_wp.py":
        "c77bce490c69d96d52fb91068d9852920cb4790d",
    "src/61_audit_observed_kickoff_wp.py":
        "98fb57981b4504007eee1bbf52339b5dc968e200",
    "src/kickoff_q4_native_features_candidate.py":
        "03bca375f042b24fb51099050cc7b3fd9ef796a7",
}
MIN_COMPLETED_GAMES = 100
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 83
MIN_SITE_GAMES = 20
PROB_EPS = 1e-12
ALLOWED_WEEKS = (6, 18)
SCORE_COLUMNS = (
    "game_id", "week", "seconds", "kicker_home",
    "receiver_margin", "actual_home_win", "synthetic_wp", "native_wp",
)


def load_fixed_step66():
    spec = importlib.util.spec_from_file_location("frozen_forward_step66", STEP66_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Frozen 2026 gating code is unavailable.")
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


def check_existing_gate_report(
    path: Path = ORIGINAL_GATE_PATH, completed: Path = CHALLENGER_REPORT_PATH
) -> str:
    """Fail closed BEFORE making any real 2026 data request."""
    if completed.exists():
        raise RuntimeError("Native challenger already evaluated; run ONCE only.")
    if not path.is_file():
        raise RuntimeError("Original frozen 2026 gate not yet complete: DO NOT score.")
    data = path.read_text()
    if (
        data.count("PREDECLARED RESEARCH DECISION:") != 1
        or "evaluation_commit=" not in data
        or "=== STEP 66: FROZEN 2026 W6-18 FORWARD VALIDATION ===" not in data
        or "NOT READY: need >=" in data
        or (
            "RESEARCH GATE PASS" not in data
            and "RESEARCH GATE FAIL/INCONCLUSIVE" not in data
        )
    ):
        raise RuntimeError("Step 66 report is incomplete or unverified.")
    return data


def enforce_frozen_provenance(step66, root: Path = ROOT) -> None:
    """Do not change any of the already observed/historical model inputs."""
    step66.enforce_frozen_artifacts(root)
    for relative, expected in PINNED_SOURCE_GIT_BLOBS.items():
        file = root / relative
        if not file.is_file() or step66.git_blob_sha(file) != expected:
            raise RuntimeError(
                f"Prospective challenger source changed: {relative}. "
                "Do not recalibrate or alter the postregistered code."
            )


def eligible_pairs_only(pbp: pd.DataFrame, step66, features: list[str]) -> pd.DataFrame:
    """Check sample availability BEFORE scoring; no early reliability peeking."""
    pairs = step66.prospective_2026_pairs(pbp, features)
    games = 0 if pairs.empty else int(pairs["game_id"].nunique())
    if games < MIN_COMPLETED_GAMES:
        raise RuntimeError(
            f"Prospective native challenger NOT READY: {games} eligible games "
            f"(requires at least {MIN_COMPLETED_GAMES}). No forecasts computed."
        )
    if (
        pairs["week"].lt(ALLOWED_WEEKS[0]).any()
        or pairs["week"].gt(ALLOWED_WEEKS[1]).any()
        or pairs[["game_id", "play_id_kick"]].duplicated().any()
    ):
        raise RuntimeError("Prospective kickoff cohort not frozen or uniquely joined.")
    return pairs


def paired_native_forecasts(
    pairs: pd.DataFrame, model, features: list[str], helper61
) -> pd.DataFrame:
    """Predict at kickoff; next-snap data and eventual result are LABEL-only.

    Identical to Step 66's native_pending / engine_pending feature vectors,
    using no post-kickoff position or elapsed-time outcome for inference.
    """
    games = int(pairs["game_id"].nunique())
    if games < MIN_COMPLETED_GAMES:
        raise RuntimeError("Gate must have at least 100 games before forecasts.")
    native = helper61["feature_frame"](pairs, "kick")
    if (
        not native["qtr"].eq(4).all()
        or not native["is_kickoff"].eq(1).all()
        or not native["yardline_100"].eq(35).all()
        or not native["ydstogo"].eq(0).all()
        or native["down"].notna().any()
        or not native["goal_to_go"].eq(0).all()
        or not native["game_seconds_remaining"].gt(40).all()
    ):
        raise ValueError("Unexpected 2026 native kickoff input encoding.")
    synthetic = native.copy()
    for name in ("down", "ydstogo", "yardline_100"):
        synthetic[name] = np.nan
    synthetic["goal_to_go"] = 0.0
    predict = helper61["model_probabilities"]
    old = np.asarray(predict(model, features, synthetic), dtype=float)
    candidate = np.asarray(predict(model, features, native), dtype=float)
    final_margin = pd.to_numeric(
        pairs["result_kick"], errors="coerce"
    ).to_numpy(dtype=float)
    if (
        not np.isfinite(final_margin).all()
        or np.any(final_margin == 0)
        or len(candidate) != len(pairs)
        or not all(np.isfinite(v).all() for v in (old, candidate))
        or not all(((0 <= v) & (v <= 1)).all() for v in (old, candidate))
    ):
        raise ValueError("No incomplete, tied, or invalid prospective outcomes.")
    receiver_home = native["is_home_posteam"].to_numpy(dtype=bool)
    home_margin = native["home_score_differential"].to_numpy(dtype=float)
    result = pd.DataFrame({
        "game_id": pairs["game_id"].astype(str).to_numpy(),
        "week": pairs["week"].to_numpy(int),
        "seconds": native["game_seconds_remaining"].to_numpy(dtype=float),
        "kicker_home": ~receiver_home,
        "receiver_margin": np.where(receiver_home, home_margin, -home_margin),
        "actual_home_win": (final_margin > 0).astype(float),
        "synthetic_wp": old,
        "native_wp": candidate,
    }, columns=SCORE_COLUMNS)
    if (
        len(result) != len(pairs)
        or result["game_id"].nunique() < MIN_COMPLETED_GAMES
        or result.isna().any().any()
    ):
        raise RuntimeError("Invalid post-gate prospective kickoff forecast rows.")
    return result


def game_cluster_gain_ci(
    paired_brier_gain: np.ndarray, game_ids: np.ndarray,
    *, draws: int = BOOTSTRAP_DRAWS, seed: int = BOOTSTRAP_SEED,
) -> list[float]:
    """Resample whole GAMES so multi-kickoff games remain together."""
    gain = np.asarray(paired_brier_gain, dtype=float)
    games = np.asarray(game_ids)
    if (
        gain.ndim != 1 or gain.size == 0 or gain.shape != games.shape
        or not np.isfinite(gain).all() or draws < 100
    ):
        raise ValueError("Game bootstrap needs paired finite gain and game IDs.")
    unique, inverse = np.unique(games, return_inverse=True)
    totals = np.bincount(inverse, weights=gain, minlength=len(unique))
    counts = np.bincount(inverse, minlength=len(unique))
    rng = np.random.default_rng(seed)
    drawn = rng.integers(0, len(unique), size=(draws, len(unique)))
    samples = totals[drawn].sum(axis=1) / counts[drawn].sum(axis=1)
    return [float(v) for v in np.quantile(samples, [0.025, 0.975])]


def paired_summary(group: pd.DataFrame, *, label: str) -> dict:
    n = int(len(group))
    n_games = int(group["game_id"].nunique())
    out = {"label": label, "kickoffs": n, "games": n_games}
    if n == 0:
        out["status"] = "NO OBSERVATIONS"
        return out
    y = group["actual_home_win"].to_numpy(dtype=float)
    old = group["synthetic_wp"].to_numpy(dtype=float)
    candidate = group["native_wp"].to_numpy(dtype=float)
    if not all(
        np.isfinite(x).all() and ((0 <= x) & (x <= 1)).all()
        for x in (y, old, candidate)
    ):
        raise ValueError("Invalid prospective score or outcome.")
    gain = (y-old)**2 - (y-candidate)**2
    def logloss(p: np.ndarray) -> float:
        c = np.clip(p, PROB_EPS, 1 - PROB_EPS)
        return float(np.mean(-y*np.log(c)-(1-y)*np.log1p(-c)))
    out.update({
        "status": "LOW SUPPORT" if n < 30 or n_games < MIN_SITE_GAMES else "DESCRIPTIVE",
        "actual_home_win_share": float(y.mean()),
        "synthetic_mean_wp": float(old.mean()),
        "native_mean_wp": float(candidate.mean()),
        "synthetic_brier": float(((y-old)**2).mean()),
        "native_brier": float(((y-candidate)**2).mean()),
        "native_brier_gain_positive_better": float(gain.mean()),
        "synthetic_logloss": logloss(old),
        "native_logloss": logloss(candidate),
        "native_logloss_gain_positive_better": float(
            logloss(old) - logloss(candidate)
        ),
        "native_mean_abs_change_pp": float(np.abs(candidate-old).mean()*100),
        "gain_ci95_game_cluster": game_cluster_gain_ci(
            gain, group["game_id"].to_numpy()
        ),
    })
    return out


def summarize_once(scored: pd.DataFrame) -> dict:
    if (
        len(scored) < MIN_COMPLETED_GAMES
        or scored["game_id"].nunique() < MIN_COMPLETED_GAMES
        or set(SCORE_COLUMNS) != set(scored.columns)
        or scored.isna().any().any()
    ):
        raise RuntimeError("Cannot summarize an ungated or incomplete cohort.")
    groups = {
        "overall": scored,
        "kicker_home": scored.loc[scored["kicker_home"]],
        "kicker_away": scored.loc[~scored["kicker_home"]],
        "last_five_gt40": scored.loc[scored["seconds"].le(300)],
        "q4_5_to_15_minutes": scored.loc[scored["seconds"].gt(300)],
    }
    report = {name: paired_summary(frame, label=name) for name, frame in groups.items()}
    overall = report["overall"]
    home, away = report["kicker_home"], report["kicker_away"]
    enough_site_games = (
        home["games"] >= MIN_SITE_GAMES and away["games"] >= MIN_SITE_GAMES
    )
    both_sites_nonnegative = (
        "native_brier_gain_positive_better" in home
        and "native_brier_gain_positive_better" in away
        and home["native_brier_gain_positive_better"] >= 0
        and away["native_brier_gain_positive_better"] >= 0
    )
    lower = overall["gain_ci95_game_cluster"][0]
    # Precommit the criterion before observing 2026 scores.
    preliminary = bool(lower > 0 and enough_site_games and both_sites_nonnegative)
    return {
        "sample_kickoffs": int(len(scored)),
        "sample_games": int(scored["game_id"].nunique()),
        "paired_results": report,
        "research_screen": {
            "overall_lower_bootstrap_bound_gt_zero": bool(lower > 0),
            "kicker_home_and_away_each_at_least_20_games": bool(enough_site_games),
            "both_site_point_brier_gains_nonnegative": bool(both_sites_nonnegative),
            "preliminary_reliability_support": preliminary,
            "promotion_to_production_authorized": False,
            "decision": (
                "PRELIMINARY SUPPORT ONLY — NO PRODUCTION PROMOTION"
                if preliminary
                else "INCONCLUSIVE/NEGATIVE — NO PRODUCTION PROMOTION"
            ),
        },
    }


def run() -> None:
    """Called by a locked once-only workflow, never by scheduled readiness."""
    os.chdir(ROOT)
    check_existing_gate_report()
    step66 = load_fixed_step66()
    enforce_frozen_provenance(step66)
    print("Step 66 one-time report verified; frozen inputs and source pinned.",
          flush=True)
    # Read actual 2026 only AFTER the original prospective gate has
    # already completed. No separate earlier data searches or peeking.
    pbp26 = nfl.load_pbp([2026]).sort(["game_id", "play_id"]).to_pandas()
    helpers = runpy.run_path(str(STEP61_PATH))
    pairs = eligible_pairs_only(pbp26, step66, helpers["FIELDS"])
    engine = runpy.run_path(str(ROOT / "src/42_build_decision_engine.py"))
    rows = paired_native_forecasts(
        pairs, engine["wp_model"], list(engine["WP_FEATURES"]), helpers
    )
    findings = summarize_once(rows)
    print("=== STEP 83: ONE-TIME POST-GATE NATIVE INPUT CHALLENGER ===",
          flush=True)
    for name, group in findings["paired_results"].items():
        if "native_brier_gain_positive_better" not in group:
            print(f"{name}: NO SUPPORT", flush=True)
            continue
        print(
            f"{name:20} n={group['kickoffs']}, games={group['games']} "
            f"synthetic={group['synthetic_brier']:.5f} "
            f"native={group['native_brier']:.5f} "
            f"paired_gain={group['native_brier_gain_positive_better']:+.5f} "
            f"cluster_CI={group['gain_ci95_game_cluster']}",
            flush=True,
        )
    report = {
        "report_name": "PRECOMMITTED_2026_NATIVE_INPUT_CHALLENGER_ONCE",
        "evaluation_year": 2026,
        "first_week": 6,
        "last_week": 18,
        "eligible_only_after_original_step66_one_time_report": True,
        "minimum_original_gate_completed_games": MIN_COMPLETED_GAMES,
        "cohort": "Same Step 66 ordinary, matched Q4 own-35 kickoffs >40s",
        "nonordinary_scoring_returns_included": False,
        "scored_once_after_original_gate": True,
        "retrained_wp_or_candidate_parameters": False,
        "previously_inspected_2025_used_for_tuning": False,
        "whole_game_bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "site_minimum_games": MIN_SITE_GAMES,
        "pinned_sources_git_blobs": PINNED_SOURCE_GIT_BLOBS,
        "pinned_wp_model_git_blob": step66.PINNED_GIT_BLOBS,
        "metrics": findings,
        "limitations": (
            "Observational ordinary Q4 kickoffs selected by actual first-snap "
            "availability; cannot certify special kickoffs, full made-FG "
            "action WP, causal GO-vs-FG choice, or team-specific calibration. "
            "CI is sampling uncertainty only, not model-selection uncertainty."
        ),
        "production_action": "KEEP FROZEN — NO AUTOMATIC PROMOTION",
    }
    CHALLENGER_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    # This path is the one-time marker. Workflow commits it only after
    # all tests pass; a rerun must fail/skip when it exists.
    CHALLENGER_REPORT_PATH.write_text(json.dumps(report, indent=2)+"\n")
    print("Published read-only challenger report for one-time commit.",
          flush=True)
    print("OFFICIAL FOURTH-DOWN ENGINE AND STREAMLIT NEVER MODIFIED.",
          flush=True)


if __name__ == "__main__":
    run()
