"""Read-only release preflight. Does not load models or fetch NFL data."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = (
    "README.md", "CHANGELOG.md", "LICENSE", "app.py",
    "requirements.txt", "requirements-dev.txt",
    "reports/88_research_demo_release_readiness.md",
    "reports/90_release_candidate_checklist.md",
    "tests/test_streamlit_app_integration.py",
    ".github/workflows/tests.yml",
    ".github/workflows/forward_2026_validation_once.yml",
    ".github/workflows/forward_2026_native_challenger_once.yml",
)
FROZEN_BLOBS = {
    "models/win_probability_model.joblib": "4cdafd1687594ff624560ad140725d9c32d72a17",
    "data/win_probability_spec.json": "857ef77f7ca0a041502ed28811fdcbf48a5d6e95",
    "src/83_forward_2026_native_challenger_once.py": "90ea87140de3aa3bd442bc5546cb549bdea6f0cf",
}


def check() -> list[str]:
    errors = []
    for path in (*REQUIRED, *FROZEN_BLOBS):
        if not (ROOT / path).is_file():
            errors.append(f"Missing required file: {path}")

    for path, expected in FROZEN_BLOBS.items():
        if not (ROOT / path).is_file():
            continue
        try:
            actual = subprocess.check_output(
                ["git", "hash-object", "--", path],
                cwd=ROOT, text=True, stderr=subprocess.STDOUT,
            ).strip()
        except (OSError, subprocess.CalledProcessError) as exc:
            errors.append(f"Cannot hash {path}: {exc}")
            continue
        if actual != expected:
            errors.append(f"Frozen blob mismatch: {path}: {actual} != {expected}")

    readme = (ROOT / "README.md").read_text(encoding="utf-8") if (ROOT / "README.md").is_file() else ""
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8") if (ROOT / "CHANGELOG.md").is_file() else ""
    checklist = (ROOT / "reports/90_release_candidate_checklist.md")
    checklist_text = checklist.read_text(encoding="utf-8") if checklist.is_file() else ""
    for path, data, phrase in (
        ("README.md", readme, "reports/88_research_demo_release_readiness.md"),
        ("README.md", readme, "reports/90_release_candidate_checklist.md"),
        ("CHANGELOG.md", changelog, "## Unreleased"),
        ("CHANGELOG.md", changelog, "## v1.1.0"),
        ("reports/90_release_candidate_checklist.md", checklist_text, "Not verified"),
    ):
        if phrase not in data:
            errors.append(f"{path} is missing release marker: {phrase}")

    if "**Current stable release: v1.1.0**" not in readme:
        errors.append("README must preserve v1.1.0 as current stable release")
    if "## v1.2.0" in changelog:
        errors.append("Found a v1.2.0 release heading; review the release gate first")
    return errors


if __name__ == "__main__":
    issues = check()
    for issue in issues:
        print("RELEASE PREFLIGHT FAIL:", issue, file=sys.stderr)
    if issues:
        sys.exit(1)
    print("Release preflight passed: files, documentation markers, and frozen Git blobs.")
    print("This does NOT attest to current CI status, hosted deployed SHA, visual QA, or release approval.")
