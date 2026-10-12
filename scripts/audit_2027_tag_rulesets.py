"""Read-only public GitHub ruleset audit for a future 2027 claim tag.

Examples (does NOT write GitHub rulesets or future NFL data):
  PYTHONPATH=. python -m scripts.audit_2027_tag_rulesets --proposed-only
  PYTHONPATH=. python -m scripts.audit_2027_tag_rulesets \
      --read-public-repo-rulesets --report-out artifacts/tag_readiness.json

The public GitHub API may HIDE bypass actors, so its response can prove a
missing protection but CANNOT certify no bypass without admin review.
No network at all unless --read-public-repo-rulesets is explicitly selected.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.request import Request, urlopen

from src.github_2027_tag_ruleset_readiness import (
    OWNER, REPOSITORY, evaluate_ruleset_details, proposed_ruleset,
    summary_sha256,
)

API_BASE = f"https://api.github.com/repos/{OWNER}/{REPOSITORY}/rulesets"
MAX_BYTES = 250_000


def _public_json(url: str) -> object:
    if url != f"{API_BASE}?per_page=100&includes_parents=true":
        if (not url.startswith(API_BASE + "/")
                or not url[len(API_BASE)+1:].isdecimal()):
            raise ValueError("Only fixed public GitHub ruleset endpoints allowed")
    request = Request(url, headers={
        "Accept": "application/vnd.github+json",
        "User-Agent": "fourth-down-ai-ruleset-audit-read-only",
        "X-GitHub-Api-Version": "2022-11-28",
    })
    with urlopen(request, timeout=8) as response:
        if response.status != 200:
            raise ValueError("GitHub ruleset read failed")
        raw = response.read(MAX_BYTES+1)
    if len(raw) > MAX_BYTES:
        raise ValueError("GitHub ruleset payload exceeds safety limit")
    return json.loads(raw)


def read_public_repo_ruleset_details() -> list[dict]:
    """GET only; complete details or fail. No assumption about bypass access."""
    summary = _public_json(f"{API_BASE}?per_page=100&includes_parents=true")
    if not isinstance(summary, list) or len(summary) >= 100:
        raise ValueError("GitHub ruleset list incomplete; cannot certify protection")
    ids = []
    for row in summary:
        if not isinstance(row, dict) or type(row.get("id")) is not int:
            raise ValueError("Malformed GitHub ruleset summary")
        ids.append(row["id"])
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate GitHub ruleset IDs")
    details = []
    for ruleset_id in ids:
        item = _public_json(f"{API_BASE}/{ruleset_id}")
        if not isinstance(item, dict) or item.get("id") != ruleset_id:
            raise ValueError("Ruleset detail does not match requested ID")
        details.append(item)
    return details


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    methods = parser.add_mutually_exclusive_group(required=True)
    methods.add_argument("--read-public-repo-rulesets", action="store_true")
    methods.add_argument("--rulesets-json", type=Path)
    methods.add_argument("--proposed-only", action="store_true")
    parser.add_argument("--report-out", type=Path)
    args = parser.parse_args()
    if args.proposed_only:
        packet = {
            "purpose": "PROPOSAL_ONLY_NOT_INSTALLED",
            "ruleset": proposed_ruleset(),
            "warning": (
                "Ruleset creation needs GitHub admin permissions. "
                "Tag creation authorization and post-2027 review separate."
            ),
            "real_2027_holdout_evaluation_authorized": False,
        }
    else:
        details = (
            read_public_repo_ruleset_details()
            if args.read_public_repo_rulesets else
            json.loads(args.rulesets_json.read_text(encoding="utf-8"))
        )
        packet = evaluate_ruleset_details(details)
        packet["ruleset_detail_snapshot_sha256"] = summary_sha256(details)
        packet["ruleset_protection_verified_in_github"] = (
            packet["non_bypassable_update_and_deletion_rules_found"]
            and args.read_public_repo_rulesets
        )
        # Even the trusted snapshot isn't enough: GitHub may omit bypass.
        packet["real_2027_holdout_evaluation_authorized"] = False
    encoded = json.dumps(packet, indent=2, sort_keys=True)+"\n"
    if args.report_out is not None:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        with args.report_out.open("x", encoding="utf-8") as out:
            out.write(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
