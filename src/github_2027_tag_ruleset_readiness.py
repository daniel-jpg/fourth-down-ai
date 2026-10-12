"""READ-ONLY GitHub tag protection preflight; never a release unlock.

A successful no-force Git push creates a shared once-only ref, but does
not prevent an admin from deleting or replacing the ref. A *separate*
repository ruleset must enforce update+deletion restrictions.

This module only audits complete GitHub ruleset detail objects for:
  refs/tags/fourth-down-ai/2027-postseason-once
It never changes rulesets, tags, GitHub settings, production predictions,
or real 2027 NFL records. An unauthenticated ruleset response often omits
bypass_actors. Such a response MUST NOT be mistaken for no bypass.

Even a PASS of these structural checks is not authorization to begin the
2027 retrospective study. Future post-season sources, release approvals
and one-time score controls remain blocked until separately implemented.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence

OWNER = "daniel-jpg"
REPOSITORY = "fourth-down-ai"
REAL_RELEASE_TAG = "refs/tags/fourth-down-ai/2027-postseason-once"
PROTECTION_TEMPLATE_NAME = "Fourth-down 2027 immutable-once claim guard"
RULES_REQUIRED = frozenset({"update", "deletion"})

# We conservatively understand only exact and one known prefix-matching
# pattern. GitHub's full fnmatch grammar has platform-specific edge cases;
# never assert that an ambiguous pattern definitely protects a ref.
KNOWN_MATCHING_INCLUDES = frozenset({
    REAL_RELEASE_TAG,
    "refs/tags/fourth-down-ai/*",
    "refs/tags/fourth-down-ai/**",
    "~ALL",
})
KNOWN_EXCLUSIONS = frozenset({
    REAL_RELEASE_TAG,
    "refs/tags/fourth-down-ai/*",
    "refs/tags/fourth-down-ai/**",
    "~ALL",
})


def proposed_ruleset() -> dict[str, Any]:
    """Config shape for admin review, NOT auto-submitted to GitHub."""
    return {
        "name": PROTECTION_TEMPLATE_NAME,
        "target": "tag",
        "enforcement": "active",
        "bypass_actors": [],
        "conditions": {
            "ref_name": {
                "include": [REAL_RELEASE_TAG],
                "exclude": [],
            },
        },
        "rules": [
            {"type": "update"},
            {"type": "deletion"},
        ],
    }


def _candidate_detail(item: Any) -> tuple[bool, str]:
    if not isinstance(item, dict):
        return False, "INVALID_RULESET_DETAIL"
    if item.get("target") != "tag":
        return False, "NOT_A_TAG_RULESET"
    if item.get("enforcement") != "active":
        return False, "NOT_ACTIVE"
    conditions = item.get("conditions")
    if not isinstance(conditions, dict) or not isinstance(
        conditions.get("ref_name"), dict
    ):
        return False, "MISSING_TARGET_REF_CONDITIONS"
    refs = conditions["ref_name"]
    includes = refs.get("include", [])
    excludes = refs.get("exclude", [])
    if not isinstance(includes, list) or not isinstance(excludes, list):
        return False, "INVALID_REF_CONDITIONS"
    if not any(isinstance(x, str) and x in KNOWN_MATCHING_INCLUDES
               for x in includes):
        return False, "NO_PROVEN_REF_MATCH"
    if any(isinstance(x, str) and x in KNOWN_EXCLUSIONS
           for x in excludes):
        return False, "TARGET_EXCLUDED"
    # Unknown exclusion expressions could still match the target;
    # do not turn an incomplete pattern audit into a release approval.
    if excludes:
        return False, "UNREVIEWED_EXCLUSION_PATTERN"
    return True, "MATCHES_TARGET_REF"


def evaluate_ruleset_details(details: Any) -> dict[str, Any]:
    """No network, no dynamic rule changes, no 2027 release authorization."""
    if not isinstance(details, list) or len(details) > 100:
        raise ValueError("Expected <=100 full GitHub ruleset detail objects")
    report = []
    effective = set()
    ambiguous = []
    bypass_unknown = []
    bypass_enabled = []
    for index, entry in enumerate(details):
        okay, reason = _candidate_detail(entry)
        ident = (
            entry.get("id") if isinstance(entry, dict) and
            type(entry.get("id")) is int else None
        )
        name = entry.get("name") if isinstance(entry, dict) else None
        row = {
            "id": ident, "name": name,
            "applies_to_release_ref": okay, "reason": reason,
            "rules_that_could_protect": [],
            "bypass_visibility_verified": False,
            "bypass_roles_detected": None,
        }
        if okay:
            rules = entry.get("rules")
            if not isinstance(rules, list):
                row["reason"] = "RULE_DETAIL_NOT_RETURNED"
                ambiguous.append(index)
            else:
                applied = {
                    r.get("type") for r in rules
                    if isinstance(r, dict) and r.get("type") in RULES_REQUIRED
                }
                row["rules_that_could_protect"] = sorted(applied)
                if "bypass_actors" not in entry:
                    # GitHub can omit bypass data for read-only callers.
                    row["reason"] = "BYPASS_ACTORS_NOT_VISIBLE"
                    bypass_unknown.append(index)
                else:
                    actors = entry["bypass_actors"]
                    if not isinstance(actors, list):
                        row["reason"] = "INVALID_BYPASS_ACTORS"
                        ambiguous.append(index)
                    elif actors:
                        row["reason"] = "PRIVILEGED_BYPASS_PRESENT"
                        row["bypass_roles_detected"] = len(actors)
                        row["bypass_visibility_verified"] = True
                        bypass_enabled.append(index)
                    else:
                        row["bypass_visibility_verified"] = True
                        row["bypass_roles_detected"] = 0
                        effective.update(applied)
        report.append(row)
    update_and_delete = RULES_REQUIRED.issubset(effective)
    if not details:
        status = "NO_GO_NO_GITHUB_RULESETS"
    elif not update_and_delete:
        status = "NO_GO_UNVERIFIED_TAG_UPDATE_AND_DELETION_GUARDS"
    else:
        status = "TAG_GUARDS_VERIFIED_BUT_FUTURE_RELEASE_DISABLED"
    # NEVER transform this into actual 2027 release approval.
    return {
        "status": status,
        "repository": f"{OWNER}/{REPOSITORY}",
        "target_tag": REAL_RELEASE_TAG,
        "inspected_ruleset_count": len(report),
        "ruleset_details": report,
        "non_bypassable_update_and_deletion_rules_found": update_and_delete,
        "unknown_bypass_count": len(bypass_unknown),
        "bypass_present_count": len(bypass_enabled),
        "ambiguous_ruleset_count": len(ambiguous),
        "creation_is_appropriately_restricted": False,
        "ruleset_cannot_be_changed_by_repo_admin": False,
        "genuine_2027_source_verified": False,
        "real_2027_holdout_evaluation_authorized": False,
        "one_time_production_tag_created": False,
        "recommendation": (
            "A repository administrator must review and install active "
            "no-bypass UPDATE and DELETION tag rules on this exact ref. "
            "Separately restrict tag creation to the approved future release "
            "identity, and audit admin ruleset changes. Do not authorize "
            "real 2027 scoring until early 2028 source/finals approval."
        ),
    }


def summary_sha256(details: Sequence[Mapping[str, Any]]) -> str:
    payload = json.dumps(
        details, sort_keys=True, separators=(",", ":"),
        allow_nan=False, ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()
