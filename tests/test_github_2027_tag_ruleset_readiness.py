"""No-network tests for strict GitHub 2027 release-tag ruleset preflight."""
from __future__ import annotations

import json

import pytest

from src import github_2027_tag_ruleset_readiness as gate
from scripts import audit_2027_tag_rulesets as cli


def protecting_rule(**changes):
    proposal = gate.proposed_ruleset()
    proposal["id"] = 42
    proposal.update(changes)
    return proposal


def test_no_rulesets_fails_closed_and_cannot_unlock_future_season():
    report = gate.evaluate_ruleset_details([])
    assert report["status"] == "NO_GO_NO_GITHUB_RULESETS"
    assert not report["non_bypassable_update_and_deletion_rules_found"]
    assert not report["real_2027_holdout_evaluation_authorized"]
    assert not report["one_time_production_tag_created"]


def test_exact_active_no_bypass_update_and_deletion_rules_detected():
    r = gate.evaluate_ruleset_details([protecting_rule()])
    assert r["status"] == "TAG_GUARDS_VERIFIED_BUT_FUTURE_RELEASE_DISABLED"
    assert r["non_bypassable_update_and_deletion_rules_found"]
    assert r["unknown_bypass_count"] == 0
    assert r["bypass_present_count"] == 0
    assert r["creation_is_appropriately_restricted"] is False
    assert r["real_2027_holdout_evaluation_authorized"] is False


def test_public_get_missing_bypass_actors_never_passes():
    item = protecting_rule()
    del item["bypass_actors"]
    r = gate.evaluate_ruleset_details([item])
    assert r["status"] == "NO_GO_UNVERIFIED_TAG_UPDATE_AND_DELETION_GUARDS"
    assert r["unknown_bypass_count"] == 1
    assert not r["non_bypassable_update_and_deletion_rules_found"]


def test_privileged_bypass_blocks_claim_protection():
    item = protecting_rule(bypass_actors=[
        {"actor_id": 1, "actor_type": "RepositoryRole", "bypass_mode": "always"}
    ])
    r = gate.evaluate_ruleset_details([item])
    assert r["bypass_present_count"] == 1
    assert not r["non_bypassable_update_and_deletion_rules_found"]


@pytest.mark.parametrize("change", [
    {"enforcement": "disabled"},
    {"enforcement": "evaluate"},
    {"target": "branch"},
    {"conditions": {"ref_name": {
        "include": ["refs/tags/irrelevant"], "exclude": [],
    }}},
    {"conditions": {"ref_name": {
        "include": [gate.REAL_RELEASE_TAG],
        "exclude": [gate.REAL_RELEASE_TAG],
    }}},
    {"conditions": {"ref_name": {
        "include": [gate.REAL_RELEASE_TAG],
        "exclude": ["refs/tags/fourth-down-ai/2027*"],
    }}},
    {"rules": [{"type": "creation"}]},
    {"rules": [{"type": "update"}]},
    {"rules": [{"type": "deletion"}]},
])
def test_unsupported_bypassed_or_partial_rules_never_certify(change):
    r = gate.evaluate_ruleset_details([protecting_rule(**change)])
    assert not r["non_bypassable_update_and_deletion_rules_found"]
    assert r["status"] == "NO_GO_UNVERIFIED_TAG_UPDATE_AND_DELETION_GUARDS"


def test_two_separate_no_bypass_rulesets_may_cover_both_rules():
    one = protecting_rule(
        id=11, rules=[{"type": "update"}],
    )
    two = protecting_rule(
        id=22, rules=[{"type": "deletion"}],
    )
    r = gate.evaluate_ruleset_details([one, two])
    assert r["non_bypassable_update_and_deletion_rules_found"]
    assert r["status"] == "TAG_GUARDS_VERIFIED_BUT_FUTURE_RELEASE_DISABLED"


def test_tag_pattern_support_is_conservative():
    matching = protecting_rule(conditions={"ref_name": {
        "include": ["refs/tags/fourth-down-ai/*"], "exclude": [],
    }})
    assert gate.evaluate_ruleset_details(
        [matching]
    )["non_bypassable_update_and_deletion_rules_found"]
    ambiguous = protecting_rule(conditions={"ref_name": {
        "include": ["refs/tags/**/*"], "exclude": [],
    }})
    assert not gate.evaluate_ruleset_details(
        [ambiguous]
    )["non_bypassable_update_and_deletion_rules_found"]


def test_proposal_cannot_claim_it_was_installed_or_enable_release(tmp_path):
    original = gate.proposed_ruleset()
    assert original["target"] == "tag"
    assert original["enforcement"] == "active"
    assert original["conditions"]["ref_name"]["include"] == [
        gate.REAL_RELEASE_TAG
    ]
    assert set(r["type"] for r in original["rules"]) == {
        "update", "deletion",
    }
    assert original["bypass_actors"] == []
    path = tmp_path / "proposed.json"
    details = [original]
    path.write_text(json.dumps(details))
    # Offline fixture is just a sample, never a real GitHub authorization.
    audited = gate.evaluate_ruleset_details(details)
    assert audited["real_2027_holdout_evaluation_authorized"] is False
    assert not audited["creation_is_appropriately_restricted"]


def test_public_readonly_query_with_no_rules_is_safe_and_no_future_api(monkeypatch):
    visited = []
    class Response:
        status = 200
        def __enter__(self):
            return self
        def __exit__(self, *ignored):
            return None
        def read(self, n):
            return b"[]"

    def fake_urlopen(request, timeout):
        visited.append(request.full_url)
        assert timeout == 8
        return Response()
    monkeypatch.setattr(cli, "urlopen", fake_urlopen)
    result = cli.read_public_repo_ruleset_details()
    assert result == []
    assert visited == [
        f"{cli.API_BASE}?per_page=100&includes_parents=true"
    ]
    assert gate.evaluate_ruleset_details(result)["status"] == (
        "NO_GO_NO_GITHUB_RULESETS"
    )


def test_public_ruleset_detail_get_never_assumes_bypass(monkeypatch):
    urls = []
    details = protecting_rule()
    del details["bypass_actors"]
    class Response:
        status = 200
        def __init__(self, data):
            self.data = data
        def __enter__(self):
            return self
        def __exit__(self, *ignored):
            return None
        def read(self, n):
            return json.dumps(self.data).encode()

    def fake_urlopen(request, timeout):
        url = request.full_url
        urls.append(url)
        if url.endswith("per_page=100&includes_parents=true"):
            return Response([{"id": 42, "name": "the tag guard"}])
        return Response(details)
    monkeypatch.setattr(cli, "urlopen", fake_urlopen)
    found = cli.read_public_repo_ruleset_details()
    assert len(found) == 1
    assert urls[1] == f"{cli.API_BASE}/42"
    assert not gate.evaluate_ruleset_details(
        found
    )["non_bypassable_update_and_deletion_rules_found"]


@pytest.mark.parametrize("url", [
    "https://not-github.example/rulesets",
    "http://api.github.com/repos/daniel-jpg/fourth-down-ai/rulesets/42",
    "https://api.github.com/repos/daniel-jpg/fourth-down-ai/rulesets/42/history",
    "https://api.github.com/repos/daniel-jpg/fourth-down-ai/rulesets/42?token=secret",
])
def test_public_rule_fetch_rejects_untrusted_url_before_network(url, monkeypatch):
    monkeypatch.setattr(
        cli, "urlopen",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("Unapproved network request")
        ),
    )
    with pytest.raises(ValueError, match="Only fixed public"):
        cli._public_json(url)
