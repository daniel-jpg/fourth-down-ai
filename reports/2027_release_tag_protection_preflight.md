# 2027 Fourth-Down AI: tag-protection preflight and post-seal shared claim

**As of October 12, 2026.** The project uses existing NFLverse/nflreadpy PBP and independently archived ESPN final-game scores. It does not depend on Sportradar, vendor email, subscription, live snap timestamps or model retraining.

**Current GitHub state:** A read-only GitHub API query of \`GET /repos/daniel-jpg/fourth-down-ai/rulesets\` on October 12, 2026 returned **an empty list**. Therefore **no GitHub ruleset protects an eventual real 2027 study tag yet**. The connected GitHub write tools support code/files, not creating or changing repository rulesets. Neither the model nor this project has installed an active tag ruleset or created a genuine release claim.

## Code implemented now, all explicitly synthetic / offline

1. [Postseason one-time evaluator](../src/postseason_2027_once_only.py) now accepts an **optional fabricated remote claim callback**. It validates a label-free forecast bundle and writes an exclusive local \`forecast_seal.json\`, then calls the shared remote tag claim with the sealed forecast SHA256 **before accessing the ESPN-shaped final scores or observed coach actions**. It accepts only the local-bare-remote synthetic claim receipt, writes \`remote_claim_receipt.json\`, and fails/marks the output directory forever if the remote claim is already used or invalid. With no callback, legacy fully fabricated local evaluation still works; it records \`remote_claim_scope=NOT_REQUESTED\`.
2. [Git once-only claim](../src/repository_once_claim.py) remains a non-force annotated tag to a **local bare Git remote only**. The new [tests](../tests/test_repository_once_claim.py) confirm two independent clones sharing a remote cannot each score the synthetic season, *even when they use different local output directories*. The second runner is refused before reading artificial outcomes. No live GitHub push is performed.
3. [Tag ruleset preflight](../src/github_2027_tag_ruleset_readiness.py) checks **fully detailed, active tag rulesets** covering the exact future ref \`refs/tags/fourth-down-ai/2027-postseason-once\`. To assert the ref is protected, it demands both \`update\` and \`deletion\` restrictions with **explicitly observed empty bypass lists** (possibly from different active rulesets); uncertain wildcard or exclusions fail closed. Unauthenticated public GitHub ruleset details can omit the bypass list: an **absent field is unknown, NOT a proven empty list**. No combination of reported protections authorizes real 2027 evaluation.
4. [Read-only preflight CLI](../scripts/audit_2027_tag_rulesets.py) supports an offline JSON fixture, the bundled ruleset proposal, or an explicit public GitHub API read (no network request by default). It never writes settings, creates tags or accesses any NFL games.
5. [Admin-reviewed draft ruleset JSON](2027_tag_protection_ruleset_proposal.json) proposes active \`update\` + \`deletion\` protection and **no bypass** for the precise future tag only. This is not an installed rule. **Creation restriction is intentionally deferred** until a dedicated, reviewed release identity is available; otherwise a creation restriction with no authorized bypass could permanently prevent the tag from ever being created.
6. [Focused read-only CI](../.github/workflows/2027_retro_existing_source_and_lock.yml) runs the historical 2025 NFLverse/independent ESPN source verification, simulated two-clone race and new ruleset-policy negative tests; checks the draft ruleset JSON against the generated proposal. GitHub workflow permissions are read-only, and nothing may push to a real GitHub release ref.

## How to inspect protections now

Run **without network**, printing only the proposal:

    PYTHONPATH=. python -m scripts.audit_2027_tag_rulesets --proposed-only

Run the **explicit public, read-only GitHub status inspection** (makes GitHub API GET requests, not POST):

    PYTHONPATH=. python -m scripts.audit_2027_tag_rulesets \
      --read-public-repo-rulesets --report-out artifacts/current_tag_readiness.json

The genuine repository currently has no rulesets, so the expected result is \`NO_GO_NO_GITHUB_RULESETS\`. If a new rule is installed, this read-only command may still return \`NO_GO_UNVERIFIED_TAG_UPDATE_AND_DELETION_GUARDS\` until an administrator reviews the complete bypass list; public APIs sometimes do not expose the actors who can bypass a ruleset. A fixture command with \`--rulesets-json\` is for test inputs, **not live installed GitHub verification**.

## One necessary admin action we cannot perform through current GitHub tools

Open **[GitHub repository rules settings](https://github.com/daniel-jpg/fourth-down-ai/settings/rules)**, then:

1. Create a **new tag ruleset**. Name: *Fourth-down 2027 immutable-once claim guard*. Scope the target to exactly \`refs/tags/fourth-down-ai/2027-postseason-once\` (or the equivalent tag-name pattern in the GitHub UI).
2. Set enforcement to **Active** and select **Restrict updates** and **Restrict deletions**. Do not give broad repository-admin, organization-admin or arbitrary app accounts an "always bypass" exemption for these two protections.
3. Leave **Restrict creations** out of this particular ruleset. Prior to the actual post-season release, create a **separate** creation rule that permits only a dedicated, approved release identity to create that tag; **no** bypass for deletion/update protections. Apply and independently review the resulting GitHub rulesets.
4. Verify the installed rules are visible in Settings, and later independently review the actual effective active rulesets and bypass list with proper admin access. Repository administrators may still modify/remove the ruleset itself; this scheme is a strong operational guard, not mathematical immutability.

**Do not create the real 2027 evaluation tag now, do not turn on future-data collector permissions, and do not run future-season scoring.** Those actions require real 2027 season completion (early 2028), complete NFLverse/ESPN archive provenance, reviewer authorization and a separately implemented scored-run release workflow.

GitHub ruleset API/documentation: [creating tag rulesets](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-rulesets/creating-rulesets-for-a-repository), [available restrictions](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-rulesets/available-rules-for-rulesets), [read-only ruleset API](https://docs.github.com/en/rest/repos/rules).

## Current explicit barriers

| Check | Current evidence | Outcome |
| --- | --- | --- |
| Existing 2025 NFLverse / independent ESPN source | 272 2025 games, 3,151 verified historical final margins | PASS (historical only) |
| Two separate synthetic runner clones racing for same ref | Shared local bare remote, non-force tag, two attempted claims | PASS (fake remote only) |
| Forecast seal created before label access and shared remote claim | Synthetic-only post-seal callback and denial of second run | TESTED (fake football only) |
| GitHub real release tag update/deletion ruleset | Repo rulesets currently **empty** | **NO-GO** |
| Real 2027 independent source, complete-season gate and tag creation | Future season not complete; release not authorized | **NO-GO** |
| Real 2027 retrospective scoring | Permanently disabled in current code | **NO-GO** |

This strengthens the infrastructure now while preserving the user-selected **nflverse-only future-season retrospective reconstruction** method. Neither a real before-snap prediction timestamp nor a causal coaching advantage is claimed.
