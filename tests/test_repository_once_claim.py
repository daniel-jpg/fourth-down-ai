"""Two independent local Git clients compete for one durable tag claim.

No test contacts GitHub or uses real NFL/2027 football data.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path
import subprocess

import pytest

from src import repository_once_claim as lock


def git(*args, cwd=None, check=True):
    proc = subprocess.run(
        ["git", *map(str, args)], cwd=cwd,
        capture_output=True, text=True, check=False, timeout=15,
    )
    if check:
        assert proc.returncode == 0, proc.stderr
    return proc


@pytest.fixture()
def worktrees(tmp_path):
    remote = tmp_path / "shared-bare.git"
    seed = tmp_path / "seed"
    git("init", "--bare", remote)
    git("init", "-b", "main", seed)
    git("-C", seed, "config", "user.name", "Synthetic Test")
    git("-C", seed, "config", "user.email", "fake@localhost.invalid")
    (seed / "README.md").write_text("Only a fake 2027 replay.\n")
    git("-C", seed, "add", "README.md")
    git("-C", seed, "commit", "-m", "Empty future-study fixture")
    git("-C", seed, "remote", "add", "origin", remote)
    git("-C", seed, "push", "-u", "origin", "main")
    git("-C", remote, "symbolic-ref", "HEAD", "refs/heads/main")
    a, b = tmp_path / "runner-A", tmp_path / "runner-B"
    git("clone", remote, a)
    git("clone", remote, b)
    return a, b, remote


def fake_hash():
    return hashlib.sha256(b"NO REAL NFL GAME OR PRODUCTION PREDICTION").hexdigest()


def test_tag_claim_survives_independent_runner_and_rejects_second(worktrees):
    a, b, remote = worktrees
    first = lock.claim_fabricated_remote_once(
        a, mode=lock.FAKE_MODE, fake_forecast_sha256=fake_hash(),
    )
    assert first["status"] == "FABRICATED_REMOTE_ONCE_CLAIM_CREATED"
    assert first["claim_ref"] == lock.FULL_REF
    assert first["remote"] == "LOCAL_BARE_GIT_ONLY"
    assert first["real_2027_evaluation_authorized"] is False
    assert first["git_tag_delete_or_force_protection_configured"] is False
    assert len(first["claim_sha256"]) == 64
    assert git(
        "-C", remote, "rev-parse", lock.FULL_REF
    ).stdout.strip() == first["tag_object"]
    message = git("-C", remote, "cat-file", "-p", first["tag_object"]).stdout
    assert fake_hash() in message
    assert '"real_2027_evaluation_authorized":false' in message
    with pytest.raises(FileExistsError, match="already exists"):
        lock.claim_fabricated_remote_once(
            b, mode=lock.FAKE_MODE, fake_forecast_sha256=fake_hash(),
        )
    with pytest.raises(FileExistsError, match="already exists"):
        lock.claim_fabricated_remote_once(
            a, mode=lock.FAKE_MODE, fake_forecast_sha256=fake_hash(),
        )


def test_simultaneous_local_runners_cannot_both_claim(worktrees):
    a, b, remote = worktrees
    def attempt(worktree):
        try:
            lock.claim_fabricated_remote_once(
                worktree, mode=lock.FAKE_MODE,
                fake_forecast_sha256=fake_hash(),
            )
            return "won"
        except FileExistsError:
            return "blocked"
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(attempt, place) for place in (a, b)]
        results = [f.result() for f in futures]
    assert sorted(results) == ["blocked", "won"]
    assert len(git(
        "-C", remote, "show-ref", "--tags"
    ).stdout.splitlines()) == 1


@pytest.mark.parametrize("mode", [
    "real_2027", "approved", "synthetic", True, None,
])
def test_real_future_claim_mode_rejected_before_any_repository_access(mode):
    class Inaccessible:
        def __getattribute__(self, attr):
            raise AssertionError("Inspected repo even though mode forbidden!")
    with pytest.raises(PermissionError, match="Real 2027 release"):
        lock.claim_fabricated_remote_once(
            Inaccessible(), mode=mode, fake_forecast_sha256=fake_hash(),
        )


def test_network_github_remote_refused_without_contact(worktrees):
    a, _, _ = worktrees
    git("-C", a, "remote", "set-url", "origin",
        "https://github.com/fake/fourth-down-ai.git")
    with pytest.raises(ValueError, match="Refusing network"):
        lock.claim_fabricated_remote_once(
            a, mode=lock.FAKE_MODE, fake_forecast_sha256=fake_hash(),
        )


def test_malformed_digest_refused_before_any_git_operation(worktrees):
    a, _, _ = worktrees
    with pytest.raises(ValueError, match="SHA-256"):
        lock.claim_fabricated_remote_once(
            a, mode=lock.FAKE_MODE, fake_forecast_sha256="2027",
        )


def test_readonly_protocol_documents_admin_tag_deletion_limit():
    from inspect import getsource
    source = getsource(lock.claim_fabricated_remote_once)
    assert "tag-protection" in source
    assert "nonforce" in source
    assert "real_2027_evaluation_authorized" in source

def test_remote_claim_precedes_any_postgame_scores_and_survives_runner_restart(
    worktrees, tmp_path,
):
    """End-to-end synthetic replay: two different runner output dirs, one tag."""
    from scripts.rehearse_2027_postseason_once import fabricate_whole_season
    from src.postseason_2027_admission import MODE
    from src.postseason_2027_once_only import run_fabricated_postseason_once

    a, b, remote = worktrees
    first_output = tmp_path / "first-runner-output"
    first = fabricate_whole_season()

    class AssertAlreadyClaimed(list):
        def __iter__(self):
            assert (first_output / "forecast_seal.json").exists()
            assert (first_output / "remote_claim_receipt.json").exists()
            assert git("-C", remote, "rev-parse", lock.FULL_REF).returncode == 0
            return super().__iter__()

    first["pbp_games"] = AssertAlreadyClaimed(first["pbp_games"])
    first["reference_games"] = AssertAlreadyClaimed(first["reference_games"])
    first["observed_plays"] = AssertAlreadyClaimed(first["observed_plays"])

    def claim_from_clone(clone):
        return lambda digest: lock.claim_fabricated_remote_once(
            clone, mode=lock.FAKE_MODE, fake_forecast_sha256=digest,
        )

    completed = run_fabricated_postseason_once(
        first, first_output, mode=MODE, pre_outcome_claim=claim_from_clone(a),
    )
    assert completed["status"] == "SYNTHETIC_POSTSEASON_REHEARSAL_ONLY"
    assert completed["remote_claim_scope"] == "LOCAL_BARE_GIT_FABRICATED_ONLY"
    receipt = __import__("json").loads(
        (first_output / "remote_claim_receipt.json").read_text()
    )
    assert completed["remote_claim_sha256"] == receipt["claim_sha256"]
    assert (first_output / "done.json").is_file()
    second_output = tmp_path / "second-runner-new-output"
    second = fabricate_whole_season()

    class NoOutcomeRead(list):
        def __iter__(self):
            raise AssertionError("Second runner inspected game outcomes!")
    second["pbp_games"] = NoOutcomeRead(second["pbp_games"])
    second["reference_games"] = NoOutcomeRead(second["reference_games"])
    second["observed_plays"] = NoOutcomeRead(second["observed_plays"])
    with pytest.raises(FileExistsError, match="already exists"):
        run_fabricated_postseason_once(
            second, second_output, mode=MODE,
            pre_outcome_claim=claim_from_clone(b),
        )
    assert (second_output / "forecast_seal.json").exists()
    assert (second_output / "failed_after_seal.txt").exists()
    assert not (second_output / "synthetic_evaluation.json").exists()
    assert not (second_output / "remote_claim_receipt.json").exists()


def test_replay_does_not_read_labels_when_claim_receipt_is_untrusted(
    worktrees, tmp_path,
):
    from scripts.rehearse_2027_postseason_once import fabricate_whole_season
    from src.postseason_2027_admission import MODE
    from src.postseason_2027_once_only import run_fabricated_postseason_once

    data = fabricate_whole_season()
    class NoOutcomeRead(list):
        def __iter__(self):
            raise AssertionError("Bad claim must stop before outcome inspection")
    data["pbp_games"] = NoOutcomeRead(data["pbp_games"])
    data["reference_games"] = NoOutcomeRead(data["reference_games"])
    data["observed_plays"] = NoOutcomeRead(data["observed_plays"])
    output = tmp_path / "fraudulent-claim"
    with pytest.raises(RuntimeError, match="verified local fabricated claim"):
        run_fabricated_postseason_once(
            data, output, mode=MODE,
            pre_outcome_claim=lambda sha: {
                "status": "FABRICATED_REMOTE_ONCE_CLAIM_CREATED",
                "remote": "REAL_GITHUB",  # must not be accepted
                "real_2027_evaluation_authorized": True,
            },
        )
    assert (output / "forecast_seal.json").is_file()
    assert (output / "failed_after_seal.txt").is_file()
    assert not (output / "synthetic_evaluation.json").exists()
