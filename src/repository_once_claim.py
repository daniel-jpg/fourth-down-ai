"""Repository-wide Git TAG claim lock, exercised ONLY on local fake remotes.

Why a tag? GitHub Actions runners lose their local files. A NONFORCE push of
a fixed annotated tag to the same remote is serialized by Git server ref
updates. Only the first claimant can create the ref; a second push is
rejected, including from independent clones. GitHub concurrency settings
alone do NOT make a durable historical once-only ledger.

SAFETY: This implementation allows ONLY local bare Git remotes and a
synthetic rehearsal ref. It NEVER pushes to github.com or creates the real
2027 evaluation tag. A separate reviewed 2028 release, external source
authentication, GitHub tag-protection policy and secure approvals are
required before any real holdout collection or one-time score.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any

FAKE_MODE = "fabricated_remote_once_only"
TAG = "fourth-down-ai/fabricated-2027-postseason-once"
FULL_REF = f"refs/tags/{TAG}"
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


def _git(where: Path, *args: str) -> str:
    try:
        proc = subprocess.run(
            ["git", "-C", str(where), *args],
            capture_output=True, text=True, timeout=15, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError("Git command failed or timed out") from exc
    if proc.returncode:
        raise RuntimeError("Git operation failed; no remote lock success")
    return proc.stdout.strip()


def require_synthetic_mode(mode: Any) -> None:
    if mode != FAKE_MODE:
        raise PermissionError(
            "Real 2027 release-claim ref creation is disabled. "
            "This module can only claim a synthetic test tag on a local "
            "bare Git repository; no network, GitHub or future outcome access."
        )


def _verify_local_bare_remote(repo: Path) -> Path:
    if not repo.is_dir() or repo.is_symlink():
        raise ValueError("A local Git worktree directory is required")
    remote_address = _git(repo, "remote", "get-url", "origin")
    if (not remote_address.startswith("/")
            or "://" in remote_address or "\n" in remote_address):
        raise ValueError("Refusing network or relative Git remote for test lock")
    remote = Path(remote_address)
    if (not remote.is_absolute() or not remote.is_dir()
            or remote.is_symlink() or remote.resolve() != remote):
        raise ValueError("Remote must be a direct, local bare Git repository")
    if _git(remote, "rev-parse", "--is-bare-repository") != "true":
        raise ValueError("Remote must be a local bare Git repository")
    return remote


def claim_fabricated_remote_once(
    repository: Path, *, mode: Any, fake_forecast_sha256: str,
) -> dict[str, Any]:
    """Claim a fixed server-side tag in a LOCAL bare remote exactly once.

    The fixed tag cannot be recreated by a second normal (nonforce) push.
    Treat tag deletion/forced replacement by a privileged repo administrator
    as an operational threat requiring tag-protection and independent audit
    during actual 2028 release design.
    """
    require_synthetic_mode(mode)  # BEFORE repository access or git subprocess
    if (not isinstance(fake_forecast_sha256, str)
            or not SHA256_RE.fullmatch(fake_forecast_sha256)):
        raise ValueError("Claim requires a lowercase SHA-256 fabricated forecast digest")
    if not isinstance(repository, Path):
        raise TypeError("Repository must be a local Path")
    root = repository.resolve()
    remote = _verify_local_bare_remote(root)
    head = _git(root, "rev-parse", "--verify", "HEAD")
    if not re.fullmatch(r"[0-9a-f]{40,64}", head):
        raise RuntimeError("Invalid source repository HEAD hash")

    # Empty means never claimed. This initial check is only advisory:
    # a second runner can race and create the tag a moment later.
    # The atomic server-side non-force ref create is the definitive lock.
    if _git(root, "ls-remote", "--refs", "origin", FULL_REF):
        raise FileExistsError("Synthetic remote evaluation claim already exists")

    metadata = {
        "mode": FAKE_MODE, "fake_forecast_sha256": fake_forecast_sha256,
        "source_git_commit": head, "real_2027_evaluation_authorized": False,
        "future_outcomes_accessed": False,
    }
    text = json.dumps(metadata, sort_keys=True, separators=(",", ":"))
    try:
        # Git tagger metadata is for a LOCAL synthetic test, never a trusted
        # timestamp or independent release authorization.
        _git(
            root, "-c", "user.name=FourthDownAIFakeLock",
            "-c", "user.email=fabricated@localhost.invalid",
            "tag", "-a", TAG, "-m", text, head,
        )
    except RuntimeError as exc:
        raise FileExistsError("Local synthetic tag already exists") from exc

    tag_object = _git(root, "rev-parse", FULL_REF)
    try:
        _git(root, "push", "origin", f"{FULL_REF}:{FULL_REF}")
    except RuntimeError as exc:
        raise FileExistsError(
            "Remote Git tag already claimed (or tag creation rejected); "
            "no second synthetic evaluation permitted"
        ) from exc
    actual = _git(root, "ls-remote", "--refs", "origin", FULL_REF)
    if actual != f"{tag_object}\t{FULL_REF}":
        raise RuntimeError("Claim tag not independently confirmed on the remote")
    return {
        "status": "FABRICATED_REMOTE_ONCE_CLAIM_CREATED",
        "claim_ref": FULL_REF,
        "remote": "LOCAL_BARE_GIT_ONLY",
        "tag_object": tag_object,
        "claim_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "source_git_commit": head,
        "real_2027_evaluation_authorized": False,
        "independent_postseason_review_complete": False,
        "git_tag_delete_or_force_protection_configured": False,
    }
