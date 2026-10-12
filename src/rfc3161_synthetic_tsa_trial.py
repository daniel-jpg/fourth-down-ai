"""Offline-first RFC 3161 trial for FABRICATED fourth-down forecast evidence.

This does NOT enable a prospective NFL witness. It can verify a detached
RFC 3161 token against an explicitly configured trust anchor and exact
synthetic commitment. It cannot establish independent snap timestamps,
source correctness, provider authorization, or real-game policy validity.

No live NFL/network access is performed unless the operator explicitly opts
into the separate public Sigstore TSA test. No 2026/2027 football data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Any
from urllib.request import Request, urlopen

POLICY_COMMIT = "31176d50af25db9dd42dae07e9b17cb5c2b930b4"
SYNTHETIC_DOMAIN = "FOURTH_DOWN_2027_FABRICATED_TIMESTAMP_TRIAL_V1"
PUBLIC_TEST_TSA = "https://timestamp.sigstore.dev/api/v1/timestamp"
MAX_QUERY_BYTES = 8192
MAX_RESPONSE_BYTES = 262144
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


def _require_sha256(value: Any) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise ValueError("Synthetic commitment requires lowercase SHA-256 hex")
    return value


def commitment_bytes(synthetic_source_packet_sha256: str) -> bytes:
    """Bind a synthetic packet hash to a non-reusable dry-run domain."""
    commitment = {
        "domain": SYNTHETIC_DOMAIN,
        "mode": "synthetic_only",
        "policy_commit": POLICY_COMMIT,
        "synthetic_source_packet_sha256": _require_sha256(
            synthetic_source_packet_sha256
        ),
    }
    return json.dumps(
        commitment, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False,
    ).encode("ascii")


def _openssl(*args: str, cwd: Path | None = None) -> str:
    binary = shutil.which("openssl")
    if binary is None:
        raise RuntimeError("OpenSSL CLI is required for RFC 3161 validation")
    try:
        result = subprocess.run(
            [binary, "ts", *args], cwd=cwd, capture_output=True,
            text=True, check=False, timeout=15,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("OpenSSL RFC 3161 command exceeded time limit") from exc
    if result.returncode:
        # Do not disclose response bytes or any source payload.
        raise ValueError("OpenSSL RFC 3161 validation failed")
    return result.stdout


def make_query(commitment: bytes) -> bytes:
    """OpenSSL creates SHA-256 RFC3161 query with random nonce and certReq."""
    if (not isinstance(commitment, bytes) or not commitment
            or len(commitment) > MAX_QUERY_BYTES):
        raise ValueError("Synthetic commitment must be small, nonempty bytes")
    with tempfile.TemporaryDirectory(prefix="fourthdown-tsa-") as tmp:
        work = Path(tmp)
        (work / "commitment.bin").write_bytes(commitment)
        _openssl(
            "-query", "-data", "commitment.bin", "-sha256", "-cert",
            "-out", "query.tsq", cwd=work,
        )
        raw = (work / "query.tsq").read_bytes()
    if not 10 <= len(raw) <= MAX_QUERY_BYTES:
        raise ValueError("Invalid RFC 3161 request size")
    return raw


def verify_response(
    *, commitment: bytes, query: bytes, response: bytes,
    trust_anchor_pem: bytes, intermediate_chain_pem: bytes,
) -> dict[str, Any]:
    """Require BOTH query/nonce AND original-commitment verification.

    A caller-provided trust root does NOT prove independent TSA custody.
    Never set a real-snap or prospective readiness flag here.
    """
    try:
        candidate = json.loads(commitment) if isinstance(commitment, bytes) else None
        source_hash = candidate["synthetic_source_packet_sha256"]
        canonical = commitment_bytes(source_hash)
    except (TypeError, KeyError, ValueError) as exc:
        raise ValueError("Only canonical fabricated forecast commitments allowed") from exc
    if commitment != canonical:
        raise ValueError("Only canonical fabricated forecast commitments allowed")
    if (not isinstance(query, bytes) or not 10 <= len(query) <= MAX_QUERY_BYTES
            or not isinstance(response, bytes)
            or not 10 <= len(response) <= MAX_RESPONSE_BYTES):
        raise ValueError("Invalid RFC 3161 request or response length")
    for label, value in (
        ("trust anchor", trust_anchor_pem),
        ("intermediate chain", intermediate_chain_pem),
    ):
        if (not isinstance(value, bytes)
                or b"-----BEGIN CERTIFICATE-----" not in value
                or len(value) > MAX_RESPONSE_BYTES):
            raise ValueError(f"Explicit PEM {label} required; no TOFU")
    with tempfile.TemporaryDirectory(prefix="fourthdown-tsa-verify-") as tmp:
        work = Path(tmp)
        for name, data in (
            ("commitment.bin", commitment),
            ("query.tsq", query), ("reply.tsr", response),
            ("root.pem", trust_anchor_pem),
            ("chain.pem", intermediate_chain_pem),
        ):
            (work / name).write_bytes(data)
        query_detail = _openssl(
            "-query", "-in", "query.tsq", "-text", cwd=work,
        )
        if ("Hash Algorithm: sha256" not in query_detail
                or "Certificate required: yes" not in query_detail
                or not re.search(r"^Nonce: 0x[0-9A-Fa-f]+$",
                                 query_detail, re.MULTILINE)):
            raise ValueError("Require SHA-256, embedded nonce and TSA signer certificate")
        common = (
            "-in", "reply.tsr", "-CAfile", "root.pem",
            "-untrusted", "chain.pem",
        )
        # Query verifies nonce and imprint associated with the exact request.
        _openssl("-verify", "-queryfile", "query.tsq", *common, cwd=work)
        # Separate -data verification binds the response to our canonical
        # synthetic commitment, not just an attacker's swapped query.
        _openssl("-verify", "-data", "commitment.bin", *common, cwd=work)
        text = _openssl("-reply", "-in", "reply.tsr", "-text", cwd=work)
    stamped = next(
        (line.split(":", 1)[1].strip()
         for line in text.splitlines() if line.startswith("Time stamp:")),
        None,
    )
    if not stamped:
        raise ValueError("Verified token did not expose timestamp")
    return {
        "status": "SYNTHETIC_RFC3161_TOKEN_VERIFIED_WITH_CALLER_TRUST",
        "commitment_sha256": hashlib.sha256(commitment).hexdigest(),
        "timestamp_response_sha256": hashlib.sha256(response).hexdigest(),
        "token_time_display": stamped,
        "nonce_and_commitment_imprint_verified": True,
        "signature_and_supplied_trust_chain_verified": True,
        "trust_anchor_externally_approved": False,
        "independent_snap_timestamp_verified": False,
        "real_2027_prospective_collection_approved": False,
        "source_type": "fabricated_only",
    }


def request_public_test_timestamp(
    query: bytes, *, allow_public_network_test: bool = False,
    tsa_url: str = PUBLIC_TEST_TSA,
) -> bytes:
    """Explicit, single opt-in POST of a fabricated SHA-256 imprint only.

    Public Sigstore availability, usage rights and rate limits must be
    reviewed separately. No API credentials, NFL games or future rows.
    """
    if not allow_public_network_test:
        raise PermissionError("Network timestamp test requires explicit opt-in")
    if tsa_url != PUBLIC_TEST_TSA:
        raise ValueError("Only exact documented HTTPS Sigstore test endpoint allowed")
    if not isinstance(query, bytes) or not 10 <= len(query) <= MAX_QUERY_BYTES:
        raise ValueError("Refusing invalid timestamp query")
    request = Request(
        tsa_url, data=query, method="POST", headers={
            "Content-Type": "application/timestamp-query",
            "Accept": "application/timestamp-reply",
        },
    )
    with urlopen(request, timeout=6) as response:
        if response.status not in (200, 201):
            raise ValueError("Public TSA returned non-success status")
        payload = response.read(MAX_RESPONSE_BYTES + 1)
    if not 10 <= len(payload) <= MAX_RESPONSE_BYTES:
        raise ValueError("Public timestamp response missing or oversized")
    return payload


def _write_new(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as fd:
        fd.write(content)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Synthetic-only fourth-down RFC3161 trial; never a live NFL witness"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("create", help="Build fake commitment and nonce query")
    build.add_argument("--synthetic-sha256", required=True)
    build.add_argument("--commitment-out", type=Path, required=True)
    build.add_argument("--query-out", type=Path, required=True)
    ask = commands.add_parser("request", help="OPTIONAL opt-in public TSA request")
    ask.add_argument("--query", type=Path, required=True)
    ask.add_argument("--response-out", type=Path, required=True)
    ask.add_argument("--allow-public-network-test", action="store_true")
    check = commands.add_parser("verify", help="Offline trust-anchor verification")
    for name in ("commitment", "query", "response", "trust-anchor", "chain"):
        check.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.command == "create":
        commitment = commitment_bytes(args.synthetic_sha256)
        query = make_query(commitment)
        _write_new(args.commitment_out, commitment)
        _write_new(args.query_out, query)
        print("FABRICATED RFC3161 query generated; no outbound request", flush=True)
    elif args.command == "request":
        token = request_public_test_timestamp(
            args.query.read_bytes(),
            allow_public_network_test=args.allow_public_network_test,
        )
        _write_new(args.response_out, token)
        print("Public TSA replied; NOT YET VERIFIED / NOT PROSPECTIVE", flush=True)
    else:
        result = verify_response(
            commitment=args.commitment.read_bytes(),
            query=args.query.read_bytes(),
            response=args.response.read_bytes(),
            trust_anchor_pem=args.trust_anchor.read_bytes(),
            intermediate_chain_pem=args.chain.read_bytes(),
        )
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
