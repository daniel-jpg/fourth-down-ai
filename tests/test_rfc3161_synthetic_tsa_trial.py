"""Offline cryptographic RFC3161 fixture; LOCAL self-signed CA IS NOT trust."""
import hashlib
from pathlib import Path
import subprocess

import pytest

from src.rfc3161_synthetic_tsa_trial import (
    commitment_bytes, make_query, verify_response,
    request_public_test_timestamp,
)


def openssl(*args, cwd):
    result = subprocess.run(
        ["openssl", *args], cwd=cwd,
        capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0, result.stderr


@pytest.fixture(scope="module")
def local_tsa(tmp_path_factory):
    base = tmp_path_factory.mktemp("fabricated-tsa")
    openssl(
        "req", "-new", "-x509", "-newkey", "rsa:2048", "-nodes",
        "-keyout", "root.key", "-out", "root.pem", "-days", "2",
        "-subj", "/CN=FABRICATED TEST ROOT CA",
        "-addext", "basicConstraints=critical,CA:TRUE",
        "-addext", "keyUsage=critical,keyCertSign,cRLSign", cwd=base,
    )
    openssl(
        "req", "-new", "-newkey", "rsa:2048", "-nodes",
        "-keyout", "tsa.key", "-out", "tsa.csr",
        "-subj", "/CN=FABRICATED TEST Timestamp Authority", cwd=base,
    )
    (base / "tsa.ext").write_text(
        "basicConstraints=critical,CA:FALSE\n"
        "keyUsage=critical,digitalSignature\n"
        "extendedKeyUsage=critical,timeStamping\n"
        "subjectKeyIdentifier=hash\n"
        "authorityKeyIdentifier=keyid,issuer\n"
    )
    openssl(
        "x509", "-req", "-in", "tsa.csr", "-CA", "root.pem",
        "-CAkey", "root.key", "-CAcreateserial", "-out", "tsa.pem",
        "-days", "2", "-sha256", "-extfile", "tsa.ext", cwd=base,
    )
    (base / "serial.txt").write_text("01\n")
    (base / "tsa.cnf").write_text(
        f"""[ tsa ]
default_tsa = fake
[ fake ]
serial = {base / 'serial.txt'}
signer_cert = {base / 'tsa.pem'}
certs = {base / 'root.pem'}
signer_key = {base / 'tsa.key'}
signer_digest = sha256
default_policy = 1.3.6.1.4.1.12123.1.1
digests = sha256
accuracy = secs:1
ordering = yes
tsa_name = yes
ess_cert_id_chain = yes
"""
    )
    return base


def issue_local_token(base, query):
    (base / "request.tsq").write_bytes(query)
    openssl(
        "ts", "-reply", "-config", "tsa.cnf", "-queryfile", "request.tsq",
        "-out", "response.tsr", cwd=base,
    )
    return (base / "response.tsr").read_bytes()


def test_fake_signed_timestamp_verifies_without_any_real_witness(local_tsa):
    source_hash = hashlib.sha256(b"FABRICATED fourth down source").hexdigest()
    commitment = commitment_bytes(source_hash)
    query = make_query(commitment)
    result = verify_response(
        commitment=commitment, query=query,
        response=issue_local_token(local_tsa, query),
        trust_anchor_pem=(local_tsa / "root.pem").read_bytes(),
        intermediate_chain_pem=(local_tsa / "tsa.pem").read_bytes(),
    )
    assert result["nonce_and_commitment_imprint_verified"] is True
    assert result["signature_and_supplied_trust_chain_verified"] is True
    assert result["trust_anchor_externally_approved"] is False
    assert result["independent_snap_timestamp_verified"] is False
    assert result["real_2027_prospective_collection_approved"] is False
    assert "GMT" in result["token_time_display"]


def test_mutation_and_swapped_query_are_rejected(local_tsa):
    commitment = commitment_bytes("0" * 64)
    query = make_query(commitment)
    reply = issue_local_token(local_tsa, query)
    fields = {
        "query": query, "response": reply,
        "trust_anchor_pem": (local_tsa / "root.pem").read_bytes(),
        "intermediate_chain_pem": (local_tsa / "tsa.pem").read_bytes(),
    }
    with pytest.raises(ValueError, match="canonical"):
        verify_response(commitment=commitment + b" ", **fields)
    with pytest.raises(ValueError, match="OpenSSL"):
        verify_response(commitment=commitment_bytes("1" * 64), **fields)
    with pytest.raises(ValueError, match="OpenSSL"):
        verify_response(
            commitment=commitment,
            **{**fields, "query": make_query(commitment)},
        )
    altered = bytearray(reply)
    altered[-15] ^= 1
    with pytest.raises(ValueError, match="OpenSSL"):
        verify_response(
            commitment=commitment, **{**fields, "response": bytes(altered)}
        )


def test_wrong_ca_fails_even_with_correct_tsa_response(local_tsa, tmp_path):
    commitment = commitment_bytes("f" * 64)
    query = make_query(commitment)
    response = issue_local_token(local_tsa, query)
    openssl(
        "req", "-x509", "-newkey", "rsa:2048", "-nodes",
        "-keyout", "wrong.key", "-out", "wrong.pem",
        "-days", "1", "-subj", "/CN=UNTRUSTED FAKE CA", cwd=tmp_path,
    )
    with pytest.raises(ValueError, match="OpenSSL"):
        verify_response(
            commitment=commitment, query=query, response=response,
            trust_anchor_pem=(tmp_path / "wrong.pem").read_bytes(),
            intermediate_chain_pem=(local_tsa / "tsa.pem").read_bytes(),
        )


def test_tsa_query_rejects_bad_digest_and_requires_network_opt_in():
    with pytest.raises(ValueError, match="SHA-256"):
        commitment_bytes("deadbeef")
    with pytest.raises(PermissionError, match="opt-in"):
        request_public_test_timestamp(b"abc")
    with pytest.raises(ValueError, match="Sigstore"):
        request_public_test_timestamp(
            b"x" * 100, allow_public_network_test=True,
            tsa_url="http://127.0.0.1:9999/timestamp",
        )


def test_no_future_data_network_or_workflow_auto_run():
    source = (
        Path(__file__).resolve().parents[1] / "src/rfc3161_synthetic_tsa_trial.py"
    ).read_text()
    assert "synthetic_only" in source
    assert "real_2027_prospective_collection_approved" in source

def test_nonce_less_timestamp_is_rejected_even_if_signature_is_valid(local_tsa):
    commitment = commitment_bytes("b" * 64)
    (local_tsa / "no-nonce-source.bin").write_bytes(commitment)
    openssl(
        "ts", "-query", "-sha256", "-cert", "-no_nonce",
        "-data", "no-nonce-source.bin", "-out", "without-nonce.tsq",
        cwd=local_tsa,
    )
    query = (local_tsa / "without-nonce.tsq").read_bytes()
    response = issue_local_token(local_tsa, query)
    with pytest.raises(ValueError, match="nonce"):
        verify_response(
            commitment=commitment, query=query, response=response,
            trust_anchor_pem=(local_tsa / "root.pem").read_bytes(),
            intermediate_chain_pem=(local_tsa / "tsa.pem").read_bytes(),
        )
