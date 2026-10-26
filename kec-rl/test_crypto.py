"""Tests for the elliptic-curve provenance chain. Run: pytest -q"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kecrl.crypto.ec_commit import ProvenanceChain, SigningKey


def test_chain_commits_and_verifies():
    chain = ProvenanceChain(SigningKey.generate())
    for i in range(5):
        chain.commit_verdict(b"digest" + bytes([i]), verdict=i % 2, score=0.9)
    assert chain.verify() is True
    assert len(chain.entries) == 5


def test_tamper_is_detected():
    chain = ProvenanceChain(SigningKey.generate())
    chain.commit_verdict(b"d0", verdict=1, score=0.8)
    chain.commit_verdict(b"d1", verdict=1, score=0.7)
    # tamper with a recorded verdict
    chain.entries[0].verdict = 0
    chain.entries[0].head = b"\x00" * 32
    assert chain.verify() is False


def test_replay_binding_changes_head():
    """Committing the same verdict twice yields different heads (timestamp+prev)."""
    chain = ProvenanceChain(SigningKey.generate())
    e1 = chain.commit_verdict(b"same", verdict=1, score=0.9)
    e2 = chain.commit_verdict(b"same", verdict=1, score=0.9)
    assert e1.head != e2.head
