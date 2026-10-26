"""Elliptic-curve verdict provenance (auxiliary; NOT a robustness mechanism).

This layer runs AFTER classification and cannot change detection accuracy. It
provides, over and above a plain hash-chained log:
  (i)   third-party non-repudiation  (ECDSA signature, HSM-held key)
  (ii)  hiding + selective disclosure (Pedersen commitment Com(m,r)=mG+rH)
  (iii) replay resistance            (digest binds timestamp + previous head)

Implementation notes
---------------------
* ECDSA uses the `cryptography` library over NIST P-256 by default.
* The Pedersen commitment is implemented over the same curve using the
  `tinyec` group arithmetic for a clear, auditable reference. For production
  use a constant-time library and an HSM/PKCS#11 backend (see docs/DEPLOY.md).
"""
from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass, field

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.utils import (
    decode_dss_signature,
)


def _sha256(*parts: bytes) -> bytes:
    h = hashlib.sha256()
    for p in parts:
        h.update(p)
    return h.digest()


# --------------------------------------------------------------------------
# Pedersen commitment over a prime-order EC group (reference implementation)
# --------------------------------------------------------------------------
try:
    from tinyec import registry as _tinyec_registry

    _CURVE = _tinyec_registry.get_curve("secp256r1")
    _G = _CURVE.g
    # Second generator H with unknown discrete log wrt G: hash-to-scalar * G.
    _H_SCALAR = int.from_bytes(_sha256(b"KEC-RL/pedersen/H"), "big") % _CURVE.field.n
    _H = _H_SCALAR * _G  # NB: for a real deployment, derive H by hash-to-curve,
    #                       not as k*G, so that log_G(H) is genuinely unknown.
    _TINYEC = True
except Exception:  # pragma: no cover - optional dependency
    _TINYEC = False


def pedersen_commit(m: int, r: int):
    """Return the commitment point Com(m, r) = m*G + r*H.

    Perfectly hiding (r uniform) and computationally binding under ECDLP.
    """
    if not _TINYEC:
        raise RuntimeError("tinyec not installed; `pip install tinyec`")
    n = _CURVE.field.n
    return (m % n) * _G + (r % n) * _H


def commit_point_bytes(point) -> bytes:
    return point.x.to_bytes(32, "big") + point.y.to_bytes(32, "big")


# --------------------------------------------------------------------------
# Signing key (software HSM stand-in)
# --------------------------------------------------------------------------
@dataclass
class SigningKey:
    """Wraps an EC private key. In production this lives in an HSM via PKCS#11."""

    _key: ec.EllipticCurvePrivateKey

    @classmethod
    def generate(cls, curve: ec.EllipticCurve | None = None) -> "SigningKey":
        return cls(ec.generate_private_key(curve or ec.SECP256R1()))

    def sign(self, digest: bytes) -> bytes:
        return self._key.sign(digest, ec.ECDSA(hashes.SHA256()))

    def public_bytes(self) -> bytes:
        return self._key.public_key().public_bytes(
            serialization.Encoding.DER,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )

    def verify(self, signature: bytes, digest: bytes) -> bool:
        try:
            self._key.public_key().verify(
                signature, digest, ec.ECDSA(hashes.SHA256())
            )
            return True
        except Exception:
            return False


# --------------------------------------------------------------------------
# Provenance chain
# --------------------------------------------------------------------------
@dataclass
class ChainEntry:
    index: int
    timestamp: float
    verdict: int
    score: float
    commitment: bytes
    prev_head: bytes
    head: bytes
    signature: bytes


@dataclass
class ProvenanceChain:
    key: SigningKey
    head: bytes = field(default_factory=lambda: _sha256(b"KEC-RL/genesis"))
    entries: list[ChainEntry] = field(default_factory=list)

    def commit_verdict(self, embedding_digest: bytes, verdict: int, score: float):
        """Commit one verdict. Digest binds timestamp + previous head (replay-proof)."""
        ts = time.time()
        n = _CURVE.field.n if _TINYEC else (2 ** 256 - 189)
        m = int.from_bytes(
            _sha256(embedding_digest, bytes([verdict]),
                    f"{score:.6f}".encode(), f"{ts:.6f}".encode(), self.head),
            "big",
        ) % n
        r = int.from_bytes(os.urandom(32), "big") % n
        if _TINYEC:
            com = commit_point_bytes(pedersen_commit(m, r))
        else:  # fallback: hash commitment (loses the hiding/homomorphism property)
            com = _sha256(m.to_bytes(32, "big"), r.to_bytes(32, "big"))
        new_head = _sha256(self.head, com)
        sig = self.key.sign(new_head)
        entry = ChainEntry(
            index=len(self.entries), timestamp=ts, verdict=verdict, score=score,
            commitment=com, prev_head=self.head, head=new_head, signature=sig,
        )
        self.entries.append(entry)
        self.head = new_head
        return entry

    def verify(self) -> bool:
        """Verify chain linkage and every signature."""
        head = _sha256(b"KEC-RL/genesis")
        for e in self.entries:
            if e.prev_head != head:
                return False
            if _sha256(head, e.commitment) != e.head:
                return False
            if not self.key.verify(e.signature, e.head):
                return False
            head = e.head
        return head == self.head
