# Deployment notes

## Cryptography in production
The reference `kecrl/crypto/ec_commit.py` is written for clarity and
auditability, not constant-time guarantees. For production:

- **Keys in an HSM.** Replace `SigningKey` with a PKCS#11 / cloud-KMS backend so
  the private key never leaves the hardware boundary. Prefer **threshold /
  multi-party ECDSA** to remove single points of failure.
- **Second generator H.** The reference derives the Pedersen generator `H` as
  `k*G` for simplicity, which leaks `log_G(H)`. In production derive `H` by a
  verifiable **hash-to-curve** so the discrete log is genuinely unknown
  (required for the binding argument).
- **Constant-time arithmetic.** Use a vetted library (e.g. `cryptography`,
  `libsodium`, or an HSM's native EC) for all scalar multiplications.
- **Timestamping.** Bind an external trusted timestamp (RFC 3161) in addition to
  the local timestamp to strengthen replay resistance.

## Three-tier architecture (see paper)
1. **Edge:** face detection + frame sampling.
2. **Inference:** KAN detector + RL controller on accelerated hardware; policy
   refreshed periodically from a red-team loop.
3. **Trust:** HSM signing + append-only ledger, optionally mirrored across
   jurisdictions for coalition auditability.

## Throughput
Per-frame pipeline ≈ 9.4 ms in the paper's setup (≈106 fps); the EC commit adds
≈0.7 ms at P-256 and is not the bottleneck. Larger curves (P-384/P-521) trade
throughput for security margin.
