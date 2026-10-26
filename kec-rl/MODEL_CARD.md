# Model Card — KEC-RL

## Overview
KEC-RL is a defensive deepfake detector combining a Kolmogorov–Arnold (KAN)
feature encoder, a PPO Markov-game controller for adaptive defence, and an
auxiliary elliptic-curve verdict-provenance layer. The robustness contribution
is KAN + RL; the elliptic-curve layer provides verdict integrity only and does
not affect detection accuracy.

## Intended use
- Research on adversarial robustness of synthetic-media detection.
- A triage / evidence-preservation layer **supporting** human analysts.

## Out-of-scope / prohibited use
- Fully automated, irreversible decisions without human adjudication.
- Mass surveillance or suppression of lawful speech.
- Any use that contravenes applicable law or the governance constraints below.

## Limitations (see paper, Challenges and Limitations)
- **Dataset coverage:** evaluated only on four face-centric benchmarks; no claim
  for full-body, document, or audio deepfakes.
- **Generalisation:** robustness established only within the stated ℓ∞ threat
  model and present generator families; semantic / physical-world / future-
  generator attacks are out of scope.
- **Deployment:** guarantees assume secure HSM key custody and a human-in-the-
  loop layer.
- **Fairness:** face analysis can exhibit demographic performance disparities;
  disaggregated auditing is required before any operational use.

## Governance constraints
- Human-in-the-loop adjudication for any consequential verdict.
- Hardware-rooted key custody (HSM; threshold signing recommended).
- Disaggregated fairness reporting prior to deployment.
- Independent oversight, scope limitation, and retention policy.

## Provenance and integrity
Verdicts are committed with an EC Pedersen commitment and ECDSA signature into
an append-only chain, giving non-repudiation, selective disclosure, and replay
resistance. This is evidentiary, not a defence against evasion.

## Licence
Code MIT; trained weights CC-BY-4.0.
