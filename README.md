# KEC-RL

**Kolmogorov–Arnold + Reinforcement-Learning robust deepfake detection, with
elliptic-curve verdict provenance.**

Reference implementation for the paper *"Mitigating Adversarial Attacks Using
Kolmogorov Elliptic-Curve Based Reinforcement Learning for Counter-Terror
Detection of Deepfake Content."*

---

## ⚠️ Status: reference implementation (results not yet populated)

Please read this before using the repository or citing its numbers.

This repository contains a **complete, runnable reference implementation** of
every component described in the paper:

- the KAN encoder and robust detector (`kecrl/models/`),
- the PPO Markov-game controller (`kecrl/models/ppo_controller.py`),
- the elliptic-curve provenance layer (`kecrl/crypto/`),
- the adaptive-attack harness — PGD / C&W / boundary with
  expectation-over-transformation (`kecrl/attacks/`),
- the measured-smoothness probes — power-iteration Lipschitz and loss-landscape
  (`kecrl/analysis/`),
- training, evaluation, and figure-generation scripts (`scripts/`).

It does **not** yet ship trained checkpoints or a frozen `results/` set, and
the numbers currently shown in the manuscript figures/tables are **illustrative
placeholders produced by `scripts/make_figures.py` from a synthetic data
block**, not measurements from trained models. To obtain real results you must
run the training and evaluation pipeline below on the public datasets; the
scripts then emit a `results/*.json` that `scripts/make_figures.py` turns into
figures, so the figures and tables cannot disagree.

When you have real runs, replace the placeholder data block in
`scripts/make_figures.py` with the produced `results/*.json`, commit the
checkpoints under `checkpoints/`, and delete this status notice. The
`CITATION`/paper numbers should be updated from the same `results/` files.

This honest-status policy is deliberate: publishing a repository that claims to
"reproduce" numbers that were never measured would mislead readers.

---

## Install

```bash
git clone https://github.com/kec-rl/kec-rl
cd kec-rl
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Smoke test (no data, no GPU required)

Verifies the whole pipeline wires together on random tensors:

```bash
pytest -q                                   # unit tests
python scripts/train.py --synthetic --episodes 2 --steps 8 --out runs/smoke
python scripts/evaluate.py --synthetic --checkpoint runs/smoke/checkpoint.pt \
    --out results/smoke.json
```

## Real training & evaluation

1. Download the datasets from their official sources (see
   [`docs/DATASETS.md`](docs/DATASETS.md)). They are **not** redistributed here.
2. Build per-dataset frame manifests:
   ```bash
   python scripts/build_manifest.py --dataset faceforensics --data-root /data/ff
   ```
3. Train (repeat per dataset / seed):
   ```bash
   python scripts/train.py --dataset faceforensics --data-root /data/ff \
       --backbone xception --episodes 500 --seed 0 --out runs/ff_s0
   ```
4. Evaluate → results JSON (the single source of truth for figures):
   ```bash
   python scripts/evaluate.py --checkpoint runs/ff_s0/checkpoint.pt \
       --dataset faceforensics --data-root /data/ff --out results/ff_s0.json
   ```
5. Regenerate figures from the results:
   ```bash
   python scripts/make_figures.py --out figures
   ```

## Configuration

See [`configs/kecrl_ff.yaml`](configs/kecrl_ff.yaml). Fixed and tunable
parameters are separated, matching the paper's hyperparameter table.

## Repository layout

```
kecrl/
  models/     kan.py  detector.py  mlp_baseline.py  ppo_controller.py
  attacks/    harness.py          # PGD / C&W / boundary + EoT, ASR metric
  crypto/     ec_commit.py        # Pedersen commitment + ECDSA chain
  analysis/   lipschitz.py        # power-iteration + loss landscape
  data/       datasets.py         # identity-disjoint splits, MTCNN align
  env.py      train.py  evaluate.py
scripts/      train.py evaluate.py build_manifest.py make_figures.py
configs/      kecrl_ff.yaml
tests/        test_components.py  test_crypto.py
docs/         DATASETS.md  DEPLOY.md  MODEL_CARD.md
```

## Intended use & governance

This is defensive research for detecting synthetic media. See
[`docs/MODEL_CARD.md`](docs/MODEL_CARD.md) for intended use, limitations, and the
governance constraints (human-in-the-loop adjudication, HSM key custody,
fairness auditing) discussed in the paper. Do not deploy automated verdicts for
irreversible action without human oversight.

## Licence

Code: MIT. Trained weights (when released): CC-BY-4.0. See `LICENSE`.

## Citation

```bibtex
@article{kecrl2026,
  title   = {Mitigating Adversarial Attacks Using Kolmogorov Elliptic-Curve
             Based Reinforcement Learning for Counter-Terror Detection of
             Deepfake Content},
  author  = {Kiran L.N. Eranki, Senior Member IEEE, Faculty of Science and Technology, ICFAI Foundation for Higher Education, Hyderabad-501203, Telangana, India, erankikiran@ifheindia.org},
  journal = {(under review)},
  year    = {2026}
}
```# kec-rl-kec-rl
Mitigating Adversarial Attacks Using Kolmogorov Elliptic-Curve Based Reinforcement Learning
