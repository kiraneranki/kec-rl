#!/usr/bin/env python3
"""Evaluate a trained KEC-RL checkpoint and write a results JSON.

The results JSON is the single source of truth consumed by
scripts/make_figures.py, so figures can never disagree with the tables.

Example:
    python scripts/evaluate.py --checkpoint runs/ff_kecrl/checkpoint.pt \
        --dataset faceforensics --data-root /data/ff --out results/ff.json
"""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kecrl.models.detector import DeepfakeDetector
from kecrl.evaluate import (clean_accuracy, robust_accuracy, auc_score,
                            epsilon_sweep, adaptive_asr_table)
from kecrl.analysis.lipschitz import global_lipschitz, loss_landscape_1d


def make_test_loader(args):
    if args.synthetic:
        from torch.utils.data import TensorDataset
        x = torch.rand(128, 3, 224, 224); y = torch.randint(0, 2, (128,))
        return DataLoader(TensorDataset(x, y), batch_size=args.batch_size)
    from kecrl.data.datasets import load_manifest, identity_disjoint_split, FrameDataset
    manifest = load_manifest(args.data_root, args.dataset)
    _, _, te = identity_disjoint_split(manifest, seed=args.seed)
    ds = FrameDataset(te, root=Path(args.data_root) / args.dataset)
    return DataLoader(ds, batch_size=args.batch_size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dataset", default="faceforensics")
    ap.add_argument("--data-root", default="")
    ap.add_argument("--backbone", default="xception")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--out", default="results/eval.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DeepfakeDetector(backbone=args.backbone, pretrained=False).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"]); model.eval()

    loader = make_test_loader(args)

    results = {
        "dataset": args.dataset,
        "clean_accuracy": clean_accuracy(model, loader, device),
        "robust_accuracy_eps_sweep": epsilon_sweep(model, loader, device),
        "auc": auc_score(model, loader, device),
    }

    # measured smoothness probes (one batch)
    x, y = next(iter(loader)); x, y = x.to(device), y.to(device)
    results["lipschitz_global_kan"] = global_lipschitz(model.encoder,
                                                       model.features(x))
    alphas, curve = loss_landscape_1d(model, x, y)
    results["loss_landscape"] = {"alphas": alphas, "loss": curve}

    # adaptive ASR on the fake subset
    fake_mask = y == 1
    if fake_mask.sum() > 0:
        results["adaptive_asr"] = adaptive_asr_table(model, x[fake_mask],
                                                      y[fake_mask])

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[done] wrote {args.out}")
    print(json.dumps(results, indent=2)[:800])


if __name__ == "__main__":
    main()
