#!/usr/bin/env python3
"""Train KEC-RL on a dataset.

Example:
    python scripts/train.py --dataset faceforensics --data-root /data/ff \
        --backbone xception --episodes 500 --out runs/ff_kecrl

This trains the full KEC-RL system (KAN encoder + robust detector + PPO
controller + EC provenance). The resulting checkpoint + returns history are
written to --out. Replace the synthetic fallback with real data by pointing
--data-root at frames prepared via scripts/build_manifest.py.
"""
import argparse
import json
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kecrl.models.detector import DeepfakeDetector
from kecrl.train import TrainConfig, train_kecrl
from kecrl.env import RewardWeights


def infinite(loader):
    while True:
        for batch in loader:
            yield batch


def make_loader(args):
    """Build a real DataLoader, or a synthetic one if --synthetic is set."""
    if args.synthetic:
        from torch.utils.data import TensorDataset
        n = 256
        x = torch.rand(n, 3, 224, 224)
        y = torch.randint(0, 2, (n,))
        return DataLoader(TensorDataset(x, y), batch_size=args.batch_size,
                          shuffle=True, drop_last=True)
    from kecrl.data.datasets import (
        load_manifest, identity_disjoint_split, save_split_indices, FrameDataset)
    manifest = load_manifest(args.data_root, args.dataset)
    tr, va, te = identity_disjoint_split(manifest, seed=args.seed)
    save_split_indices(args.dataset, args.seed, (tr, va, te))
    ds = FrameDataset(tr, root=Path(args.data_root) / args.dataset)
    return DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                      num_workers=args.workers, drop_last=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="faceforensics")
    ap.add_argument("--data-root", default="")
    ap.add_argument("--backbone", default="xception")
    ap.add_argument("--episodes", type=int, default=500)
    ap.add_argument("--steps", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--eps", type=float, default=0.03)
    ap.add_argument("--lambda-curv", type=float, default=1e-2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--synthetic", action="store_true",
                    help="use random data (smoke test; no real training)")
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--out", default="runs/exp")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    loader = make_loader(args)
    data_iter = infinite(loader)

    model = DeepfakeDetector(backbone=args.backbone,
                             pretrained=not args.no_pretrained)
    cfg = TrainConfig(episodes=args.episodes, steps_per_episode=args.steps,
                      eps=args.eps, lambda_curv=args.lambda_curv, lr=args.lr,
                      seed=args.seed)
    weights = RewardWeights(a1=1.0, a2=1.0, a3=0.2, a4=0.05)

    model, ppo, chain, returns = train_kecrl(model, data_iter, cfg, weights)

    torch.save({"model": model.state_dict(),
                "ppo": ppo.net.state_dict(),
                "config": vars(args)},
               Path(args.out) / "checkpoint.pt")
    with open(Path(args.out) / "returns.json", "w") as f:
        json.dump(returns, f)
    print(f"[done] chain verified = {chain.verify()}; "
          f"saved to {args.out}/checkpoint.pt")


if __name__ == "__main__":
    main()
