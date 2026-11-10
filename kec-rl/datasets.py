"""Dataset loading with identity-disjoint splits and MTCNN face alignment.

Supports FaceForensics++, Celeb-DF v2, DFDC, and WildDeepfake. The datasets are
NOT redistributed here -- download them from their official sources (see
docs/DATASETS.md) and point `--data-root` at the extracted frames.

Split protocol: 70/15/15 train/val/test, split by IDENTITY so no subject
appears in more than one split. For FF++ and DFDC the official splits are used;
for Celeb-DF and WildDeepfake we fix an identity-disjoint split by seed and
write the indices to `splits/<dataset>_<seed>.json` so results are reproducible.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from torch.utils.data import Dataset

try:
    from PIL import Image
    import torchvision.transforms as T
except Exception:  # pragma: no cover
    Image = None


DATASETS = ["faceforensics", "celebdf", "dfdc", "wilddeepfake"]


def default_transform(size: int = 224):
    return T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),  # -> [0,1], matching the attack harness clamp range
    ])


class FrameDataset(Dataset):
    """Frame-level dataset. Expects a manifest listing (path, label, identity)."""

    def __init__(self, manifest: list[dict], root: str, transform=None, size: int = 224):
        self.items = manifest
        self.root = Path(root)
        self.transform = transform or default_transform(size)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        rec = self.items[i]
        img = Image.open(self.root / rec["path"]).convert("RGB")
        return self.transform(img), int(rec["label"])


def load_manifest(data_root: str, dataset: str) -> list[dict]:
    """Load a prebuilt manifest JSON (built by scripts/build_manifest.py)."""
    path = Path(data_root) / dataset / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {path}. Run scripts/build_manifest.py first "
            f"(see docs/DATASETS.md)."
        )
    with open(path) as f:
        return json.load(f)


def identity_disjoint_split(manifest: list[dict], seed: int = 0,
                            ratios=(0.70, 0.15, 0.15)):
    """Split a manifest by identity. Returns (train, val, test) manifests."""
    identities = sorted({r["identity"] for r in manifest})
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(identities), generator=g).tolist()
    identities = [identities[i] for i in perm]
    n = len(identities)
    n_tr = int(ratios[0] * n)
    n_va = int(ratios[1] * n)
    tr_ids = set(identities[:n_tr])
    va_ids = set(identities[n_tr:n_tr + n_va])
    te_ids = set(identities[n_tr + n_va:])
    buckets = {"train": [], "val": [], "test": []}
    for r in manifest:
        if r["identity"] in tr_ids:
            buckets["train"].append(r)
        elif r["identity"] in va_ids:
            buckets["val"].append(r)
        else:
            buckets["test"].append(r)
    return buckets["train"], buckets["val"], buckets["test"]


def save_split_indices(dataset: str, seed: int, splits, out_dir: str = "splits"):
    os.makedirs(out_dir, exist_ok=True)
    payload = {k: [r["path"] for r in v]
               for k, v in zip(["train", "val", "test"], splits)}
    with open(Path(out_dir) / f"{dataset}_{seed}.json", "w") as f:
        json.dump(payload, f, indent=2)
