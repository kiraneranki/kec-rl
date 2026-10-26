#!/usr/bin/env python3
"""Build a frame manifest for a dataset.

Scans <data-root>/<dataset>/frames/<video_id>/*.png and writes manifest.json
with (path, label, identity). Labelling and identity extraction are
dataset-specific; the hooks below cover the common layout and should be adapted
to each dataset's directory/metadata convention (see docs/DATASETS.md).
"""
import argparse
import json
from pathlib import Path


def label_and_identity(dataset: str, video_id: str) -> tuple[int, str]:
    """Return (label, identity) for a video id. ADAPT per dataset.

    Default heuristic: ids containing 'real'/'orig' are real (0); else fake (1).
    Identity = leading token of the video id.
    """
    vid = video_id.lower()
    label = 0 if ("real" in vid or "orig" in vid) else 1
    identity = video_id.split("_")[0]
    return label, identity


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--data-root", required=True)
    args = ap.parse_args()

    base = Path(args.data_root) / args.dataset / "frames"
    if not base.exists():
        raise SystemExit(f"No frames dir at {base} (see docs/DATASETS.md)")

    records = []
    for vid_dir in sorted(base.iterdir()):
        if not vid_dir.is_dir():
            continue
        label, identity = label_and_identity(args.dataset, vid_dir.name)
        for frame in sorted(vid_dir.glob("*.png")):
            rel = frame.relative_to(Path(args.data_root) / args.dataset)
            records.append({"path": str(rel), "label": label,
                            "identity": identity})

    out = Path(args.data_root) / args.dataset / "manifest.json"
    with open(out, "w") as f:
        json.dump(records, f)
    print(f"[done] {len(records)} frames -> {out}")


if __name__ == "__main__":
    main()
