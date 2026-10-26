# Datasets

The four benchmarks are **not** redistributed in this repository. Obtain each
from its official source under its own licence, then extract frames and point
`--data-root` at the result.

| Dataset | Source / access | Notes |
|---|---|---|
| FaceForensics++ | Request access via the official GitHub form | 1,000 pristine + 4,000 manipulated (4 methods); use the official split |
| Celeb-DF v2 | Request access via the official form | high-quality celebrity face-swaps |
| DFDC | Kaggle (DeepFake Detection Challenge) | large, diverse; use preview/full as stated |
| WildDeepfake | Official request form | in-the-wild, heterogeneous quality |

## Expected on-disk layout

```
<data-root>/
  faceforensics/
    frames/<video_id>/<frame>.png
    manifest.json        # built by scripts/build_manifest.py
  celebdf/ ...
  dfdc/ ...
  wilddeepfake/ ...
```

## Manifest format

`manifest.json` is a list of records:

```json
[
  {"path": "frames/vid001/000012.png", "label": 1, "identity": "id17"},
  {"path": "frames/vid204/000003.png", "label": 0, "identity": "id88"}
]
```

* `label`: 1 = fake, 0 = real.
* `identity`: a subject/source identifier used to make the train/val/test
  split **identity-disjoint** (no subject appears in two splits).

Build it with:

```bash
python scripts/build_manifest.py --dataset faceforensics --data-root /data/ff
```

## Splits

`kecrl/data/datasets.py` performs a 70/15/15 identity-disjoint split and writes
the exact indices to `splits/<dataset>_<seed>.json`. For FF++ and DFDC prefer
the official splits where available; the released index files pin whatever was
used so results are reproducible on the same partitions.
