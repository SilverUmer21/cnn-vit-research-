"""Synthetic stand-in for the PlantVillage folder, for testing the whole pipeline without the real
data (e.g. on a laptop or in CI). Never used for any reported result."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def make_smoke_data(manifest_csv, out_root, out_manifest, per_class=(6, 2, 2), seed=0):
    """Take `per_class` (train, val, test) rows per class from the real manifest and write a
    256x256 JPEG for each at the same relative path. Each class gets its own colour pattern, so
    a model can actually learn something in a few steps."""
    rng = np.random.default_rng(seed)
    m = pd.read_csv(manifest_csv)
    parts = []
    for split, n in zip(("train", "val", "test"), per_class):
        parts.append(m[m["split"] == split].groupby("label", group_keys=False).head(n))
    sub = pd.concat(parts).reset_index(drop=True)
    classes = sorted(m["label"].unique())
    base = {c: rng.integers(30, 225, 3) for c in classes}
    yy, xx = np.mgrid[0:256, 0:256]
    for _, r in sub.iterrows():
        k = classes.index(r["label"])
        img = np.empty((256, 256, 3), np.float32)
        img[:] = base[r["label"]]
        img += 40 * np.sin((xx * (k % 7 + 1) + yy * (k // 7 + 1)) / 20.0)[..., None]
        img += rng.normal(0, 8, img.shape)
        dst = Path(out_root) / r["path"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(dst, "JPEG", quality=90)
    Path(out_manifest).parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out_manifest, index=False)
    return sub
