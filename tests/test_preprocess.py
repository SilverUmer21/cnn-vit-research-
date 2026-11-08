# Checks that the GPU-side recipe (src/experiments/data.py) matches Member 1's CPU recipe
# (src/dataset/preprocessing.py). Run from the repo root: python -m tests.test_preprocess <image> ...
import sys

import torch
from PIL import Image

from src.dataset.preprocessing import build_transform
from src.experiments.data import Preprocess, load_uint8

device = "cuda" if torch.cuda.is_available() else "cpu"
paths = sys.argv[1:]
assert paths, "pass a few image paths"

for model in ("mobilenetv2", "mobilevit_xs"):
    prep = Preprocess(model).to(device)
    worst = 0.0
    for p in paths:
        with Image.open(p) as im:
            cpu = build_transform(model, train=False)(im.convert("RGB"))
        gpu = prep(load_uint8(p)[None].to(device), train=False)[0].cpu()
        assert gpu.shape == cpu.shape, (gpu.shape, cpu.shape)
        worst = max(worst, (gpu - cpu).abs().max().item())
    # one uint8 level after normalisation: 1/255 (MobileViT) or 2/255 (MobileNetV2, std 0.5)
    tol = 2 / 255 + 1e-6
    print(f"{model}: eval max abs difference vs CPU recipe = {worst:.5f} (tolerance {tol:.5f})")
    assert worst <= tol

    x = load_uint8(paths[0])[None].repeat(8, 1, 1, 1).to(device)
    torch.manual_seed(0)
    a = prep(x, train=True)
    torch.manual_seed(0)
    b = prep(x, train=True)
    assert torch.equal(a, b), "augmentation is not reproducible under the same seed"
    assert not torch.equal(a[0], a[1]), "images in a batch got identical augmentation"
    lo, hi = (-1.0, 1.0) if model == "mobilenetv2" else (0.0, 1.0)
    assert lo <= a.min() and a.max() <= hi, (a.min(), a.max())
    print(f"{model}: train augmentation reproducible, per-image, values in [{lo}, {hi}]")
print("OK")
