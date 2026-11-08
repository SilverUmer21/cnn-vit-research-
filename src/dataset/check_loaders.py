# Checks the loaders and the two image recipes. Exits with 1 if a check fails.
# Usage (from the repo root): python -m src.dataset.check_loaders <manifest_csv> <data_root>
import sys

import pandas as pd
import torch
from PIL import Image

from .datasets import make_loader, seed_everything
from .preprocessing import SPECS, augmentation, build_transform

failures = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""), flush=True)
    if not ok:
        failures.append(name)


def main():
    manifest_csv, data_root = sys.argv[1], sys.argv[2]
    manifest = pd.read_csv(manifest_csv)
    counts = manifest["split"].value_counts().to_dict()

    for model, spec in SPECS.items():
        size = spec["size"]
        for split in ("train", "val", "test"):
            loader = make_loader(model, split, manifest_csv, data_root)
            ds = loader.dataset
            check(f"{model}/{split}: size matches the manifest", len(ds) == counts[split],
                  f"{len(ds):,} images")
            x, y = next(iter(loader))
            check(f"{model}/{split}: batch shape and type",
                  tuple(x.shape) == (32, 3, size, size) and x.dtype == torch.float32
                  and isinstance(x, torch.Tensor), f"{tuple(x.shape)} {x.dtype}")
            lo, hi = float(x.min()), float(x.max())
            allowed = (-1.0, 1.0) if spec["mean"] is not None else (0.0, 1.0)
            check(f"{model}/{split}: values in {allowed}", allowed[0] <= lo and hi <= allowed[1],
                  f"min {lo:.3f}, max {hi:.3f}")
            check(f"{model}/{split}: labels are class numbers 0-37", int(y.min()) >= 0 and int(y.max()) <= 37)
        check(f"{model}: 38 classes", len(ds.classes) == 38)

    # channel order and scaling on a pure red image
    red = Image.new("RGB", (256, 256), (255, 0, 0))
    v = build_transform("mobilevit_xs", False)(red)
    n = build_transform("mobilenetv2", False)(red)
    check("MobileViT: BGR order (red lands in the last channel)",
          torch.allclose(v[:, 0, 0], torch.tensor([0.0, 0.0, 1.0])), str(v[:, 0, 0].tolist()))
    check("MobileNetV2: RGB order, mean/std 0.5 (red -> 1, others -> -1)",
          torch.allclose(n[:, 0, 0], torch.tensor([1.0, -1.0, -1.0])), str(n[:, 0, 0].tolist()))

    # augmentation: same for both models, training only
    check("Same augmentation for both models",
          [repr(t) for t in build_transform("mobilenetv2", True).transforms[:3]]
          == [repr(t) for t in build_transform("mobilevit_xs", True).transforms[:3]]
          == [repr(t) for t in augmentation()])
    leaf = Image.open(f"{data_root}/{manifest['path'][0]}").convert("RGB")
    ev = build_transform("mobilevit_xs", False)
    tr = build_transform("mobilevit_xs", True)
    check("Validation/test images are not augmented", torch.equal(ev(leaf), ev(leaf)))
    check("Training images are augmented", any(not torch.equal(tr(leaf), tr(leaf)) for _ in range(5)))

    # same seed gives the same batches
    batches = []
    for _ in range(2):
        seed_everything(42)
        batches.append(next(iter(make_loader("mobilevit_xs", "train", manifest_csv, data_root, batch_size=8))))
    check("Seed 42 gives identical batches",
          torch.equal(batches[0][0], batches[1][0]) and torch.equal(batches[0][1], batches[1][1]))

    # worker processes (the recipe has to be picklable for Windows)
    x, _ = next(iter(make_loader("mobilenetv2", "train", manifest_csv, data_root, batch_size=8, num_workers=2)))
    check("Works with worker processes", tuple(x.shape) == (8, 3, 224, 224))

    print(f"\n{'ALL CHECKS PASSED' if not failures else 'FAILED: ' + ', '.join(failures)}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
