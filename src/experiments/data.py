"""Data loading built on Member 1's manifest and image recipes.

Speed: Colab's free tier has 2 CPU cores, which cannot run augmentation and resizing fast enough
for the GPU (about 160 images/s). So the workers only decode JPEGs into uint8 tensors, and the
recipe (the same augmentation, resize, scaling, normalisation and channel order as
src/dataset/preprocessing.py) runs on the GPU in `Preprocess`. tests/test_preprocess.py checks it
against Member 1's CPU recipe.

Reproducibility: shuffling depends only on (seed, epoch) and augmentation draws from the global
torch RNG, which the trainer re-seeds every epoch, so a resumed run sees the same batches.
"""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode, v2

from src.dataset.datasets import _seed_worker
from src.dataset.preprocessing import SPECS, augmentation

from .config import IMAGENET_MEAN, IMAGENET_STD

IMAGE_SIZE = 256   # every PlantVillage image is 256 x 256 (reports/audit.md)


# ---------------------------------------------------------------------------------------------
# dataset: decoded uint8 images
# ---------------------------------------------------------------------------------------------
def load_uint8(path) -> torch.Tensor:
    with Image.open(path) as im:
        im = im.convert("RGB")
    if im.size != (IMAGE_SIZE, IMAGE_SIZE):
        im = im.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    return torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1)   # 3 x H x W, uint8, RGB


class ImageDataset(Dataset):
    """Same rows, labels and class numbering as src.dataset.datasets.PlantVillageDataset."""

    def __init__(self, manifest_csv, data_root, split):
        from pathlib import Path
        m = pd.read_csv(manifest_csv)
        self.classes = sorted(m["label"].unique())
        index = {c: i for i, c in enumerate(self.classes)}
        rows = m[m["split"] == split]
        self.root = Path(data_root)
        self.paths = rows["path"].tolist()
        self.labels = [index[c] for c in rows["label"]]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return load_uint8(self.root / self.paths[i]), self.labels[i]


# ---------------------------------------------------------------------------------------------
# the model recipe, on the GPU
# ---------------------------------------------------------------------------------------------
class Preprocess(nn.Module):
    """uint8 RGB batch [B, 3, 256, 256] -> model input. `native` is Member 1's recipe; `imagenet`
    (ablation only) uses RGB and the ImageNet mean/std from the original proposal."""

    def __init__(self, model_key: str, variant: str = "native"):
        super().__init__()
        spec = SPECS[model_key]
        self.size = spec["size"]
        if variant == "native":
            mean, std, self.bgr = spec["mean"], spec["std"], spec["bgr"]
        elif variant == "imagenet":
            mean, std, self.bgr = IMAGENET_MEAN, IMAGENET_STD, False
        else:
            raise ValueError(f"unknown preprocessing variant {variant!r}")
        self.normalize = mean is not None
        self.register_buffer("mean", torch.tensor(mean or (0.0, 0.0, 0.0)).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std or (1.0, 1.0, 1.0)).view(1, 3, 1, 1))
        self.augment = v2.Compose(augmentation())   # flip, rotation +-15, brightness/contrast 0.2

    @torch.no_grad()
    def forward(self, x: torch.Tensor, train: bool) -> torch.Tensor:
        if train:
            # per image, so every image gets its own random parameters
            x = torch.stack([self.augment(img) for img in x])
        x = x.float()
        if x.shape[-1] != self.size:
            x = v2.functional.resize(x, [self.size, self.size], interpolation=InterpolationMode.BILINEAR,
                                     antialias=True).round_().clamp_(0, 255)   # uint8 levels, as on the CPU
        x = x / 255.0
        if self.normalize:
            x = (x - self.mean.to(x.device)) / self.std.to(x.device)
        if self.bgr:
            x = x.flip(1)
        return x.contiguous()


# ---------------------------------------------------------------------------------------------
# loaders
# ---------------------------------------------------------------------------------------------
class EpochSeededLoader:
    """`loader.set_epoch(e)` fixes the shuffle order (and worker seeds) of epoch `e`, so a resumed
    run repeats exactly the epochs an uninterrupted run would have seen."""

    def __init__(self, dataset, batch_size, shuffle, seed, num_workers, pin_memory, prefetch_factor):
        self.seed = seed
        self.generator = torch.Generator()
        kwargs = dict(batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      pin_memory=pin_memory, generator=self.generator, worker_init_fn=_seed_worker)
        if num_workers > 0:
            kwargs["prefetch_factor"] = prefetch_factor
            kwargs["persistent_workers"] = False
        self.loader = DataLoader(dataset, **kwargs)
        self.set_epoch(0)

    def set_epoch(self, epoch: int):
        self.generator.manual_seed(self.seed * 1000 + epoch)

    @property
    def dataset(self):
        return self.loader.dataset

    def __iter__(self):
        return iter(self.loader)

    def __len__(self):
        return len(self.loader)


def make_loaders(paths, cfg, seed, device="cuda"):
    """Train/val/test loaders of uint8 images. Same manifest and split for every run."""
    out = {}
    for split in ("train", "val", "test"):
        train = split == "train"
        out[split] = EpochSeededLoader(
            ImageDataset(paths.manifest, paths.data_root, split),
            cfg.batch_size if train else cfg.eval_batch_size, shuffle=train, seed=seed,
            num_workers=cfg.num_workers, pin_memory=(device == "cuda"), prefetch_factor=cfg.prefetch_factor)
    return out


# ---------------------------------------------------------------------------------------------
# robustness (E5): deterministic corruptions applied to the RGB image before the model recipe
# ---------------------------------------------------------------------------------------------
# Severity levels 1-3. Chosen to stay within what a phone camera in the field can produce
# (sensor noise, defocus, poor light, low contrast, heavy compression, low resolution).
CORRUPTIONS = {
    "gaussian_noise": (0.04, 0.08, 0.12),   # noise std on the [0, 1] scale
    "gaussian_blur": (1.0, 2.0, 3.0),       # blur radius in pixels (on 256 x 256 images)
    "low_brightness": (0.7, 0.5, 0.3),      # brightness factor
    "low_contrast": (0.7, 0.5, 0.3),        # contrast factor
    "jpeg": (30, 15, 5),                    # JPEG quality
    "low_resolution": (2, 4, 8),            # downscale factor before upscaling back
}


def corrupt(image: Image.Image, kind: str, severity: int, rng: np.random.Generator) -> Image.Image:
    """Apply corruption `kind` at `severity` (0 = clean, 1-3 = increasing) to an RGB image."""
    if severity == 0 or kind == "clean":
        return image
    level = CORRUPTIONS[kind][severity - 1]
    if kind == "gaussian_noise":
        a = np.asarray(image, dtype=np.float32) / 255.0
        a = np.clip(a + rng.normal(0.0, level, a.shape).astype(np.float32), 0, 1)
        return Image.fromarray((a * 255 + 0.5).astype(np.uint8))
    if kind == "gaussian_blur":
        return image.filter(ImageFilter.GaussianBlur(radius=level))
    if kind == "low_brightness":
        return ImageEnhance.Brightness(image).enhance(level)
    if kind == "low_contrast":
        return ImageEnhance.Contrast(image).enhance(level)
    if kind == "jpeg":
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=int(level))
        buf.seek(0)
        with Image.open(buf) as im:
            return im.convert("RGB")
    if kind == "low_resolution":
        w, h = image.size
        small = image.resize((w // level, h // level), Image.BILINEAR)
        return small.resize((w, h), Image.BILINEAR)
    raise ValueError(f"unknown corruption {kind!r}")


class CorruptedDataset(ImageDataset):
    """Test images after one corruption. Noise depends only on the image index, so every model
    sees exactly the same corrupted pixels."""

    def __init__(self, manifest_csv, data_root, kind, severity, split="test", seed=0):
        super().__init__(manifest_csv, data_root, split)
        self.kind, self.severity, self.seed = kind, severity, seed

    def __getitem__(self, i):
        with Image.open(self.root / self.paths[i]) as im:
            im = im.convert("RGB")
        im = corrupt(im, self.kind, self.severity, np.random.default_rng([self.seed, i]))
        return torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1), self.labels[i]
