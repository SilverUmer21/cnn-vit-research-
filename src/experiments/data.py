"""Data loaders built on Member 1's manifest, dataset class and image recipes.

What this adds on top of ``src.dataset.datasets.make_loader``:
- worker processes, pinned memory and prefetching (the default loader uses 0 workers, which
  starves the GPU on Colab);
- shuffling and augmentation that depend only on (seed, epoch), so a run resumed after a Colab
  disconnect sees exactly the batches it would have seen without the disconnect;
- the ImageNet-normalisation variant used only for the preprocessing ablation (E6);
- a test-set dataset with deterministic image corruptions for the robustness test (E5).
"""
from __future__ import annotations

import io

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode, v2

from src.dataset.datasets import PlantVillageDataset, _seed_worker
from src.dataset.preprocessing import SPECS, augmentation, build_transform

from .config import IMAGENET_MEAN, IMAGENET_STD


# ---------------------------------------------------------------------------------------------
# transforms
# ---------------------------------------------------------------------------------------------
def make_transform(model_key: str, train: bool, variant: str = "native"):
    """`native` is Member 1's recipe, unchanged. `imagenet` keeps the same augmentation, size and
    resize, but uses RGB and the ImageNet mean/std written in the original proposal."""
    if variant == "native":
        return build_transform(model_key, train)
    if variant != "imagenet":
        raise ValueError(f"unknown preprocessing variant {variant!r}")
    size = SPECS[model_key]["size"]
    steps = augmentation() if train else []
    steps += [
        v2.Resize((size, size), interpolation=InterpolationMode.BILINEAR, antialias=True),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    return v2.Compose(steps)


def to_display(x: torch.Tensor, model_key: str, variant: str = "native") -> np.ndarray:
    """Undo the model's normalisation/channel order: CHW tensor -> HWC RGB float array in [0, 1]."""
    x = x.detach().float().cpu().clone()
    if variant == "imagenet":
        x = x * torch.tensor(IMAGENET_STD)[:, None, None] + torch.tensor(IMAGENET_MEAN)[:, None, None]
    else:
        spec = SPECS[model_key]
        if spec["bgr"]:
            x = x.flip(0)
        if spec["mean"] is not None:
            x = x * torch.tensor(spec["std"])[:, None, None] + torch.tensor(spec["mean"])[:, None, None]
    return x.clamp(0, 1).permute(1, 2, 0).numpy()


# ---------------------------------------------------------------------------------------------
# loaders
# ---------------------------------------------------------------------------------------------
class EpochSeededLoader:
    """Wraps a DataLoader so that `loader.set_epoch(e)` fixes the shuffle order AND the
    per-worker augmentation seeds for epoch `e`.

    PyTorch draws both the sampler permutation and the workers' base seed from the DataLoader's
    generator when an iterator is created, so re-seeding that generator at the start of every
    epoch makes epoch `e` identical whether or not the run was interrupted before it. Workers are
    not persistent for the same reason (they would keep their first seed)."""

    def __init__(self, dataset, batch_size, shuffle, seed, num_workers, pin_memory, prefetch_factor,
                 drop_last=False):
        self.seed = seed
        self.generator = torch.Generator()
        kwargs = dict(batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      pin_memory=pin_memory, generator=self.generator, worker_init_fn=_seed_worker,
                      drop_last=drop_last)
        if num_workers > 0:
            kwargs["prefetch_factor"] = prefetch_factor
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


def make_loaders(paths, model_key, cfg, seed, variant="native", device="cuda"):
    """Train/val/test loaders for one run. Same manifest and split for every run."""
    pin = device == "cuda"
    out = {}
    for split in ("train", "val", "test"):
        train = split == "train"
        ds = PlantVillageDataset(paths.manifest, paths.data_root, split, make_transform(model_key, train, variant))
        out[split] = EpochSeededLoader(
            ds, cfg.batch_size if train else cfg.eval_batch_size, shuffle=train, seed=seed,
            num_workers=cfg.num_workers, pin_memory=pin, prefetch_factor=cfg.prefetch_factor)
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


class CorruptedMultiModelDataset(Dataset):
    """Test images, corrupted once, then prepared for every model's recipe. All models therefore
    see exactly the same corrupted pixels; noise depends only on the image index."""

    def __init__(self, manifest_csv, data_root, kind, severity, model_keys, split="test", seed=0):
        base = PlantVillageDataset(manifest_csv, data_root, split, transform=None)
        self.root, self.paths, self.labels, self.classes = base.root, base.paths, base.labels, base.classes
        self.kind, self.severity, self.seed = kind, severity, seed
        self.transforms = {k: build_transform(k, train=False) for k in model_keys}

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.root / self.paths[i]) as im:
            im = im.convert("RGB")
        rng = np.random.default_rng([self.seed, i])
        im = corrupt(im, self.kind, self.severity, rng)
        return {k: t(im) for k, t in self.transforms.items()}, self.labels[i]
