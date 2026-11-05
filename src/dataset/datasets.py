# Shared data loader. It reads splits/manifest.csv and never re-splits.
# Usage: loader = make_loader("mobilevit_xs", "train", manifest_csv, data_root)
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .preprocessing import build_transform

SPLITS = ("train", "val", "test")


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


class PlantVillageDataset(Dataset):
    def __init__(self, manifest_csv, data_root, split, transform):
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
        manifest = pd.read_csv(manifest_csv)
        # class numbers: all class names in alphabetical order, the same for every split
        self.classes = sorted(manifest["label"].unique())
        index = {name: i for i, name in enumerate(self.classes)}
        rows = manifest[manifest["split"] == split]
        self.root = Path(data_root)
        self.paths = rows["path"].tolist()
        self.labels = [index[name] for name in rows["label"]]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.root / self.paths[i]) as image:
            image = image.convert("RGB")
        return self.transform(image), self.labels[i]


def make_loader(model_name, split, manifest_csv, data_root, batch_size=32, num_workers=0, seed=42):
    train = split == "train"
    dataset = PlantVillageDataset(manifest_csv, data_root, split, build_transform(model_name, train))
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=num_workers,
                      generator=generator, worker_init_fn=_seed_worker)
