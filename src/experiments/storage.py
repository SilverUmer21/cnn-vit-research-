"""Crash-safe file writing. On Colab the results folder is on Google Drive, and a runtime can be
killed at any moment, so every file is written to a temporary name first and then renamed:
a half-written checkpoint can never replace a good one."""
from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch


def _atomic(dst: Path, write_fn):
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    # write on the fast local disk, then copy next to the destination and rename (atomic on the
    # same file system, including the Drive mount)
    with tempfile.TemporaryDirectory() as tmp:
        local = Path(tmp) / dst.name
        write_fn(local)
        staged = dst.with_name(dst.name + ".partial")
        shutil.copyfile(local, staged)
        os.replace(staged, dst)


def save_torch(obj, dst):
    _atomic(dst, lambda p: torch.save(obj, p))


def load_torch(src, map_location="cpu"):
    return torch.load(src, map_location=map_location, weights_only=False)


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def save_json(obj, dst):
    _atomic(dst, lambda p: p.write_text(json.dumps(obj, indent=2, default=_jsonable)))


def load_json(src):
    return json.loads(Path(src).read_text())


def save_text(text, dst):
    _atomic(dst, lambda p: p.write_text(text))


def save_npz(dst, **arrays):
    _atomic(dst, lambda p: np.savez_compressed(p, **arrays))


def save_dataframe(df, dst):
    _atomic(dst, lambda p: df.to_csv(p, index=False))
