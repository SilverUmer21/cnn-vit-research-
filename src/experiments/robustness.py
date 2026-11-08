"""Robustness / generalisation test (E5).

The trained models (all seeds, both architectures) are evaluated on the test set after common
image degradations at three severities. No model is retrained. Every image is corrupted once and
the same corrupted pixels go to every model, so differences come from the models only.
This does not replace field data (out of scope, see the proposal's limitations); it only shows how
quickly each model degrades when lab images get closer to phone-camera conditions."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .data import CORRUPTIONS, CorruptedDataset, Preprocess
from .evaluate import classification_metrics


@torch.inference_mode()
def evaluate_corruptions(models: dict, paths, device, batch_size=128, num_workers=2,
                         corruptions=None, max_batches=None, log=print):
    """models: {run_name: (model_key, nn.Module)}. Returns a long DataFrame with one row per
    (run, corruption, severity)."""
    keys = sorted({k for k, _ in models.values()})
    preps = {k: Preprocess(k).to(device) for k in keys}
    for _, m in models.values():
        m.to(device).eval()
    corruptions = corruptions or list(CORRUPTIONS)
    settings = [("clean", 0)] + [(c, s) for c in corruptions for s in (1, 2, 3)]
    rows = []
    for kind, severity in settings:
        ds = CorruptedDataset(paths.manifest, paths.data_root, kind, severity)
        loader = DataLoader(ds, batch_size=batch_size, num_workers=num_workers,
                            pin_memory=(device == "cuda"))
        preds = {name: [] for name in models}
        labels = []
        for i, (x, y) in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            labels.append(y)
            x = x.to(device, non_blocking=True)
            inputs = {k: preps[k](x, train=False) for k in keys}
            for name, (key, model) in models.items():
                preds[name].append(model(inputs[key]).argmax(1).cpu())
        y = torch.cat(labels).numpy()
        for name, (key, _) in models.items():
            m = classification_metrics(y, torch.cat(preds[name]).numpy(), len(ds.classes))
            rows.append({"run": name, "model": key, "corruption": kind, "severity": severity,
                         "accuracy": m["accuracy"], "macro_f1": m["macro_f1"]})
        acc = {k: np.mean([r["accuracy"] for r in rows[-len(models):] if r["model"] == k]) for k in keys}
        log(f"  {kind:15s} s{severity}: " + "  ".join(f"{k} {v:.4f}" for k, v in acc.items()))
    return pd.DataFrame(rows)


def summarise_robustness(df: pd.DataFrame) -> pd.DataFrame:
    """Per model and corruption: mean +- std over seeds of accuracy, averaged over severities,
    plus the drop from clean accuracy in percentage points."""
    clean = df[df["corruption"] == "clean"].set_index("run")["accuracy"]
    d = df[df["corruption"] != "clean"].copy()
    d["drop_pp"] = (d["run"].map(clean) - d["accuracy"]) * 100
    per_run = d.groupby(["model", "corruption", "run"], as_index=False)[["accuracy", "drop_pp"]].mean()
    out = per_run.groupby(["model", "corruption"]).agg(
        accuracy_mean=("accuracy", "mean"), accuracy_std=("accuracy", "std"),
        drop_pp_mean=("drop_pp", "mean"), drop_pp_std=("drop_pp", "std")).reset_index()
    return out.fillna(0.0)
