"""Predictions, metrics and statistical comparison of the two models.

Metrics follow the proposal (Section IV-E): accuracy and macro-averaged F1 (each of the 38 classes
weighted equally). Macro precision and recall are reported as well, since the guidance asks for
all relevant metrics and they explain where F1 comes from."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from scipy.stats import binomtest
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_recall_fscore_support)


@torch.inference_mode()
def predict(model, loader, prep, device, amp=False, max_batches=None):
    """Returns (logits float32 [N, C], labels int64 [N], mean cross-entropy loss).
    `prep` turns the loader's uint8 images into model input (data.Preprocess)."""
    model.eval()
    logits_all, labels_all = [], []
    loss_sum, n = 0.0, 0
    use_amp = amp and device == "cuda"
    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x = prep(x.to(device, non_blocking=True), train=False)
        y = y.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            logits = model(x)
        logits = logits.float()
        loss_sum += torch.nn.functional.cross_entropy(logits, y, reduction="sum").item()
        n += y.numel()
        logits_all.append(logits.cpu())
        labels_all.append(y.cpu())
    logits = torch.cat(logits_all).numpy().astype(np.float32)
    labels = torch.cat(labels_all).numpy().astype(np.int64)
    return logits, labels, loss_sum / max(n, 1)


def classification_metrics(labels, preds, num_classes):
    """Overall metrics as a flat dict (fractions, not percentages)."""
    p, r, f1, _ = precision_recall_fscore_support(
        labels, preds, labels=np.arange(num_classes), average="macro", zero_division=0)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_precision": float(p),
        "macro_recall": float(r),
        "macro_f1": float(f1),
        "n": int(len(labels)),
        "errors": int((labels != preds).sum()),
    }


def per_class_metrics(labels, preds, class_names):
    p, r, f1, support = precision_recall_fscore_support(
        labels, preds, labels=np.arange(len(class_names)), average=None, zero_division=0)
    return pd.DataFrame({"class": class_names, "precision": p, "recall": r, "f1": f1,
                         "support": support})


def confusion(labels, preds, num_classes, normalize=None):
    return confusion_matrix(labels, preds, labels=np.arange(num_classes), normalize=normalize)


def top_confusions(labels, preds, class_names, k=10):
    cm = confusion(labels, preds, len(class_names))
    np.fill_diagonal(cm, 0)
    idx = np.dstack(np.unravel_index(np.argsort(-cm, axis=None), cm.shape))[0][:k]
    rows = [{"true": class_names[i], "predicted": class_names[j], "count": int(cm[i, j]),
             "share_of_true_class": float(cm[i, j] / max((labels == i).sum(), 1))}
            for i, j in idx if cm[i, j] > 0]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------
# aggregation over seeds and paired statistics
# ---------------------------------------------------------------------------------------------
def mean_std(values):
    """Mean and sample standard deviation (ddof=1; 0 when only one value)."""
    a = np.asarray(values, dtype=float)
    return float(a.mean()), float(a.std(ddof=1)) if a.size > 1 else 0.0


def mcnemar_exact(labels, pred_a, pred_b):
    """Exact McNemar test on the same test images. b = A right & B wrong, c = A wrong & B right."""
    a_ok, b_ok = pred_a == labels, pred_b == labels
    b = int((a_ok & ~b_ok).sum())
    c = int((~a_ok & b_ok).sum())
    p = 1.0 if b + c == 0 else float(binomtest(b, b + c, 0.5).pvalue)
    return {"a_right_b_wrong": b, "a_wrong_b_right": c, "p_value": p}


def _macro_f1_from_counts(tp, fp, fn):
    denom = 2 * tp + fp + fn
    f1 = np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)
    return f1.mean(axis=-1)


def paired_bootstrap(labels, pred_a, pred_b, num_classes, n_boot=2000, seed=0):
    """95% percentile bootstrap CIs for the differences (B - A) in accuracy and macro-F1, resampling
    test images with replacement (same resample for both models)."""
    rng = np.random.default_rng(seed)
    n = len(labels)
    acc_d, f1_d = np.empty(n_boot), np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        y, a, b = labels[idx], pred_a[idx], pred_b[idx]
        acc_d[i] = (b == y).mean() - (a == y).mean()
        f1s = []
        for p in (a, b):
            tp = np.bincount(y[p == y], minlength=num_classes)
            pred_count = np.bincount(p, minlength=num_classes)
            true_count = np.bincount(y, minlength=num_classes)
            f1s.append(_macro_f1_from_counts(tp, pred_count - tp, true_count - tp))
        f1_d[i] = f1s[1] - f1s[0]
    ci = lambda d: (float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)))
    return {"accuracy_diff_ci95": ci(acc_d), "macro_f1_diff_ci95": ci(f1_d)}
