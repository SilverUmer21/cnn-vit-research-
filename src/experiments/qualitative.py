"""Picking the images shown in the qualitative figures (E4). Selection is deterministic (seed) and
rule-based, so the figures are representative rather than hand-picked:
- correct: random correctly classified test images, one per randomly chosen class;
- failures: each model's misclassifications, most confident first (confident mistakes are the
  most informative ones);
- disagreements: images exactly one of the two models gets right.
All use the seed-42 run of each model, so every image is classified by matching runs."""
from __future__ import annotations

import numpy as np

from .config import MODEL_ORDER
from .data import make_transform
from .gradcam import GradCAM


def _softmax(logits):
    z = logits - logits.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def pair_for_seed(preds, seed):
    d = {p["model"]: p for p in preds.values() if p["seed"] == seed}
    if not all(m in d for m in MODEL_ORDER):
        raise ValueError(f"seed {seed}: both models must have finished")
    return d


def _line(model_key, names, pred, prob):
    from .analysis import class_label, display
    return (f"{display(model_key)}: {class_label(names[pred])} ({prob:.2f})", model_key)


def select_examples(pair, kind, k=12, seed=0):
    a, b = (pair[m] for m in MODEL_ORDER)
    y, names, paths = a["labels"], a["class_names"], a["paths"]
    pa, pb = _softmax(a["logits"]), _softmax(b["logits"])
    ya, yb = pa.argmax(1), pb.argmax(1)
    rng = np.random.default_rng(seed)
    if kind == "correct":
        ok = np.where((ya == y) & (yb == y))[0]
        classes = rng.permutation(np.unique(y[ok]))[:k]
        idx = [rng.choice(ok[y[ok] == c]) for c in classes]
    elif kind in MODEL_ORDER:
        p, yp = (pa, ya) if kind == MODEL_ORDER[0] else (pb, yb)
        wrong = np.where(yp != y)[0]
        idx = wrong[np.argsort(-p[wrong, yp[wrong]])][:k]
    elif kind == "disagree":
        only_a = np.where((ya == y) & (yb != y))[0]
        only_b = np.where((ya != y) & (yb == y))[0]
        half = k // 2
        idx = list(rng.permutation(only_a)[:half]) + list(rng.permutation(only_b)[:k - min(half, len(only_a))])
        idx = idx[:k]
    else:
        raise ValueError(kind)
    items = []
    for i in idx:
        items.append({"index": int(i), "path": str(paths[i]), "true": names[y[i]],
                      "lines": [_line(MODEL_ORDER[0], names, ya[i], pa[i, ya[i]]),
                                _line(MODEL_ORDER[1], names, yb[i], pb[i, yb[i]])]})
    return items


def gradcam_rows(models, pair, indices, data_root, device="cpu"):
    """models: {model_key: trained Classifier}. Returns rows for plots.gradcam_grid."""
    from PIL import Image
    names, paths, labels = pair[MODEL_ORDER[0]]["class_names"], pair[MODEL_ORDER[0]]["paths"], pair[MODEL_ORDER[0]]["labels"]
    cams = {m: GradCAM(models[m].to(device)) for m in MODEL_ORDER}
    rows = []
    try:
        for i in indices:
            with Image.open(f"{data_root}/{paths[i]}") as im:
                im = im.convert("RGB")
            row = {"image": np.asarray(im.resize((256, 256)), dtype=np.float32) / 255.0,
                   "true": names[labels[i]]}
            for m in MODEL_ORDER:
                x = make_transform(m, train=False)(im)[None].to(device)
                cam, c, prob = cams[m](x)
                row[m] = (cam, names[c], prob)
            rows.append(row)
    finally:
        for c in cams.values():
            c.remove()
    return rows


def gradcam_indices(pair, n_correct=3, n_wrong=3, seed=0):
    """Mix of images both models get right and images at least one model gets wrong."""
    a, b = (pair[m] for m in MODEL_ORDER)
    y = a["labels"]
    ya, yb = a["logits"].argmax(1), b["logits"].argmax(1)
    rng = np.random.default_rng(seed)
    both = np.where((ya == y) & (yb == y))[0]
    some_wrong = np.where((ya != y) | (yb != y))[0]
    picked = list(rng.permutation(both)[:n_correct]) + list(rng.permutation(some_wrong)[:n_wrong])
    return [int(i) for i in picked]


def load_best(paths, run, model_key, class_names, device="cpu"):
    """Rebuild the architecture (no download of new weights needed beyond the config) and load the
    fine-tuned weights from best.pt."""
    from . import storage
    from .models import build_model
    model = build_model(model_key, class_names, pretrained=False)
    model.load_state_dict(storage.load_torch(paths.run_dir(run) / "best.pt"))
    return model.to(device).eval()

