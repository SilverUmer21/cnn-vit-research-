"""Figures for the paper. Each function saves a PDF (for LaTeX) and a PNG (for the notebook) and
returns the matplotlib figure. Sizes match an IEEE two-column page (3.5 in / 7.16 in wide).

Colour: one fixed colour per model everywhere (identity never changes between figures); a single
blue ramp for magnitudes (confusion matrices); text stays black/grey, never the series colour."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

from .analysis import class_label, display, load_history
from .config import MODEL_ORDER
from .data import CORRUPTIONS

COLORS = {"mobilenetv2": "#2a78d6", "mobilevit_xs": "#eb6834"}
SPLIT_COLORS = {"train": "#2a78d6", "val": "#eb6834", "test": "#1baf7a"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SINGLE, DOUBLE = 3.5, 7.16
BLUES = LinearSegmentedColormap.from_list("blues", ["#ffffff", "#cfe0f6", "#7fb0ec", "#2a78d6", "#0d3f80"])


def style():
    plt.rcParams.update({
        "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7,
        "ytick.labelsize": 7, "legend.fontsize": 7, "axes.edgecolor": INK2, "axes.labelcolor": INK,
        "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.axisbelow": True, "lines.linewidth": 1.5, "legend.frameon": False,
        "savefig.dpi": 300, "savefig.bbox": "tight", "pdf.fonttype": 42, "font.family": "DejaVu Sans",
    })


style()


def save(fig, out_dir: Path, name: str):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.pdf")
    fig.savefig(out_dir / f"{name}.png", dpi=200)
    return fig


def _load_image(data_root, rel_path, size=256):
    with Image.open(Path(data_root) / rel_path) as im:
        return np.asarray(im.convert("RGB").resize((size, size)), dtype=np.float32) / 255.0


# ---------------------------------------------------------------------------------------------
# data preparation
# ---------------------------------------------------------------------------------------------
def class_distribution(manifest: pd.DataFrame, out_dir):
    counts = manifest.groupby(["label", "split"]).size().unstack(fill_value=0)[["train", "val", "test"]]
    counts = counts.loc[counts.sum(1).sort_values().index]
    fig, ax = plt.subplots(figsize=(DOUBLE, 6.2))
    left = np.zeros(len(counts))
    y = np.arange(len(counts))
    for s in ("train", "val", "test"):
        ax.barh(y, counts[s], left=left, height=0.72, color=SPLIT_COLORS[s], edgecolor="white",
                linewidth=1.0, label={"train": "Train", "val": "Validation", "test": "Test"}[s])
        left += counts[s].to_numpy()
    for yi, total in zip(y, left):
        ax.text(total + 40, yi, f"{int(total):,}", va="center", fontsize=6, color=INK2)
    ax.set_yticks(y, [class_label(c) for c in counts.index], fontsize=6)
    ax.set_xlabel("Images")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", ncols=3)
    ax.set_xlim(0, left.max() * 1.1)
    ax.set_title("Images per class and split (80/10/10, grouped and stratified)", loc="left")
    return save(fig, out_dir, "data_class_distribution")


def sample_grid(manifest: pd.DataFrame, data_root, out_dir, seed=42):
    """One training image per class."""
    rng = np.random.default_rng(seed)
    classes = sorted(manifest["label"].unique())
    cols = 8
    rows = int(np.ceil(len(classes) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(DOUBLE, rows * 1.08))
    for ax in axes.flat:
        ax.axis("off")
    for ax, c in zip(axes.flat, classes):
        paths = manifest[(manifest["label"] == c) & (manifest["split"] == "train")]["path"].to_numpy()
        ax.imshow(_load_image(data_root, paths[rng.integers(len(paths))]))
        crop, _, disease = class_label(c).partition(": ")
        ax.set_title(f"{crop}\n{disease}", fontsize=5, pad=2)
    fig.subplots_adjust(wspace=0.05, hspace=0.45)
    return save(fig, out_dir, "data_samples")


# ---------------------------------------------------------------------------------------------
# E1: learning curves
# ---------------------------------------------------------------------------------------------
def learning_curves(paths, runs: pd.DataFrame, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE, 2.3))
    panels = [("train_loss", "val_loss", "Cross-entropy loss"), ("val_acc", None, "Validation accuracy"),
              ("lr_end", None, "Learning rate")]
    for m in [k for k in MODEL_ORDER if k in set(runs["model"])]:
        hists = [load_history(paths, r) for r in runs[runs["model"] == m]["run"]]
        for ax, (k1, k2, title) in zip(axes, panels):
            for key, ls in [(k1, "-"), (k2, "--")]:
                if key is None:
                    continue
                h = pd.concat(hists).groupby("epoch")[key]
                mean, lo, hi = h.mean(), h.min(), h.max()
                ax.plot(mean.index, mean, ls, color=COLORS[m], marker="o", ms=2.5,
                        label=f"{display(m)} {'train' if key == 'train_loss' else 'val' if key == 'val_loss' else ''}".strip())
                if len(hists) > 1:
                    ax.fill_between(mean.index, lo, hi, color=COLORS[m], alpha=0.15, linewidth=0)
            ax.set_title(title, loc="left")
            ax.set_xlabel("Epoch")
            best = [int(h.loc[h["val_loss"].idxmin(), "epoch"]) for h in hists]
            if k1 == "val_acc":
                for b in best:
                    ax.axvline(b, color=COLORS[m], lw=0.6, alpha=0.5, ls=":")
    for ax in axes:
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    axes[0].set_yscale("log")
    axes[0].legend(fontsize=6)
    axes[1].legend([plt.Line2D([], [], color=COLORS[m]) for m in MODEL_ORDER],
                   [display(m) for m in MODEL_ORDER], fontsize=6, loc="lower right")
    axes[2].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    fig.tight_layout()
    return save(fig, out_dir, "e1_learning_curves")


# ---------------------------------------------------------------------------------------------
# E1/E4: per-class F1 and confusion
# ---------------------------------------------------------------------------------------------
def per_class_f1(pc: pd.DataFrame, out_dir):
    a, b = MODEL_ORDER
    pc = pc.sort_values(f"f1_{a}", ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(DOUBLE, 6.0))
    y = np.arange(len(pc))
    for m, off in [(a, -0.17), (b, 0.17)]:
        ax.errorbar(100 * pc[f"f1_{m}"], y + off, xerr=100 * pc[f"f1_std_{m}"], fmt="o", ms=3.5,
                    color=COLORS[m], elinewidth=0.8, capsize=0, label=display(m))
    ax.set_yticks(y, [f"{class_label(c)} (n={int(s)})" for c, s in zip(pc["class"], pc["support"])], fontsize=6)
    ax.invert_yaxis()
    lo = max(0, 100 * min(pc[f"f1_{a}"].min(), pc[f"f1_{b}"].min()) - 2)
    ax.set_xlim(lo, 100.5)
    ax.set_xlabel("Test F1 (%), mean over seeds; bars: ±1 std")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower left")
    ax.set_title("Per-class F1", loc="left")
    return save(fig, out_dir, "e1_per_class_f1")


def confusion_errors(cm: np.ndarray, class_names, model_key, out_dir):
    """Off-diagonal confusion counts (mean per run); the diagonal is left blank so errors are visible."""
    off = cm.astype(float).copy()
    np.fill_diagonal(off, np.nan)
    fig, ax = plt.subplots(figsize=(DOUBLE * 0.82, DOUBLE * 0.78))
    vmax = max(np.nanmax(off), 1)
    im = ax.imshow(np.where(off > 0, off, np.nan), cmap=BLUES, vmin=0, vmax=vmax)
    ax.set_facecolor("white")
    n = len(class_names)
    ticks = [f"{i + 1} {class_label(c)}" for i, c in enumerate(class_names)]
    ax.set_xticks(range(n), [str(i + 1) for i in range(n)], fontsize=5)
    ax.set_yticks(range(n), ticks, fontsize=5)
    ax.set_xlabel("Predicted class (number)")
    ax.set_ylabel("True class")
    ax.grid(False)
    for i in range(n):
        ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, color="#f1f0ec", lw=0))
    for i, j in zip(*np.where(off >= max(1.0, vmax * 0.25))):
        ax.text(j, i, f"{off[i, j]:.0f}" if off[i, j] >= 10 else f"{off[i, j]:.1f}", ha="center",
                va="center", fontsize=4.5, color="white" if off[i, j] > vmax * 0.6 else INK)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("Misclassified test images (mean per run)", fontsize=6)
    ax.set_title(f"{display(model_key)}: test-set errors (diagonal = correct, hidden)", loc="left")
    return save(fig, out_dir, f"e4_confusion_{model_key}")


# ---------------------------------------------------------------------------------------------
# E3: accuracy vs cost
# ---------------------------------------------------------------------------------------------
def tradeoff(main: pd.DataFrame, dep: pd.DataFrame, out_dir):
    panels = [("params_m", "Parameters (M)"), ("gmacs", "GMACs per image"),
              ("cpu1_ms", "CPU latency, 1 thread (ms)")]
    if "gpu_ms" in dep.columns:
        panels.append(("gpu_ms", "GPU latency, FP32 (ms)"))
    M, D = main.set_index("model"), dep.set_index("model")
    fig, axes = plt.subplots(1, len(panels), figsize=(DOUBLE, 2.0), sharey=True)
    for ax, (col, label) in zip(axes, panels):
        xs = []
        for m in MODEL_ORDER:
            if m not in M.index or m not in D.index:
                continue
            x, yv, ye = D.at[m, col], M.at[m, "accuracy_mean"], M.at[m, "accuracy_std"]
            xs.append(x)
            ax.errorbar(x, yv, yerr=ye, fmt="o", ms=6, color=COLORS[m], capsize=2, elinewidth=1,
                        mec="white", mew=1)
            ax.annotate(display(m), (x, yv), textcoords="offset points", xytext=(0, 7), ha="center",
                        fontsize=6, color=INK2)
        ax.set_xlabel(label)
        if xs:
            pad = (max(xs) - min(xs)) * 0.35 or max(xs) * 0.2
            ax.set_xlim(max(0, min(xs) - pad), max(xs) + pad)
    axes[0].set_ylabel("Test accuracy (%)")
    ymin = (M["accuracy_mean"] - M["accuracy_std"]).min()
    ymax = (M["accuracy_mean"] + M["accuracy_std"]).max()
    span = max(ymax - ymin, 0.2)
    axes[0].set_ylim(ymin - span * 0.6, ymax + span * 0.9)
    fig.suptitle("Accuracy against deployment cost (mean ± std over seeds)", x=0.01, ha="left", fontsize=8.5)
    fig.tight_layout()
    return save(fig, out_dir, "e3_tradeoff")


# ---------------------------------------------------------------------------------------------
# E5: robustness
# ---------------------------------------------------------------------------------------------
def robustness(df: pd.DataFrame, out_dir):
    kinds = [k for k in CORRUPTIONS if k in set(df["corruption"])]
    cols = 3
    rows = int(np.ceil(len(kinds) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(DOUBLE, 1.9 * rows), sharey=True)
    clean = df[df["corruption"] == "clean"]
    for ax, k in zip(axes.flat, kinds):
        for m in MODEL_ORDER:
            d = pd.concat([clean[clean["model"] == m], df[(df["corruption"] == k) & (df["model"] == m)]])
            g = d.groupby("severity")["accuracy"]
            mean, std = 100 * g.mean(), 100 * g.std().fillna(0)
            ax.plot(mean.index, mean, "-o", ms=3.5, color=COLORS[m], label=display(m))
            ax.fill_between(mean.index, mean - std, mean + std, color=COLORS[m], alpha=0.15, lw=0)
        levels = ", ".join(str(v) for v in CORRUPTIONS[k])
        ax.set_title(f"{k.replace('_', ' ').capitalize()} ({levels})", loc="left", fontsize=7)
        ax.set_xticks([0, 1, 2, 3], ["clean", "1", "2", "3"])
    for ax in axes.flat[len(kinds):]:
        ax.axis("off")
    for ax in axes[:, 0] if rows > 1 else [axes[0]]:
        ax.set_ylabel("Test accuracy (%)")
    for ax in (axes[-1] if rows > 1 else axes):
        ax.set_xlabel("Severity")
    axes.flat[0].legend(loc="lower left", fontsize=6)
    fig.tight_layout()
    return save(fig, out_dir, "e5_robustness")


# ---------------------------------------------------------------------------------------------
# E4: qualitative examples and Grad-CAM
# ---------------------------------------------------------------------------------------------
def example_grid(items, data_root, out_dir, name, title, cols=4):
    """items: list of dicts with path, true, lines (list of (text, model_key))."""
    import textwrap
    rows = int(np.ceil(len(items) / cols)) or 1
    fig, axes = plt.subplots(rows, cols, figsize=(DOUBLE, rows * 2.45))
    axes = np.atleast_1d(axes).reshape(rows, cols)
    for ax in axes.flat:
        ax.axis("off")
    for ax, it in zip(axes.flat, items):
        ax.imshow(_load_image(data_root, it["path"]))
        ax.set_title(textwrap.fill("True: " + class_label(it["true"]), 34), fontsize=5.5, pad=2)
        text = "\n".join(textwrap.fill(t.replace("MobileNetV2", "MNv2").replace("MobileViT-XS", "MViT"), 36)
                         for t, _ in it["lines"])
        ax.text(0.5, -0.04, text, transform=ax.transAxes, ha="center", va="top", fontsize=5.2, color=INK,
                linespacing=1.25)
    fig.suptitle(title + "  (MNv2 = MobileNetV2, MViT = MobileViT-XS; confidence in brackets)",
                 x=0.01, ha="left", fontsize=8)
    fig.subplots_adjust(wspace=0.15, hspace=0.75, top=0.9)
    return save(fig, out_dir, name)


def gradcam_grid(rows, out_dir, name="e4_gradcam"):
    """rows: list of dicts with image (HWC [0,1]), true, and per-model (overlay, pred, prob)."""
    import textwrap

    from .gradcam import overlay
    n = len(rows)
    fig, axes = plt.subplots(n, 3, figsize=(SINGLE * 1.3, n * 1.45))
    axes = np.atleast_2d(axes)
    for r, row in enumerate(rows):
        axes[r, 0].imshow(row["image"])
        axes[r, 0].set_title("True:\n" + textwrap.fill(class_label(row["true"]), 30), fontsize=5, pad=2)
        for c, m in enumerate(MODEL_ORDER, start=1):
            cam, pred, prob = row[m]
            axes[r, c].imshow(overlay(row["image"], cam))
            ok = "correct" if pred == row["true"] else "wrong"
            axes[r, c].set_title(f"{display(m)} ({ok}, p={prob:.2f})\n" + textwrap.fill(class_label(pred), 30),
                                 fontsize=5, pad=2, color=INK)
        for ax in axes[r]:
            ax.axis("off")
    fig.suptitle("Grad-CAM: regions that raised the predicted class score", x=0.01, ha="left", fontsize=8)
    fig.subplots_adjust(wspace=0.04, hspace=0.55, top=0.93)
    return save(fig, out_dir, name)
