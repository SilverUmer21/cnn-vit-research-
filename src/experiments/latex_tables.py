"""LaTeX versions of every results table. Each function returns the table source; the notebook writes
them to results/tables/*.tex and the paper \\input{}s them."""
from __future__ import annotations

import pandas as pd

from .analysis import METRICS, class_label, display, latex_table, pm, tex_escape
from .config import MODEL_ORDER

DATASET_NOTE = "Test set: 5{,}431 images, 38 classes (fixed split, seed 42)."


def dataset_split(manifest: pd.DataFrame):
    rows = []
    n = len(manifest)
    for s in ("train", "val", "test"):
        m = manifest[manifest["split"] == s]
        rows.append([s.capitalize() if s != "val" else "Validation", f"{len(m):,}".replace(",", "{,}"),
                     f"{100 * len(m) / n:.1f}", f"{m['group_id'].nunique():,}".replace(",", "{,}"),
                     f"{m.groupby('label').size().min()}--{m.groupby('label').size().max()}"])
    rows.append(["Total", f"{n:,}".replace(",", "{,}"), "100.0",
                 f"{manifest['group_id'].nunique():,}".replace(",", "{,}"), ""])
    return latex_table(["Split", "Images", "Share (\\%)", "Duplicate groups", "Images per class"], rows,
                       "Dataset split used by every experiment (stratified by class, near-duplicate "
                       "groups kept within one split).", "tab:split")


def main_results(main: pd.DataFrame, n_test=5431):
    best = {k: main[f"{k}_mean"].max() for k, _ in METRICS}
    rows = []
    for _, r in main.iterrows():
        cells = [display(r["model"])]
        cells += [pm(r[f"{k}_mean"], r[f"{k}_std"], 2, bold=r[f"{k}_mean"] == best[k]) for k, _ in METRICS]
        cells.append(pm(r["errors_mean"], r["errors_std"], 1))
        rows.append(cells)
    seeds = int(main["seeds"].min())
    return latex_table(
        ["Model"] + [f"{n} (\\%)" for _, n in METRICS] + [f"Errors (of {n_test:,})".replace(",", "{,}")],
        rows,
        f"Test-set classification performance (E1), mean $\\pm$ standard deviation over {seeds} "
        "training seeds. Best mean per column in bold.", "tab:main", wide=True)


def per_seed(per_seed_df: pd.DataFrame):
    rows = [[display(r["model"]), str(r["seed"]), f"{100 * r['accuracy']:.2f}", f"{100 * r['macro_f1']:.2f}",
             str(int(r["errors"])), str(int(r["best_epoch"])), str(int(r["epochs_run"]))]
            for _, r in per_seed_df.iterrows()]
    return latex_table(["Model", "Seed", "Accuracy (\\%)", "Macro F1 (\\%)", "Errors", "Best epoch",
                        "Epochs run"], rows,
                       "Test results of every training run. Best epoch = epoch with the lowest validation "
                       "loss (its weights are evaluated); training stopped after 5 epochs without improvement "
                       "or at 20 epochs.", "tab:perseed")


def statistics(stats: pd.DataFrame):
    rows = []
    for _, r in stats.iterrows():
        p = r["mcnemar_p"]
        rows.append([str(r["seed"]), f"{r['acc_diff_pp']:+.2f}",
                     f"[{r['acc_ci_low_pp']:+.2f}, {r['acc_ci_high_pp']:+.2f}]",
                     f"[{r['f1_ci_low_pp']:+.2f}, {r['f1_ci_high_pp']:+.2f}]",
                     str(int(r["only_mobilenetv2_right"])), str(int(r["only_mobilevit_right"])),
                     f"{p:.3f}" if p >= 0.001 else "$<$0.001"])
    return latex_table(["Seed", "$\\Delta$Acc (pp)", "95\\% CI $\\Delta$Acc", "95\\% CI $\\Delta$F1",
                        "Only MNv2 right", "Only MViT right", "McNemar $p$"], rows,
                       "Paired comparison on the same test images, MobileViT-XS minus MobileNetV2 (pp = "
                       "percentage points). Confidence intervals: 2{,}000 paired bootstrap resamples. "
                       "McNemar: exact two-sided test on the images only one model classified correctly.",
                       "tab:stats", wide=True)


def training_cost(cost: pd.DataFrame):
    rows = [[display(r["model"]), pm(r["epochs_run_mean"], r["epochs_run_std"], 1),
             pm(r["best_epoch_mean"], r["best_epoch_std"], 1),
             pm(r["mean_epoch_train_time_s_mean"] / 60, r["mean_epoch_train_time_s_std"] / 60, 2),
             pm(r["total_train_time_s_mean"] / 60, r["total_train_time_s_std"] / 60, 1),
             pm(r["mean_train_images_per_s_mean"], r["mean_train_images_per_s_std"], 0),
             pm(r["peak_train_mem_mib_mean"], r["peak_train_mem_mib_std"], 0)]
            for _, r in cost.iterrows()]
    gpus = tex_escape(", ".join(sorted(set(", ".join(cost["gpus"]).split(", ")))))
    return latex_table(["Model", "Epochs run", "Best epoch", "Min / epoch", "Total min",
                        "Train img/s", "Peak mem. (MiB)"], rows,
                       f"Training cost (mean $\\pm$ std over seeds; batch 32, FP16 mixed precision, "
                       f"GPU: {gpus}). Total time includes validation after every epoch.", "tab:traincost",
                       wide=True)


def deployability(dep: pd.DataFrame):
    has_gpu = "gpu_ms" in dep.columns
    header = ["Model", "Input", "Params (M)", "Size (MB)", "GMACs", "CPU 1 thr. (ms)", "CPU all thr. (ms)"]
    if has_gpu:
        header += ["GPU FP32 (ms)", "GPU FP16 (ms)", "GPU img/s", "GPU mem. (MiB)"]
    low = {c: dep[c].min() for c in ("params_m", "size_mb_fp32", "gmacs", "cpu1_ms", "cpuall_ms",
                                     "gpu_ms", "gpu16_ms", "mem_bs1_mib") if c in dep}
    high_tp = dep["throughput"].max() if has_gpu else None

    def b(v, col, s):
        return f"\\textbf{{{s}}}" if col in low and v == low[col] else s

    rows = []
    for _, r in dep.iterrows():
        cells = [display(r["model"]), f"{r['input']}$^2$", b(r["params_m"], "params_m", f"{r['params_m']:.2f}"),
                 b(r["size_mb_fp32"], "size_mb_fp32", f"{r['size_mb_fp32']:.2f}"),
                 b(r["gmacs"], "gmacs", f"{r['gmacs']:.3f}"),
                 b(r["cpu1_ms"], "cpu1_ms", pm(r["cpu1_ms"], r["cpu1_std"], 1)),
                 b(r["cpuall_ms"], "cpuall_ms", pm(r["cpuall_ms"], r["cpuall_std"], 1))]
        if has_gpu:
            cells += [b(r["gpu_ms"], "gpu_ms", pm(r["gpu_ms"], r["gpu_std"], 2)),
                      b(r["gpu16_ms"], "gpu16_ms", pm(r["gpu16_ms"], r["gpu16_std"], 2)),
                      f"\\textbf{{{r['throughput']:.0f}}}" if r["throughput"] == high_tp else f"{r['throughput']:.0f}",
                      b(r["mem_bs1_mib"], "mem_bs1_mib", f"{r['mem_bs1_mib']:.1f}")]
        rows.append(cells)
    hw = f" on an NVIDIA {tex_escape(dep['gpu_name'].iloc[0])}" if has_gpu else ""
    note = ("Latency: batch size 1, mean $\\pm$ std of 100 timed forward passes after 10 warm-up passes, "
            f"preprocessing excluded{hw}; CPU on the same machine "
            f"({int(dep['cpu_threads_all'].iloc[0])} threads for ``all''). Size: saved FP32 weights. "
            "MACs: one forward pass of one image at the model's input size (FLOPs = 2 $\\times$ MACs). "
            "Throughput: batch 32, FP32. GPU memory: peak extra memory for one image. Best value in bold.")
    return latex_table(header, rows, "Deployability of the two models (E2).", "tab:deploy",
                       wide=True, note=note)


def tradeoff(tr: pd.DataFrame):
    rows = []
    for _, r in tr.iterrows():
        v = r["mobilevit_vs_mobilenetv2"]
        rows.append([r["quantity"], f"{v:+.2f} pp" if r["kind"] == "difference" else f"{v:.2f}$\\times$"])
    return latex_table(["Quantity", "MobileViT-XS vs MobileNetV2"], rows,
                       "Accuracy gained versus cost added by MobileViT-XS (E3). Differences in percentage "
                       "points (pp, mean over seeds); costs as ratios (values above 1 mean MobileViT-XS is "
                       "larger or slower, except throughput where above 1 means faster).", "tab:tradeoff")


def per_class(pc: pd.DataFrame, top=None):
    a, b = MODEL_ORDER
    pc = pc.sort_values("diff") if "diff" in pc else pc
    if top:
        pc = pd.concat([pc.head(top), pc.tail(top)]).drop_duplicates("class")
    rows = [[tex_escape(class_label(r["class"])), str(int(r["support"])),
             f"{100 * r[f'f1_{a}']:.2f}", f"{100 * r[f'f1_{b}']:.2f}", f"{100 * r['diff']:+.2f}"]
            for _, r in pc.iterrows()]
    return latex_table(["Class", "Test images", "F1 MNv2 (\\%)", "F1 MViT (\\%)", "$\\Delta$ (pp)"], rows,
                       "Per-class test F1 (mean over seeds), sorted by the difference MobileViT-XS minus "
                       "MobileNetV2. Classes with fewer than 40 test images are noisy: one error moves F1 by "
                       "several points.", "tab:perclass", align="lrrrr", size="\\scriptsize")


def top_confusions(tc: pd.DataFrame):
    rows = [[display(r["model"]), tex_escape(class_label(r["true"])), tex_escape(class_label(r["predicted"])),
             f"{r['mean_count']:.1f}", str(r["class_size"])] for _, r in tc.iterrows()]
    return latex_table(["Model", "True class", "Predicted as", "Errors / run", "Class size"], rows,
                       "Most frequent confusions on the test set (mean count per training run).",
                       "tab:confusions", align="lllrr", size="\\scriptsize", wide=True)


def robustness(summary: pd.DataFrame, clean: dict):
    a, b = MODEL_ORDER
    s = summary.set_index(["corruption", "model"])
    rows = [["Clean", pm(100 * clean[a][0], 100 * clean[a][1]), pm(100 * clean[b][0], 100 * clean[b][1]), "--", "--"]]
    for c in summary["corruption"].unique():
        ra, rb = s.loc[(c, a)], s.loc[(c, b)]
        rows.append([c.replace("_", " ").capitalize(),
                     pm(100 * ra["accuracy_mean"], 100 * ra["accuracy_std"]),
                     pm(100 * rb["accuracy_mean"], 100 * rb["accuracy_std"]),
                     f"{ra['drop_pp_mean']:.2f}", f"{rb['drop_pp_mean']:.2f}"])
    mean_a = summary[summary["model"] == a]["accuracy_mean"].mean()
    mean_b = summary[summary["model"] == b]["accuracy_mean"].mean()
    rows.append(["Mean over corruptions", f"{100 * mean_a:.2f}", f"{100 * mean_b:.2f}",
                 f"{summary[summary['model'] == a]['drop_pp_mean'].mean():.2f}",
                 f"{summary[summary['model'] == b]['drop_pp_mean'].mean():.2f}"])
    return latex_table(["Corruption", "Acc. MNv2 (\\%)", "Acc. MViT (\\%)", "Drop MNv2 (pp)", "Drop MViT (pp)"],
                       rows, "Robustness test (E5): test accuracy under synthetic image corruptions, averaged "
                       "over three severities, mean $\\pm$ std over seeds; drop = clean minus corrupted accuracy.",
                       "tab:robust")


def ablation(ab: pd.DataFrame):
    rows = [[display(r["model"]), f"{r['native_acc']:.2f}", f"{r['imagenet_acc']:.2f}", f"{r['acc_diff_pp']:+.2f}",
             f"{r['native_f1']:.2f}", f"{r['imagenet_f1']:.2f}", f"{r['f1_diff_pp']:+.2f}"]
            for _, r in ab.iterrows()]
    return latex_table(["Model", "Acc. native", "Acc. ImageNet", "$\\Delta$Acc", "F1 native", "F1 ImageNet",
                        "$\\Delta$F1"], rows,
                       "Preprocessing ablation (E6, seed 42): each model's own pretrained preprocessing "
                       "(native) versus the ImageNet mean/std in RGB order stated in the original proposal. "
                       "Values in \\%; $\\Delta$ in pp (ImageNet minus native).", "tab:ablation", wide=True)
