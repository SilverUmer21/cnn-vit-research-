"""Collect finished runs and turn them into the results tables (CSV for the notebook, LaTeX for the
paper). Every number in the paper comes from here, so nothing is copied by hand."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import storage
from .config import MODEL_ORDER, MODELS
from .evaluate import (mcnemar_exact, mean_std, paired_bootstrap, per_class_metrics)

METRICS = [("accuracy", "Accuracy"), ("macro_precision", "Macro precision"),
           ("macro_recall", "Macro recall"), ("macro_f1", "Macro F1")]


def display(model_key):
    return MODELS[model_key]["display"]


# ---------------------------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------------------------
def load_runs(paths, variant="native"):
    """One row per finished run, plus its test predictions."""
    rows, preds = [], {}
    for rd in sorted(paths.runs.glob(f"*_{variant}_s*")):
        if not (rd / "DONE").exists():
            continue
        s = storage.load_json(rd / "summary.json")
        rows.append({**{k: v for k, v in s.items() if k != "test"}, **s["test"]})
        z = np.load(rd / "test_predictions.npz", allow_pickle=False)
        preds[s["run"]] = {"model": s["model"], "seed": s["seed"], "logits": z["logits"],
                           "labels": z["labels"], "paths": z["paths"],
                           "class_names": [str(c) for c in z["class_names"]]}
    df = pd.DataFrame(rows)
    if not df.empty:
        df["order"] = df["model"].map(MODEL_ORDER.index)
        df = df.sort_values(["order", "seed"]).drop(columns="order").reset_index(drop=True)
    return df, preds


def load_history(paths, run):
    return pd.read_csv(paths.run_dir(run) / "history.csv")


# ---------------------------------------------------------------------------------------------
# tables (as DataFrames)
# ---------------------------------------------------------------------------------------------
def per_seed_table(runs):
    cols = ["model", "seed", "accuracy", "macro_precision", "macro_recall", "macro_f1", "errors",
            "best_epoch", "epochs_run"]
    return runs[cols].copy()


def main_results(runs):
    """Mean and std over seeds of the test metrics, in percent."""
    out = []
    for m in [k for k in MODEL_ORDER if k in set(runs["model"])]:
        r = runs[runs["model"] == m]
        row = {"model": m, "seeds": len(r)}
        for key, _ in METRICS:
            row[f"{key}_mean"], row[f"{key}_std"] = (100 * v for v in mean_std(r[key]))
        row["errors_mean"], row["errors_std"] = mean_std(r["errors"])
        out.append(row)
    return pd.DataFrame(out)


def paired_statistics(preds, n_boot=2000):
    """MobileViT-XS minus MobileNetV2, per seed: McNemar exact test and bootstrap 95% CIs."""
    by_seed = {}
    for p in preds.values():
        by_seed.setdefault(p["seed"], {})[p["model"]] = p
    rows = []
    for seed, d in sorted(by_seed.items()):
        if not all(m in d for m in MODEL_ORDER):
            continue
        a, b = d[MODEL_ORDER[0]], d[MODEL_ORDER[1]]
        assert (a["labels"] == b["labels"]).all() and (a["paths"] == b["paths"]).all()
        y, pa, pb = a["labels"], a["logits"].argmax(1), b["logits"].argmax(1)
        mc = mcnemar_exact(y, pa, pb)
        bs = paired_bootstrap(y, pa, pb, len(a["class_names"]), n_boot=n_boot, seed=seed)
        rows.append({"seed": seed,
                     "acc_diff_pp": 100 * ((pb == y).mean() - (pa == y).mean()),
                     "acc_ci_low_pp": 100 * bs["accuracy_diff_ci95"][0],
                     "acc_ci_high_pp": 100 * bs["accuracy_diff_ci95"][1],
                     "f1_ci_low_pp": 100 * bs["macro_f1_diff_ci95"][0],
                     "f1_ci_high_pp": 100 * bs["macro_f1_diff_ci95"][1],
                     "only_mobilenetv2_right": mc["a_right_b_wrong"],
                     "only_mobilevit_right": mc["a_wrong_b_right"],
                     "mcnemar_p": mc["p_value"]})
    return pd.DataFrame(rows)


def training_cost(runs):
    out = []
    for m in [k for k in MODEL_ORDER if k in set(runs["model"])]:
        r = runs[runs["model"] == m]
        row = {"model": m}
        for k in ("epochs_run", "best_epoch", "mean_epoch_train_time_s", "total_train_time_s",
                  "mean_train_images_per_s", "peak_train_mem_mib"):
            row[f"{k}_mean"], row[f"{k}_std"] = mean_std(r[k])
        row["gpus"] = ", ".join(sorted({g for gs in r["gpus"] for g in gs}))
        out.append(row)
    return pd.DataFrame(out)


def deployability(bench: dict):
    rows = []
    for m in [k for k in MODEL_ORDER if k in bench]:
        b = bench[m]
        row = {"model": m, "input": b["input_size"], "params_m": b["parameters"] / 1e6,
               "size_mb_fp32": b["size_mb_fp32"], "size_mb_fp16": b["size_mb_fp16"],
               "gmacs": b["gmacs"], "gflops": b["gflops"],
               "cpu1_ms": b["cpu_1t"]["mean_ms"], "cpu1_std": b["cpu_1t"]["std_ms"],
               "cpuall_ms": b["cpu_allt"]["mean_ms"], "cpuall_std": b["cpu_allt"]["std_ms"],
               "cpu_threads_all": b["cpu_allt"]["threads"]}
        if "gpu_fp32" in b:
            row.update({"gpu_ms": b["gpu_fp32"]["mean_ms"], "gpu_std": b["gpu_fp32"]["std_ms"],
                        "gpu_p95": b["gpu_fp32"]["p95_ms"],
                        "gpu16_ms": b["gpu_fp16"]["mean_ms"], "gpu16_std": b["gpu_fp16"]["std_ms"],
                        "throughput": b["gpu_throughput_img_s"],
                        "mem_bs1_mib": b["gpu_peak_mem_mib_bs1"], "mem_bs32_mib": b["gpu_peak_mem_mib_bs32"],
                        "gpu_name": b["gpu_name"]})
        rows.append(row)
    return pd.DataFrame(rows)


def tradeoff(main, dep):
    """Relative cost of MobileViT-XS against MobileNetV2 next to its accuracy difference."""
    a, b = MODEL_ORDER
    if not {a, b} <= set(main["model"]) or not {a, b} <= set(dep["model"]):
        return pd.DataFrame()
    M, D = main.set_index("model"), dep.set_index("model")
    rows = [("Accuracy (pp)", M.at[b, "accuracy_mean"] - M.at[a, "accuracy_mean"], "difference"),
            ("Macro F1 (pp)", M.at[b, "macro_f1_mean"] - M.at[a, "macro_f1_mean"], "difference")]
    for col, label in [("params_m", "Parameters"), ("size_mb_fp32", "File size (FP32)"),
                       ("gmacs", "MACs"), ("gpu_ms", "GPU latency (FP32, bs 1)"),
                       ("gpu16_ms", "GPU latency (FP16, bs 1)"), ("cpu1_ms", "CPU latency (1 thread)"),
                       ("cpuall_ms", "CPU latency (all threads)"), ("throughput", "GPU throughput")]:
        if col in D.columns and pd.notna(D.at[a, col]):
            rows.append((label, D.at[b, col] / D.at[a, col], "ratio"))
    return pd.DataFrame(rows, columns=["quantity", "mobilevit_vs_mobilenetv2", "kind"])


def per_class_table(preds):
    """Per-class F1 of each model (mean over seeds) and the difference."""
    frames = []
    for p in preds.values():
        pc = per_class_metrics(p["labels"], p["logits"].argmax(1), p["class_names"])
        pc["model"], pc["seed"] = p["model"], p["seed"]
        frames.append(pc)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames)
    g = df.groupby(["class", "model"]).agg(f1=("f1", "mean"), f1_std=("f1", "std"),
                                           support=("support", "first")).reset_index()
    wide = g.pivot(index="class", columns="model", values=["f1", "f1_std"])
    wide.columns = [f"{v}_{m}" for v, m in wide.columns]
    wide["support"] = g.groupby("class")["support"].first()
    if all(f"f1_{m}" in wide for m in MODEL_ORDER):
        wide["diff"] = wide[f"f1_{MODEL_ORDER[1]}"] - wide[f"f1_{MODEL_ORDER[0]}"]
    return wide.reset_index().fillna(0.0)


def mean_confusion(preds, model_key):
    from .evaluate import confusion
    mats = [confusion(p["labels"], p["logits"].argmax(1), len(p["class_names"]))
            for p in preds.values() if p["model"] == model_key]
    return np.mean(mats, axis=0) if mats else None


def top_confusions_all(preds, k=8):
    rows = []
    for m in MODEL_ORDER:
        cm = mean_confusion(preds, m)
        if cm is None:
            continue
        names = next(p["class_names"] for p in preds.values() if p["model"] == m)
        off = cm.copy()
        np.fill_diagonal(off, 0)
        for flat in np.argsort(-off, axis=None)[:k]:
            i, j = np.unravel_index(flat, off.shape)
            if off[i, j] > 0:
                rows.append({"model": m, "true": names[i], "predicted": names[j],
                             "mean_count": float(off[i, j]), "class_size": int(cm[i].sum())})
    return pd.DataFrame(rows)


def ablation_table(native_runs, ablation_runs):
    rows = []
    for _, r in ablation_runs.iterrows():
        nat = native_runs[(native_runs["model"] == r["model"]) & (native_runs["seed"] == r["seed"])]
        if nat.empty:
            continue
        n = nat.iloc[0]
        rows.append({"model": r["model"], "seed": r["seed"],
                     "native_acc": 100 * n["accuracy"], "imagenet_acc": 100 * r["accuracy"],
                     "native_f1": 100 * n["macro_f1"], "imagenet_f1": 100 * r["macro_f1"],
                     "native_best_epoch": n["best_epoch"], "imagenet_best_epoch": r["best_epoch"]})
    df = pd.DataFrame(rows)
    if not df.empty:
        df["acc_diff_pp"] = df["imagenet_acc"] - df["native_acc"]
        df["f1_diff_pp"] = df["imagenet_f1"] - df["native_f1"]
    return df


# ---------------------------------------------------------------------------------------------
# LaTeX
# ---------------------------------------------------------------------------------------------
def tex_escape(s) -> str:
    s = str(s)
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"),
                 ("#", r"\#"), ("_", r"\_"), ("{", r"\{"), ("}", r"\}")]:
        s = s.replace(a, b)
    return s


def pm(mean, std, digits=2, bold=False):
    s = f"{mean:.{digits}f} $\\pm$ {std:.{digits}f}"
    return f"\\textbf{{{s}}}" if bold else s


def class_label(name: str) -> str:
    """'Tomato___Early_blight' -> 'Tomato: Early blight' (for tables and figures)."""
    crop, _, disease = name.partition("___")
    crop = crop.replace("_", " ").replace(",", "").replace("(including sour)", "").replace("(maize)", "").strip()
    disease = re.sub(r"\s+", " ", disease.replace("_", " ")).strip()
    return f"{crop}: {disease}"


def latex_table(header, rows, caption, label, align=None, note=None, wide=False, size="\\footnotesize"):
    """Booktabs table. `rows` are lists of already-formatted cell strings."""
    env = "table*" if wide else "table"
    align = align or ("l" + "c" * (len(header) - 1))
    lines = [f"\\begin{{{env}}}[!t]", "\\centering", size,
             f"\\caption{{{caption}}}", f"\\label{{{label}}}",
             f"\\begin{{tabular}}{{{align}}}", "\\toprule",
             " & ".join(header) + " \\\\", "\\midrule"]
    lines += [" & ".join(r) + " \\\\" for r in rows]
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    if note:
        lines.append(f"\\par\\smallskip\\parbox{{\\linewidth}}{{\\scriptsize {note}}}")
    lines.append(f"\\end{{{env}}}")
    return "\n".join(lines) + "\n"


def write_tex(text, path: Path):
    storage.save_text(text, path)
    return path
