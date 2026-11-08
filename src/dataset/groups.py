# Puts duplicate and near-duplicate images under one group_id.
# Usage: python src/dataset/groups.py <scratch_dir> <reports_dir> [threshold]
# Without a threshold it only writes the analysis; with one it also writes groups.csv.
import sys
from pathlib import Path

import numpy as np
import pandas as pd

THRESHOLDS = [0.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0]


def md_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(c).replace("|", "\\|") for c in r) + " |")
    return "\n".join(lines)


def components(n, edges):
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)   # smallest index is the root: deterministic
    return np.array([find(x) for x in range(n)])


def edges_for(df, cand, thr):
    same = cand[cand["same_class"] & ((cand["rms"] <= thr) | cand["md5_equal"])]
    pairs = list(zip(same["i"], same["j"]))
    # byte-identical files that the nearest-neighbour search may have missed
    for _, idx in df.groupby("md5").indices.items():
        for k in idx[1:]:
            pairs.append((int(idx[0]), int(k)))
    return pairs


def summarise(df, root):
    g = pd.Series(root, name="g")
    size = g.map(g.value_counts())
    multi = size > 1
    top = g.value_counts().head(5).tolist()
    span = df.assign(g=root).groupby("g")["split"].nunique()
    n_groups = g.nunique()
    return {
        "groups": n_groups,
        "multi_groups": int((g.value_counts() > 1).sum()),
        "imgs_in_multi": int(multi.sum()),
        "max": top[0],
        "top5": ", ".join(map(str, top)),
        "spanning": int((span > 1).sum()),
    }


def main():
    scratch, reports = Path(sys.argv[1]), Path(sys.argv[2])
    chosen = float(sys.argv[3]) if len(sys.argv) > 3 else None
    df = pd.read_csv(scratch / "images.csv")
    cand = pd.read_csv(scratch / "candidate_pairs.csv")
    n = len(df)

    rows = []
    for thr in THRESHOLDS:
        root = components(n, edges_for(df, cand, thr))
        s = summarise(df, root)
        rows.append([f"{thr:g}", f"{s['groups']:,}", f"{s['multi_groups']:,}",
                     f"{s['imgs_in_multi']:,}", s["max"], s["top5"], s["spanning"]])

    md = ["# Group assignment (step 4)\n"]
    add = md.append
    add(f"Images: {n:,}. Edges: byte-identical files, plus same-class near-duplicate pairs with "
        "grayscale RMS <= T (from `src/dataset/duplicates.py`). Cross-class pairs are never merged. "
        "T = 0 means byte-identical files only.\n")
    add("## 1. Group structure by threshold\n")
    add(md_table(["T (RMS)", "Groups", "Groups with 2+ images", "Images in such groups",
                  "Largest group", "Five largest", "Groups spanning train and val"], rows))
    add("")

    if chosen is not None:
        root = components(n, edges_for(df, cand, chosen))
        order = {r: k for k, r in enumerate(sorted(set(root)), 1)}
        df["group_id"] = [f"g{order[r]:05d}" for r in root]
        df[["path", "cls", "split", "group_id"]].rename(columns={"split": "orig_split"}).to_csv(
            scratch / "groups.csv", index=False)
        sizes = df["group_id"].map(df["group_id"].value_counts())
        add(f"## 2. Chosen threshold: T = {chosen:g}\n")
        multi = df[sizes > 1]
        add(f"Groups: {df['group_id'].nunique():,}. Groups with 2+ images: "
            f"{multi['group_id'].nunique():,} covering {len(multi):,} images. "
            f"Largest group: {int(sizes.max())}.\n")
        size_dist = df.drop_duplicates("group_id").assign(
            s=lambda d: d["group_id"].map(df["group_id"].value_counts()))["s"]
        add(md_table(["Group size", "Groups"],
                     [[k, f"{v:,}"] for k, v in size_dist.value_counts().sort_index().items()]))
        add("")
        cls_tot = df["cls"].value_counts()
        mx = df.assign(s=sizes).groupby("cls")["s"].max()
        worst = pd.DataFrame({"class_size": cls_tot, "max_group": mx})
        worst["test_quota"] = (worst["class_size"] * 0.10).round().astype(int)
        worst = worst.sort_values("max_group", ascending=False).head(8)
        add("Classes with the largest groups (a group must fit inside one split; "
            "test quota = 10% of the class):\n")
        add(md_table(["Class", "Class size", "Largest group", "Test quota"],
                     [[c, f"{r.class_size:,}", int(r.max_group), int(r.test_quota)]
                      for c, r in worst.iterrows()]))
        add("")
        span = df.groupby("group_id")["split"].nunique()
        sp = df[df["group_id"].isin(span[span > 1].index)]
        add(f"Groups that span the shipped train and val: {int((span > 1).sum()):,} "
            f"({len(sp):,} images). These were leaking in the shipped split.\n")

    (reports / "groups.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
