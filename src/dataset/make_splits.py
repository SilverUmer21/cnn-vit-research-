# Makes the 80/10/10 split with seed 42. Whole groups stay in one split.
# Usage: python -I src/dataset/make_splits.py <scratch_dir> <manifest_csv>
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

SEED = 42
FOLD_TO_SPLIT = {0: "test", 1: "val"}   # every other fold is train


def main():
    scratch, out = Path(sys.argv[1]), Path(sys.argv[2])
    g = pd.read_csv(scratch / "groups.csv").sort_values("path").reset_index(drop=True)

    folds = np.full(len(g), -1)
    sgkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=SEED)
    for k, (_, test_idx) in enumerate(sgkf.split(g["path"], g["cls"], g["group_id"])):
        folds[test_idx] = k
    assert (folds >= 0).all()

    manifest = pd.DataFrame({
        "path": g["path"],
        "label": g["cls"],
        "group_id": g["group_id"],
        "split": [FOLD_TO_SPLIT.get(f, "train") for f in folds],
    })
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(out, index=False, lineterminator="\n")

    n = len(manifest)
    print(f"wrote {out}  ({n:,} rows)")
    counts = manifest["split"].value_counts()
    for s in ("train", "val", "test"):
        print(f"  {s:5s} {counts[s]:6,d}  {100 * counts[s] / n:5.2f}%")
    share = manifest.pivot_table(index="label", columns="split", values="path",
                                 aggfunc="count", fill_value=0)
    share = share.div(share.sum(axis=1), axis=0) * 100
    dev = (share[["train", "val", "test"]] - pd.Series({"train": 80, "val": 10, "test": 10})).abs()
    print(f"  largest per-class deviation from 80/10/10: {dev.to_numpy().max():.2f} points")
    print("  classes deviating by more than 1 point:",
          ", ".join(dev.index[(dev > 1).any(axis=1)]) or "none")


if __name__ == "__main__":
    main()
