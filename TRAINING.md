# Training on Google Colab (free tier)

## One-time setup
1. Download the dataset zip from Kaggle: https://www.kaggle.com/datasets/mohitsingh1804/plantvillage
   (Download button, do not unzip).
2. Upload it to Google Drive at `My Drive/cnn-vit-research/data/` (any name ending in `.zip`).

## Every session
1. Open `notebooks/experiments.ipynb` in Colab (File → Open notebook → GitHub →
   `SilverUmer21/cnn-vit-research-`, branch `feature/training`).
2. Runtime → Change runtime type → **T4 GPU**.
3. Runtime → **Run all**. Allow Drive access when asked. Keep the tab open.

If Colab disconnects or the GPU quota runs out, come back later and **Run all** again. Finished
runs are skipped; an unfinished run continues from its last finished epoch.

## What is saved (in `My Drive/cnn-vit-research/`)
```
runs/<model>_<preprocessing>_s<seed>/
    config.json, history.csv, best.pt, last.pt (until the run ends),
    test_predictions.npz, metrics.json, summary.json, DONE
results/
    benchmark.json, summary.json
    tables/*.csv and *.tex      (tables for the paper)
    figures/*.pdf and *.png     (figures for the paper)
```

## Runs (in this order)
| # | Run | Purpose |
|---|---|---|
| 1-6 | MobileNetV2 and MobileViT-XS, seeds 42, 43, 44 | E1 main comparison (mean ± std) |
| 7-8 | both models, ImageNet mean/std, seed 42 | E6 preprocessing ablation |

After training, the same notebook runs E2 (deployability benchmark), E3 (trade-off),
E4 (confusions, examples, Grad-CAM) and E5 (robustness). Benchmarks must run in one session so
both models are timed on the same GPU.

## Code
- `src/dataset/`: Member 1's audit, split and image recipes (unchanged).
- `src/experiments/`: training, evaluation, benchmark, robustness, Grad-CAM, tables and figures.
  All settings are in `src/experiments/config.py`.
