# Handoff: data split and loaders (Member 1)

For Members 2 and 3. Everything here is checked by `src/dataset/check_splits.py` and
`src/dataset/check_loaders.py`.

## What you get

- `splits/manifest.csv`: columns `path, label, group_id, split`. This is the only source of truth
  for the split. 54,304 images: train 43,442, val 5,431, test 5,431. Never re-split.
- `src/dataset/datasets.py`: `make_loader(model_name, split, manifest_csv, data_root)`.
- `src/dataset/preprocessing.py`: the image recipe for each model.

## How to use

Run from the repo root. `data_root` is the folder that contains `train/` and `val/` (the
`PlantVillage` folder of the Kaggle copy). Paths in the manifest are relative to it.

```python
from src.dataset.datasets import make_loader, seed_everything

seed_everything(42)
manifest, root = "splits/manifest.csv", "data/PlantVillage"
train = make_loader("mobilevit_xs", "train", manifest, root)   # or "mobilenetv2"
val = make_loader("mobilevit_xs", "val", manifest, root)
test = make_loader("mobilevit_xs", "test", manifest, root)
class_names = train.dataset.classes   # 38 names, the list index is the class number
```

Default batch size is 32 and the seed is 42. Pass `num_workers` for more loading speed.
The loader already prepares the images, so do not apply the Hugging Face image processor again.

## Image recipes (read from the Hugging Face preprocessor_config.json files)

| | `mobilenetv2` | `mobilevit_xs` |
|---|---|---|
| Assumed checkpoint | `google/mobilenet_v2_1.0_224` | `apple/mobilevit-x-small` |
| Input size | 224 | 256 |
| Channel order | RGB | BGR |
| Scaling | divide by 255 | divide by 255 |
| Mean / std | 0.5 / 0.5 (range -1 to 1) | none (range 0 to 1) |

Training images get the same augmentation for both models: random horizontal flip, rotation
of up to 15 degrees (empty corners are black), and brightness/contrast jitter of 0.2.
Validation and test images are only resized and normalized.

## Differences from the proposal

- The proposal uses the ImageNet mean/std for both models. Neither checkpoint uses it.
- MobileViT-XS needs BGR channel order.
- The Hugging Face configs resize and then centre-crop (this cuts off 11 to 12% of the image
  edge, since our images are already 256x256). We resize the whole image instead, so the leaf
  edges stay. This is a deliberate choice.
- If you use different checkpoints from the two assumed above, tell Member 1: the settings
  must be read again from their configs.

## Things to know

- All images are 256x256. MobileViT-XS uses them as they are; MobileNetV2 is resized to 224.
  FLOPs and latency therefore reflect the architecture and the input size together.
- The smallest test sets are tiny (`Potato___healthy` has 15 test images, several classes
  have 27 to 42), so per-class scores for those classes are noisy.
- The split keeps byte-identical and near-duplicate images together. The same leaf photographed
  from a clearly different angle is not reliably detected, so some leaf-level similarity across
  splits may remain.
- One image was excluded from the 54,305 downloaded: a `.png` photo of a pepper fruit that is
  not a lab leaf image (details in `reports/audit.md`).
- The Hugging Face MobileNetV2 outputs 1,001 classes (an extra background class at index 0).
  This only matters if the original head is kept; we replace it with 38 classes.

## Reports

`reports/audit.md` (what the data looked like), `reports/duplicates.md` and `reports/groups.md`
(duplicate detection and grouping), `reports/split_report.md` (the split and its checks).
