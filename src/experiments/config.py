"""Single place for every experimental setting.

All values follow the proposal (Methodology, Section IV-D) unless a comment says otherwise.
Change nothing here between the two models: the comparison is only fair if both runs share it.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path

# ---------------------------------------------------------------------------------------------
# Models. Checkpoint names were confirmed with Member 1; their preprocessor configs were read
# and src/dataset/preprocessing.py reproduces them (see tests/test_preprocessing_matches_hf.py).
# ---------------------------------------------------------------------------------------------
MODELS = {
    "mobilenetv2": {
        "checkpoint": "google/mobilenet_v2_1.0_224",
        "display": "MobileNetV2",
        "family": "CNN",
        "input_size": 224,
        # last convolutional feature map (1280 x 7 x 7) used for Grad-CAM
        "cam_layer": "net.mobilenet_v2.conv_1x1",
    },
    "mobilevit_xs": {
        "checkpoint": "apple/mobilevit-x-small",
        "display": "MobileViT-XS",
        "family": "Hybrid CNN-ViT",
        "input_size": 256,
        # last convolutional feature map (384 x 8 x 8), after the final MobileViT block
        "cam_layer": "net.mobilevit.conv_1x1_exp",
    },
}
MODEL_ORDER = ["mobilenetv2", "mobilevit_xs"]

# Preprocessing variants. "native" is Member 1's recipe (the main experiments). "imagenet" is the
# proposal's original ImageNet mean/std in RGB, used only for the preprocessing ablation (E6).
PREPROCESSING_VARIANTS = ("native", "imagenet")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Training seeds. The data split is fixed (seed 42, splits/manifest.csv); only model
# initialisation of the new head, data order, augmentation and dropout change between seeds.
SEEDS = (42, 43, 44)
ABLATION_SEED = 42


@dataclass(frozen=True)
class TrainConfig:
    # --- proposal, Section IV-D -----------------------------------------------------------
    epochs: int = 10                 # maximum epochs (proposal: 20; reduced for the Colab GPU budget)
    batch_size: int = 32
    lr: float = 1e-4                 # AdamW initial learning rate
    weight_decay: float = 1e-4       # decoupled weight decay (AdamW), applied to all parameters
    patience: int = 5                # early stopping on validation loss
    min_delta: float = 0.0           # an epoch counts as an improvement if val loss decreases
    # --- implementation choices (same for both models) ------------------------------------
    scheduler: str = "cosine_per_step"   # cosine annealing to 0 over `epochs`, stepped per batch
    amp: bool = True                 # FP16 mixed precision on CUDA (T4 tensor cores); off on CPU
    eval_batch_size: int = 128       # evaluation only; does not change results (BN in eval mode)
    num_workers: int = 2             # Colab free tier has 2 vCPUs
    prefetch_factor: int = 4
    label_smoothing: float = 0.0     # plain categorical cross-entropy, as in the proposal
    grad_clip_norm: float | None = None   # not used (not in the proposal)
    # --- limits for quick local tests (None = full data) ----------------------------------
    max_train_batches: int | None = None
    max_eval_batches: int | None = None

    def to_dict(self):
        return asdict(self)

    def with_overrides(self, **kw):
        return replace(self, **kw)


@dataclass(frozen=True)
class BenchmarkConfig:
    # proposal, Section IV-E: 100 timed runs after 10 warm-up runs, batch size 1
    warmup: int = 10
    runs: int = 100
    batch_size: int = 1
    throughput_batch_size: int = 32
    throughput_runs: int = 30
    cpu_threads: tuple = (1, 0)      # 0 = all available cores


@dataclass
class Paths:
    """Where everything lives. On Colab `persist` is a Google Drive folder, so it survives
    disconnects; `local_data` is the fast local disk copy of the images."""
    repo: Path
    persist: Path
    data_root: Path
    manifest: Path | None = None     # defaults to <repo>/splits/manifest.csv (the only split we use)

    def __post_init__(self):
        self.repo, self.persist, self.data_root = Path(self.repo), Path(self.persist), Path(self.data_root)
        self.manifest = Path(self.manifest) if self.manifest else self.repo / "splits" / "manifest.csv"

    @property
    def runs(self):
        return self.persist / "runs"

    @property
    def results(self):
        return self.persist / "results"

    @property
    def tables(self):
        return self.results / "tables"

    @property
    def figures(self):
        return self.results / "figures"

    def run_dir(self, run_name):
        return self.runs / run_name

    def make(self):
        for p in (self.runs, self.results, self.tables, self.figures):
            p.mkdir(parents=True, exist_ok=True)
        return self


def run_name(model_key: str, seed: int, variant: str = "native") -> str:
    """e.g. mobilenetv2_native_s42"""
    if model_key not in MODELS:
        raise KeyError(f"unknown model {model_key!r}; expected one of {list(MODELS)}")
    if variant not in PREPROCESSING_VARIANTS:
        raise KeyError(f"unknown preprocessing variant {variant!r}")
    return f"{model_key}_{variant}_s{seed}"


def main_runs(seeds=SEEDS):
    """Main experiment (E1) run list, interleaved by seed so a paired result exists early."""
    return [(m, s, "native") for s in seeds for m in MODEL_ORDER]


def ablation_runs(seed=ABLATION_SEED):
    """Preprocessing ablation (E6): proposal's ImageNet normalisation, one seed per model."""
    return [(m, seed, "imagenet") for m in MODEL_ORDER]
