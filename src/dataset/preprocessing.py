# Image recipes for the two models. The numbers come from each model's Hugging Face
# preprocessor_config.json (google/mobilenet_v2_1.0_224 and apple/mobilevit-x-small).
# The augmentation is shared. Only size, normalization and channel order differ.
import torch
from torchvision.transforms import InterpolationMode, v2

SPECS = {
    "mobilenetv2": {"size": 224, "mean": (0.5, 0.5, 0.5), "std": (0.5, 0.5, 0.5), "bgr": False},
    "mobilevit_xs": {"size": 256, "mean": None, "std": None, "bgr": True},
}
ROTATION_DEGREES = 15
JITTER = 0.2


class ReverseChannels(torch.nn.Module):
    # RGB to BGR: MobileViT expects blue first
    def forward(self, x):
        return x.flip(-3)


def augmentation():
    # same policy for both models, used on training images only
    return [
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomRotation(ROTATION_DEGREES),
        v2.ColorJitter(brightness=JITTER, contrast=JITTER),
    ]


def build_transform(model_name, train):
    spec = SPECS[model_name]
    steps = augmentation() if train else []
    # resize the whole image (no centre crop) so the leaf edges are kept
    size = (spec["size"], spec["size"])
    steps += [
        v2.Resize(size, interpolation=InterpolationMode.BILINEAR, antialias=True),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ]
    if spec["mean"] is not None:
        steps.append(v2.Normalize(mean=spec["mean"], std=spec["std"]))
    if spec["bgr"]:
        steps.append(ReverseChannels())
    return v2.Compose(steps)
