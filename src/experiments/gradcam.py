"""Grad-CAM (Selvaraju et al., 2017) for qualitative comparison (E4).

The heat map shows which image regions raised the score of the chosen class, computed on the last
convolutional feature map of each model (MobileNetV2: 1280x7x7; MobileViT-XS: 384x8x8, after the
last MobileViT block, so it already contains the transformer's global mixing)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .config import MODELS


class GradCAM:
    def __init__(self, model):
        self.model = model.eval()
        layer = model.get_submodule(MODELS[model.model_key]["cam_layer"])
        self._acts = self._grads = None
        self._hook = layer.register_forward_hook(self._save)

    def _save(self, _module, _inputs, out):
        # a tensor hook (not a module backward hook) is robust to in-place activations
        self._acts = out.detach()
        if out.requires_grad:
            out.register_hook(lambda g: setattr(self, "_grads", g.detach()))

    def __call__(self, x: torch.Tensor, class_idx=None):
        """x: [1, 3, H, W] already prepared for the model. Returns (cam [H, W] in [0, 1], class, prob)."""
        x = x.clone().requires_grad_(True)
        with torch.enable_grad():
            logits = self.model(x)
            probs = logits.softmax(1)
            c = int(logits.argmax(1)) if class_idx is None else int(class_idx)
            self.model.zero_grad(set_to_none=True)
            logits[0, c].backward()
        weights = self._grads.mean(dim=(2, 3), keepdim=True)          # global-average gradients
        cam = F.relu((weights * self._acts).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = cam - cam.min()
        cam = cam / cam.max().clamp_min(1e-8)
        return cam.detach().cpu().numpy(), c, float(probs[0, c].detach())

    def remove(self):
        self._hook.remove()


def overlay(image_rgb: np.ndarray, cam: np.ndarray, alpha=0.45, cmap="inferno"):
    """image in [0, 1], HWC; cam resized to the image if needed."""
    import matplotlib

    if cam.shape != image_rgb.shape[:2]:
        t = torch.from_numpy(cam)[None, None]
        cam = F.interpolate(t, size=image_rgb.shape[:2], mode="bilinear", align_corners=False)[0, 0].numpy()
    heat = matplotlib.colormaps[cmap](cam)[..., :3]
    return np.clip((1 - alpha) * image_rgb + alpha * heat, 0, 1)
