"""Model construction. Only the ImageNet classification head is replaced (38 classes); the
pretrained architectures are not modified, as stated in the proposal (Section IV-B)."""
from __future__ import annotations

import io

import torch
from torch import nn

from .config import MODELS


class Classifier(nn.Module):
    """Thin wrapper so every model is called as `model(x) -> logits` (plain tensors in and out).
    This keeps training, FLOP counting, latency timing and Grad-CAM identical for both models."""

    def __init__(self, net: nn.Module, model_key: str):
        super().__init__()
        self.net = net
        self.model_key = model_key

    def forward(self, x):
        return self.net(pixel_values=x).logits


def build_model(model_key: str, class_names, pretrained: bool = True) -> Classifier:
    """ImageNet-pretrained backbone from Hugging Face with a fresh 38-class head.

    pretrained=False builds the same architecture with random weights (offline tests only)."""
    from transformers import AutoConfig, AutoModelForImageClassification
    from transformers.utils import logging as hf_logging

    spec = MODELS[model_key]
    id2label = {i: c for i, c in enumerate(class_names)}
    label2id = {c: i for i, c in id2label.items()}
    kwargs = dict(id2label=id2label, label2id=label2id)   # 38 labels -> 38-class head
    if pretrained:
        # the head size changes (1001/1000 -> 38), so the old head is dropped and re-initialised;
        # the library reports that at length, which is expected here, so its log is muted
        verbosity = hf_logging.get_verbosity()
        hf_logging.set_verbosity_error()
        try:
            net = AutoModelForImageClassification.from_pretrained(
                spec["checkpoint"], ignore_mismatched_sizes=True, **kwargs)
        finally:
            hf_logging.set_verbosity(verbosity)
    else:
        config = AutoConfig.from_pretrained(spec["checkpoint"], **kwargs)
        net = AutoModelForImageClassification.from_config(config)
    model = Classifier(net, model_key)
    head = model.net.classifier
    assert head.out_features == len(class_names), "classification head was not replaced"
    return model


def count_parameters(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def state_dict_size_mb(model: nn.Module, dtype=torch.float32) -> float:
    """Size of the saved weights (torch.save of the state dict) in MB (10^6 bytes)."""
    sd = {k: (v.to(dtype) if v.is_floating_point() else v) for k, v in model.state_dict().items()}
    buf = io.BytesIO()
    torch.save(sd, buf)
    return buf.getbuffer().nbytes / 1e6
