"""Export the fine-tuned seed-42 models to ONNX and check them against PyTorch.

The ONNX files are what edge/phone_benchmark.html runs on a phone (ONNX Runtime Web).
Usage (repo root): python -m edge.export_onnx <runs_dir> <out_dir>
  runs_dir: the Drive folder with <model>_native_s42/best.pt
"""
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
import torch

from src.experiments.config import MODEL_ORDER, MODELS, run_name
from src.experiments.models import build_model


def export(runs_dir, out_dir, seed=42):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    names = sorted(pd.read_csv("splits/manifest.csv")["label"].unique())
    torch.manual_seed(0)
    report = {}
    for key in MODEL_ORDER:
        model = build_model(key, names, pretrained=False)
        state = torch.load(Path(runs_dir) / run_name(key, seed) / "best.pt", map_location="cpu")
        model.load_state_dict(state)
        model.eval()
        size = MODELS[key]["input_size"]
        x = torch.randn(1, 3, size, size)
        path = out_dir / f"{key}.onnx"
        torch.onnx.export(model, (x,), str(path), input_names=["pixel_values"], output_names=["logits"],
                          opset_version=18, dynamo=True, external_data=False)

        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        with torch.no_grad():
            ref = model(x).numpy()
        out = session.run(None, {"pixel_values": x.numpy()})[0]
        # the fixed input the phone page also uses to check that it computes the same function
        grey = np.full((1, 3, size, size), 0.5, np.float32)
        logits = session.run(None, {"pixel_values": grey})[0][0]
        report[key] = {"onnx_mb": path.stat().st_size / 1e6, "max_abs_diff": float(np.abs(out - ref).max()),
                       "same_prediction": bool((out.argmax(1) == ref.argmax(1)).all()),
                       "check_top_class": int(logits.argmax()), "check_score": float(logits.max())}
    return pd.DataFrame(report).T


if __name__ == "__main__":
    print(export(sys.argv[1], sys.argv[2]).to_string())
