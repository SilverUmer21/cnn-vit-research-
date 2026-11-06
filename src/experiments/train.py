"""Resumable fine-tuning of one (model, seed, preprocessing) run.

A run folder on persistent storage (Google Drive on Colab) holds:
  config.json            settings, library versions and hardware of the run
  last.pt                full training state after the latest finished epoch (deleted when done)
  best.pt                weights with the lowest validation loss so far
  history.csv            one row per epoch
  test_predictions.npz   test logits, labels and image paths of the best checkpoint
  metrics.json           test metrics of the best checkpoint
  summary.json           everything needed for the results tables
  DONE                   written last; a run with DONE is never trained again

Calling `train_run` again after a disconnect continues from `last.pt`. Because data order,
augmentation and dropout depend only on (seed, epoch), the resumed run sees the same batches it
would have seen without the interruption.
"""
from __future__ import annotations

import math
import platform
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from . import storage
from .config import MODELS, TrainConfig, run_name
from .data import make_loaders
from .evaluate import classification_metrics, predict
from .models import build_model, count_parameters


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def environment_info(device: str) -> dict:
    import transformers
    info = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "device": device,
        "cpu": platform.processor() or platform.machine(),
        "cpu_count": __import__("os").cpu_count(),
    }
    if device == "cuda":
        info["gpu"] = torch.cuda.get_device_name(0)
        info["cuda"] = torch.version.cuda
        info["cudnn"] = torch.backends.cudnn.version()
    return info


def _cosine(total_steps: int):
    return lambda step: 0.5 * (1.0 + math.cos(math.pi * min(step, total_steps) / total_steps))


def _train_one_epoch(model, loader, optimizer, scheduler, scaler, device, cfg):
    model.train()
    use_amp = cfg.amp and device == "cuda"
    loss_sum = torch.zeros((), device=device)
    correct = torch.zeros((), device=device, dtype=torch.long)
    n = 0
    for i, (x, y) in enumerate(loader):
        if cfg.max_train_batches is not None and i >= cfg.max_train_batches:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            logits = model(x)
            loss = F.cross_entropy(logits, y, label_smoothing=cfg.label_smoothing)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        if cfg.grad_clip_norm is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        # accumulate on the GPU: no .item() per step, so no CPU-GPU synchronisation per batch
        loss_sum += loss.detach().float() * y.numel()
        correct += (logits.argmax(1) == y).sum()
        n += y.numel()
    loss = (loss_sum / max(n, 1)).item()
    if not math.isfinite(loss):
        raise FloatingPointError("training loss is not finite; stopping this run")
    return loss, correct.item() / max(n, 1), n


def train_run(paths, model_key, seed, variant="native", cfg: TrainConfig = TrainConfig(),
              device=None, pretrained=True, log=print):
    """Train (or resume, or skip if finished) one run and return its summary dict."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    name = run_name(model_key, seed, variant)
    rd = paths.run_dir(name)
    if (rd / "DONE").exists():
        log(f"[{name}] already finished, skipping")
        return storage.load_json(rd / "summary.json")
    rd.mkdir(parents=True, exist_ok=True)

    if device == "cuda":
        torch.backends.cudnn.benchmark = True   # fixed input size: let cuDNN pick fast kernels
    seed_everything(seed)
    loaders = make_loaders(paths, model_key, cfg, seed, variant, device)
    class_names = loaders["train"].dataset.classes
    model = build_model(model_key, class_names, pretrained=pretrained).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
                                  fused=(device == "cuda"))
    steps_per_epoch = len(loaders["train"])
    if cfg.max_train_batches is not None:
        steps_per_epoch = min(steps_per_epoch, cfg.max_train_batches)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _cosine(cfg.epochs * steps_per_epoch))
    scaler = torch.amp.GradScaler("cuda", enabled=(cfg.amp and device == "cuda"))

    state = {"epoch": -1, "best_val_loss": float("inf"), "best_epoch": -1, "bad_epochs": 0,
             "stopped_early": False, "history": []}
    if (rd / "last.pt").exists():
        ckpt = storage.load_torch(rd / "last.pt")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        state = ckpt["state"]
        log(f"[{name}] resuming after epoch {state['epoch'] + 1} "
            f"(best val loss {state['best_val_loss']:.4f} at epoch {state['best_epoch'] + 1})")
    else:
        total, trainable = count_parameters(model)
        storage.save_json({
            "run": name, "model": model_key, "checkpoint": MODELS[model_key]["checkpoint"],
            "seed": seed, "preprocessing": variant, "train_config": cfg.to_dict(),
            "parameters": total, "trainable_parameters": trainable,
            "steps_per_epoch": steps_per_epoch, "environment": environment_info(device),
        }, rd / "config.json")
        log(f"[{name}] starting: {trainable:,} trainable parameters, {steps_per_epoch} steps/epoch")

    gpu = torch.cuda.get_device_name(0) if device == "cuda" else "cpu"
    while state["epoch"] + 1 < cfg.epochs and not state["stopped_early"]:
        epoch = state["epoch"] + 1
        torch.manual_seed(seed * 1000 + epoch)          # dropout etc. depend on (seed, epoch) only
        loaders["train"].set_epoch(epoch)
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        train_loss, train_acc, n_train = _train_one_epoch(
            model, loaders["train"], optimizer, scheduler, scaler, device, cfg)
        if device == "cuda":
            torch.cuda.synchronize()
        train_time = time.perf_counter() - t0
        peak_mem = torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else float("nan")

        t1 = time.perf_counter()
        logits, labels, val_loss = predict(model, loaders["val"], device, amp=cfg.amp,
                                           max_batches=cfg.max_eval_batches)
        val_time = time.perf_counter() - t1
        val = classification_metrics(labels, logits.argmax(1), len(class_names))

        improved = val_loss < state["best_val_loss"] - cfg.min_delta
        if improved:
            state.update(best_val_loss=val_loss, best_epoch=epoch, bad_epochs=0)
            storage.save_torch(model.state_dict(), rd / "best.pt")
        else:
            state["bad_epochs"] += 1
        state["stopped_early"] = state["bad_epochs"] >= cfg.patience
        state["history"].append({
            "epoch": epoch + 1, "lr_end": scheduler.get_last_lr()[0],
            "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val["accuracy"], "val_macro_f1": val["macro_f1"],
            "train_time_s": train_time, "val_time_s": val_time,
            "train_images_per_s": n_train / train_time, "peak_train_mem_mib": peak_mem,
            "improved": improved, "gpu": gpu,
        })
        state["epoch"] = epoch
        storage.save_torch({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                            "state": state}, rd / "last.pt")
        storage.save_dataframe(pd.DataFrame(state["history"]), rd / "history.csv")
        log(f"[{name}] epoch {epoch + 1:2d}/{cfg.epochs}  train loss {train_loss:.4f} acc {train_acc:.4f}"
            f" | val loss {val_loss:.4f} acc {val['accuracy']:.4f} F1 {val['macro_f1']:.4f}"
            f" | {train_time:.0f}s ({n_train / train_time:.0f} img/s)" + ("  *best" if improved else ""))
        if state["stopped_early"]:
            log(f"[{name}] early stop: no val-loss improvement for {cfg.patience} epochs")

    return _finalise(rd, name, model, loaders, class_names, state, device, cfg, log)


def _finalise(rd, name, model, loaders, class_names, state, device, cfg, log):
    """Evaluate the best checkpoint on the test set once, save everything, mark the run DONE."""
    model.load_state_dict(storage.load_torch(rd / "best.pt"))
    logits, labels, test_loss = predict(model, loaders["test"], device, amp=False,
                                        max_batches=cfg.max_eval_batches)
    preds = logits.argmax(1)
    test = classification_metrics(labels, preds, len(class_names))
    test["loss"] = test_loss
    paths_ = np.asarray(loaders["test"].dataset.paths[:len(labels)])
    storage.save_npz(rd / "test_predictions.npz", logits=logits, labels=labels, paths=paths_,
                     class_names=np.asarray(class_names))
    storage.save_json(test, rd / "metrics.json")

    hist = pd.DataFrame(state["history"])
    config = storage.load_json(rd / "config.json")
    summary = {
        "run": name, "model": config["model"], "seed": config["seed"],
        "preprocessing": config["preprocessing"], "parameters": config["parameters"],
        "epochs_run": int(len(hist)), "best_epoch": int(state["best_epoch"] + 1),
        "stopped_early": bool(state["stopped_early"]),
        "best_val_loss": float(state["best_val_loss"]),
        "best_val_acc": float(hist.loc[hist["epoch"] == state["best_epoch"] + 1, "val_acc"].iloc[0]),
        "mean_epoch_train_time_s": float(hist["train_time_s"].mean()),
        "total_train_time_s": float((hist["train_time_s"] + hist["val_time_s"]).sum()),
        "mean_train_images_per_s": float(hist["train_images_per_s"].mean()),
        "peak_train_mem_mib": float(hist["peak_train_mem_mib"].max()),
        "gpus": sorted(hist["gpu"].unique().tolist()),
        "test": test,
    }
    storage.save_json(summary, rd / "summary.json")
    storage.save_text("done\n", rd / "DONE")
    (rd / "last.pt").unlink(missing_ok=True)   # optimizer state no longer needed; saves Drive space
    log(f"[{name}] TEST  acc {test['accuracy']:.4f}  macro-F1 {test['macro_f1']:.4f}"
        f"  ({test['errors']} errors / {test['n']})")
    return summary
