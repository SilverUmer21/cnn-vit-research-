"""Deployability measurements (E2): parameters, file size, FLOPs, latency, throughput, memory.

Latency follows the proposal (Section IV-E): batch size 1, 10 warm-up runs (discarded), then 100
timed runs, both models on the same hardware in the same session. Only the forward pass is timed
(no image decoding or preprocessing). GPU timing uses CUDA events; CPU timing uses
time.perf_counter. CPU latency with 1 thread is the closest stand-in for a single mobile core.
"""
from __future__ import annotations

import time

import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode

from .config import MODELS, BenchmarkConfig
from .models import count_parameters, state_dict_size_mb


def count_flops(model, input_size, device="cpu"):
    """FLOPs of one forward pass for one image, counted by PyTorch's FlopCounterMode
    (convolutions and matrix multiplications, 1 multiply-add = 2 FLOPs). MACs = FLOPs / 2,
    which is the unit the MobileNetV2 and MobileViT papers report."""
    model = model.to(device).eval()
    x = torch.randn(1, 3, input_size, input_size, device=device)
    with FlopCounterMode(display=False) as counter, torch.inference_mode():
        model(x)
    flops = counter.get_total_flops()
    return {"gflops": flops / 1e9, "gmacs": flops / 2e9}


def _summary(times_ms):
    t = np.asarray(times_ms)
    return {"mean_ms": float(t.mean()), "std_ms": float(t.std(ddof=1)), "median_ms": float(np.median(t)),
            "p95_ms": float(np.percentile(t, 95)), "min_ms": float(t.min()), "runs": int(t.size)}


@torch.inference_mode()
def gpu_latency(model, input_size, cfg: BenchmarkConfig, batch_size=None, half=False):
    batch_size = batch_size or cfg.batch_size
    model = model.to("cuda").eval()
    x = torch.randn(batch_size, 3, input_size, input_size, device="cuda")
    ctx = torch.autocast("cuda", dtype=torch.float16, enabled=half)
    with ctx:
        for _ in range(cfg.warmup):
            model(x)
    torch.cuda.synchronize()
    times = []
    for _ in range(cfg.runs):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        with ctx:
            start.record()
            model(x)
            end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return _summary(times)


@torch.inference_mode()
def cpu_latency(model, input_size, cfg: BenchmarkConfig, threads=1):
    previous = torch.get_num_threads()
    if threads:
        torch.set_num_threads(threads)
    try:
        model = model.to("cpu").eval()
        x = torch.randn(cfg.batch_size, 3, input_size, input_size)
        for _ in range(cfg.warmup):
            model(x)
        times = []
        for _ in range(cfg.runs):
            t0 = time.perf_counter()
            model(x)
            times.append((time.perf_counter() - t0) * 1000)
        out = _summary(times)
        out["threads"] = torch.get_num_threads()
        return out
    finally:
        torch.set_num_threads(previous)


@torch.inference_mode()
def gpu_throughput(model, input_size, cfg: BenchmarkConfig):
    """Images per second at batch size 32 (FP32), after the same warm-up."""
    model = model.to("cuda").eval()
    x = torch.randn(cfg.throughput_batch_size, 3, input_size, input_size, device="cuda")
    for _ in range(cfg.warmup):
        model(x)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(cfg.throughput_runs):
        model(x)
    torch.cuda.synchronize()
    return cfg.throughput_batch_size * cfg.throughput_runs / (time.perf_counter() - t0)


@torch.inference_mode()
def gpu_peak_memory_mib(model, input_size, batch_size):
    model = model.to("cuda").eval()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    model(torch.randn(batch_size, 3, input_size, input_size, device="cuda"))
    torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated() - base) / 2**20


def benchmark_model(model, model_key, cfg: BenchmarkConfig = BenchmarkConfig(), log=print):
    """All deployability numbers for one trained model (weights do not change any of them)."""
    size = MODELS[model_key]["input_size"]
    total, trainable = count_parameters(model)
    out = {
        "model": model_key, "input_size": size,
        "parameters": total, "trainable_parameters": trainable,
        "size_mb_fp32": state_dict_size_mb(model, torch.float32),
        "size_mb_fp16": state_dict_size_mb(model, torch.float16),
        **count_flops(model, size, "cpu"),
    }
    for threads in cfg.cpu_threads:
        key = f"cpu_{threads or 'all'}t"
        log(f"  {model_key}: CPU latency ({threads or 'all'} thread(s))")
        out[key] = cpu_latency(model, size, cfg, threads)
    if torch.cuda.is_available():
        log(f"  {model_key}: GPU latency, throughput, memory")
        out["gpu_name"] = torch.cuda.get_device_name(0)
        out["gpu_fp32"] = gpu_latency(model, size, cfg, half=False)
        out["gpu_fp16"] = gpu_latency(model, size, cfg, half=True)
        out["gpu_throughput_img_s"] = gpu_throughput(model, size, cfg)
        out["gpu_peak_mem_mib_bs1"] = gpu_peak_memory_mib(model, size, 1)
        out["gpu_peak_mem_mib_bs32"] = gpu_peak_memory_mib(model, size, cfg.throughput_batch_size)
        model.to("cpu")
    return out
