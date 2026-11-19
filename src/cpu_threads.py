"""CPU thread-pool settings for a new TensorFlow process.

Call ``install_cpu_thread_env`` before importing TensorFlow. A process that
has already imported TensorFlow keeps the pool it started with.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml

DEFAULT_CPU_THREADS = 10
INTER_OP_THREADS = 1


def resolve_thread_settings(argv: list[str] | None = None) -> tuple[int, int]:
    """CLI thread flags, then the YAML values, then the laptop defaults.

    Returns ``(intra_op, inter_op)``. Inter-op stays at 1 unless the config
    raises it. A GPU run wants a few inter-op threads so the input pipeline
    can overlap the device.
    """

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--cpu-threads", dest="cpu_threads", type=int, default=None)
    parser.add_argument("--inter-op-threads", dest="inter_op_threads", type=int, default=None)
    parser.add_argument("--config", default="configs/default.yaml")
    args, _unknown = parser.parse_known_args(argv)
    loaded = _load_config(args.config)
    intra = args.cpu_threads
    inter = args.inter_op_threads
    if intra is None:
        intra = int(loaded.get("cpu_threads") or DEFAULT_CPU_THREADS)
    if inter is None:
        inter = int(loaded.get("inter_op_threads") or INTER_OP_THREADS)
    return _positive(intra), _positive(inter)


def resolve_cpu_threads(argv: list[str] | None = None) -> int:
    """CLI ``--cpu-threads``, then the YAML ``cpu_threads`` value, then 10."""

    return resolve_thread_settings(argv)[0]


def install_cpu_thread_env(cpu_threads: int, inter_op_threads: int = INTER_OP_THREADS) -> int:
    """Point oneDNN and OpenMP at the requested pools before TensorFlow starts."""

    intra = _positive(cpu_threads)
    inter = _positive(inter_op_threads)
    intra_text = str(intra)
    os.environ["TF_NUM_INTRAOP_THREADS"] = intra_text
    os.environ["TF_NUM_INTEROP_THREADS"] = str(inter)
    os.environ["OMP_NUM_THREADS"] = intra_text
    os.environ["MKL_NUM_THREADS"] = intra_text
    return intra


def install_gpu_performance_env(argv: list[str] | None = None) -> None:
    """Turn on CUDA allocator and kernel autotune before TensorFlow is imported.

    No-op unless the selected YAML sets ``gpu_performance: true``. Existing
    environment variables are left alone so a school machine can override them.
    """

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", default="configs/default.yaml")
    args, _unknown = parser.parse_known_args(argv)
    loaded = _load_config(args.config)
    if not loaded.get("gpu_performance"):
        return
    os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    os.environ.setdefault("TF_CUDNN_USE_AUTOTUNE", "1")
    os.environ.setdefault("TF_GPU_THREAD_MODE", "gpu_private")
    os.environ.setdefault("TF_GPU_THREAD_COUNT", "2")


def _load_config(path: str) -> dict:
    config_path = Path(path)
    if not config_path.exists():
        return {}
    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        return {}
    return loaded


def _positive(value: int) -> int:
    if value < 1:
        raise ValueError(f"cpu_threads must be at least 1, got {value}")
    return value
