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


def resolve_cpu_threads(argv: list[str] | None = None) -> int:
    """CLI ``--cpu-threads``, then the YAML ``cpu_threads`` value, then 10."""

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--cpu-threads", dest="cpu_threads", type=int, default=None)
    parser.add_argument("--config", default="configs/default.yaml")
    args, _unknown = parser.parse_known_args(argv)
    if args.cpu_threads is not None:
        return _positive(args.cpu_threads)

    path = Path(args.config)
    if path.exists():
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if isinstance(loaded, dict) and loaded.get("cpu_threads") not in (None, ""):
            return _positive(int(loaded["cpu_threads"]))
    return DEFAULT_CPU_THREADS


def install_cpu_thread_env(cpu_threads: int) -> int:
    """Point oneDNN and OpenMP at the physical-core pool before TensorFlow starts."""

    intra = _positive(cpu_threads)
    intra_text = str(intra)
    os.environ["TF_NUM_INTRAOP_THREADS"] = intra_text
    os.environ["TF_NUM_INTEROP_THREADS"] = str(INTER_OP_THREADS)
    os.environ["OMP_NUM_THREADS"] = intra_text
    os.environ["MKL_NUM_THREADS"] = intra_text
    return intra


def _positive(value: int) -> int:
    if value < 1:
        raise ValueError(f"cpu_threads must be at least 1, got {value}")
    return value
