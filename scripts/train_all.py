"""Train every recognizer, one after another, on one GPU.

    python scripts/train_all.py --data-dir "data/cleaned_data(50_50)"

The RTX 5070 Ti has one 16 GB device. This command gives that device to one
model, waits until the run finishes, then starts the next model. A second
process on the same card splits the memory and the CUDA cores, so each epoch
gets slower and a transfer model is likely to run out of memory.

Batch size and learning rate come from the selected YAML. The default file
is configs/rtx5070ti.yaml. This RTX 4060 desktop uses configs/rtx4060.yaml.
Epoch 1 of each model compiles the graph and is slower than the epochs after it.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
# Same order as src.models.MODEL_NAMES. Kept here so this launcher does not import TensorFlow.
MODEL_ORDER = (
    "custom_cnn",
    "residual_cnn",
    "mobilenet_v2",
    "efficientnet_b0",
    "mobilenet_v3_small",
)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.parallel:
        raise SystemExit(
            "Do not train these models at the same time on one RTX 5070 Ti. "
            "They would share 16 GB and the same CUDA cores. Each epoch would be "
            "slower, and EfficientNet is likely to run out of memory. "
            "This command already trains them one after another, which is the fast way."
        )

    os.chdir(ROOT)
    config_path = Path(args.config)
    if not config_path.is_file():
        raise SystemExit(f"Config not found: {config_path}")
    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise SystemExit(f"{config_path} must contain a YAML mapping")

    models = args.models or list(MODEL_ORDER)
    unknown = [name for name in models if name not in MODEL_ORDER]
    if unknown:
        raise SystemExit(f"Unknown model name: {', '.join(unknown)}. Choose from: {', '.join(MODEL_ORDER)}")

    data_dir = args.data_dir or str(loaded.get("data_dir") or "")
    print(f"Config: {config_path}")
    print(f"Data: {data_dir or '(value inside the config)'}")
    print("Order: " + ", ".join(models))
    print("One model at a time. Epoch 1 of each model includes graph compile.")
    for name in models:
        batch, learning_rate = _resolved_hparams(loaded, name)
        print(f"  {name}: batch {batch}, learning rate {learning_rate}")

    if args.dry_run:
        for name in models:
            print(" ".join(_command(args, loaded, name, data_dir)))
        return 0

    if not args.skip_gpu_check:
        probe = subprocess.run([sys.executable, str(ROOT / "scripts" / "check_gpu.py")], cwd=ROOT)
        if probe.returncode != 0:
            print("GPU check failed. Training was not started.")
            return probe.returncode

    lock_path = ROOT / "runs" / ".train_all.lock"
    acquire_lock(lock_path)
    finished: list[str] = []
    try:
        for name in models:
            command = _command(args, loaded, name, data_dir)
            print("Starting " + " ".join(command), flush=True)
            result = subprocess.run(command, cwd=ROOT)
            if result.returncode != 0:
                print(f"{name} stopped with exit code {result.returncode}.")
                if finished:
                    print("Finished before the failure: " + ", ".join(finished))
                print(f"If the log says the GPU ran out of memory, lower that model's batch_size in {config_path} and run this command again.")
                print("Runs that already finished are kept. A second launch skips nothing, so pass --models with only the names still missing.")
                return result.returncode
            finished.append(name)
    finally:
        if lock_path.exists():
            lock_path.unlink()

    print("All requested models finished: " + ", ".join(finished))
    data_flag = f' --data-dir "{data_dir}"' if data_dir else ""
    print("Next, score the test set:")
    print(f"  python scripts/evaluate_final.py{data_flag}")
    print("Then write the assignment report:")
    print(f"  python scripts/write_report.py --config {config_path}{data_flag}")
    return 0


def _command(args: argparse.Namespace, loaded: dict, name: str, data_dir: str) -> list[str]:
    batch, learning_rate = _resolved_hparams(loaded, name)
    # configs/rtx4060.yaml -> runs/<model>/rtx4060_<model>
    run_prefix = Path(args.config).stem
    command = [
        sys.executable,
        "-m",
        "src.train",
        "--config",
        str(args.config),
        "--model",
        name,
        "--run-name",
        f"{run_prefix}_{name}",
        "--batch-size",
        str(batch),
        "--learning-rate",
        str(learning_rate),
    ]
    if data_dir:
        command.extend(["--data-dir", data_dir])
    if args.epochs is not None:
        command.extend(["--epochs", str(args.epochs)])
    if args.output_dir:
        command.extend(["--output-dir", args.output_dir])
    return command


def _resolved_hparams(loaded: dict, name: str) -> tuple[object, object]:
    overrides = loaded.get("model_overrides") or {}
    chosen = overrides.get(name) if isinstance(overrides, dict) else None
    if not isinstance(chosen, dict):
        chosen = {}
    batch = chosen.get("batch_size", loaded.get("batch_size"))
    learning_rate = chosen.get("learning_rate", loaded.get("learning_rate"))
    return batch, learning_rate


def acquire_lock(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        text = path.read_text(encoding="utf-8").strip()
        pid = int(text) if text.isdigit() else 0
        if _pid_alive(pid):
            raise SystemExit(
                f"Training is already running (pid {pid}). "
                "Leave that process alone so it keeps the whole GPU. "
                f"If that pid is gone, delete {path} and start again."
            )
        path.unlink()
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        os.write(descriptor, str(os.getpid()).encode("ascii"))
    finally:
        os.close(descriptor)


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes

        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
        if handle:
            ctypes.windll.kernel32.CloseHandle(handle)
            return True
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train every handwriting model on one GPU, one at a time.")
    parser.add_argument("--config", default="configs/rtx5070ti.yaml")
    parser.add_argument("--data-dir", dest="data_dir", default=None)
    parser.add_argument("--output-dir", dest="output_dir", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--models", nargs="+", default=None, help="Subset, in the order given. Default: all five.")
    parser.add_argument("--dry-run", action="store_true", help="Print the training commands and exit.")
    parser.add_argument("--skip-gpu-check", action="store_true", help="Start training without the matmul probe.")
    parser.add_argument("--parallel", action="store_true", help="Refused. One card trains one model at a time.")
    return parser.parse_args(argv)


if __name__ == "__main__":
    sys.exit(main())
