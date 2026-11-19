"""Train one character recognizer.

Run from the repository root:

    python -m src.train --config configs/laptop_debug.yaml --data-dir data/cleaned_data
    python -m src.train --model mobilenet_v2 --max-classes 50 --epochs 10
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

from src.cpu_threads import (
    install_cpu_thread_env,
    install_gpu_performance_env,
    resolve_thread_settings,
)

# TensorFlow reads the thread pool when it is imported. Set it first, and only
# for a real training launch. Importing this module must not change a process
# that is already running.
if __name__ == "__main__":
    _intra, _inter = resolve_thread_settings(sys.argv[1:])
    install_cpu_thread_env(_intra, _inter)
    install_gpu_performance_env(sys.argv[1:])

from src.data_pipeline import attach_gpu_augmentation, build_split_index, make_datasets
from src.models import MODEL_NAMES, build_model

logger = logging.getLogger("dit5411.train")

DEFAULTS = {
    "data_dir": "data/cleaned_data",
    "output_dir": "runs",
    "model": "custom_cnn",
    "image_size": 64,
    "batch_size": 32,
    "epochs": 15,
    "patience": 5,
    "learning_rate": 0.001,
    "seed": 5411,
    "train_per_class": 40,
    "val_ratio": 0.0,
    "target_samples_per_class": 200,
    "max_classes": None,
    "classes_file": None,
    "weights": "imagenet",
    "freeze_base": False,
    "cache": "",
    "shuffle_buffer": 8192,
    "cpu_threads": 10,
    "inter_op_threads": 1,
    "gpu_performance": False,
    "jit_compile": False,
    "augment_on_gpu": False,
    "drop_remainder": False,
    "host_cache": False,
    "input_threads": 0,
    "deterministic": True,
    "target_epoch_seconds": None,
    "model_overrides": {},
    "initial_weights": None,
    "run_name": None,
    "augmentation": {
        "rotation_degrees": 15,
        "translation": 0.10,
        "zoom": 0.15,
        "scale": 0.10,
        "shear_x": 0.25,
        "shear_y": 0.15,
        "clean_probability": 0.5,
    },
}


def main(argv: list[str] | None = None) -> None:
    configure_logging()
    config = parse_config(argv)

    import tensorflow as tf

    # Memory growth has to be set before the first GPU operation. set_seed
    # touches TensorFlow, so the device is configured first.
    device = configure_device(
        tf,
        cpu_threads=int(config["cpu_threads"]),
        inter_op_threads=int(config["inter_op_threads"]),
        gpu_performance=bool(config.get("gpu_performance")),
    )
    set_seed(int(config["seed"]))
    logger.info(
        "TensorFlow %s | device: %s | policy: %s | CPU threads: %s intra-op, %s inter-op",
        tf.__version__,
        device,
        tf.keras.mixed_precision.global_policy().name,
        int(config["cpu_threads"]),
        int(config["inter_op_threads"]),
    )

    index = build_split_index(
        data_dir=config["data_dir"],
        train_per_class=int(config["train_per_class"]),
        val_ratio=float(config["val_ratio"]),
        max_classes=_optional_int(config.get("max_classes")),
        classes_file=config.get("classes_file") or None,
        target_samples_per_class=int(config["target_samples_per_class"]),
    )
    logger.info("\n%s", index.summary())
    if config.get("gpu_performance") and not config.get("max_classes") and index.num_classes < 13065:
        logger.warning(
            "This run has %s classes. The assignment asks for 13,065 output neurons. "
            "Pass the full cleaned_data folder with --data-dir when you report the final result.",
            index.num_classes,
        )

    image_size = int(config["image_size"])
    batch_size = int(config["batch_size"])
    augment_on_gpu = bool(config.get("augment_on_gpu"))
    train_ds, val_ds, test_ds = make_datasets(
        index,
        image_size=image_size,
        batch_size=batch_size,
        seed=int(config["seed"]),
        augmentation=config.get("augmentation") or {},
        cache=str(config.get("cache") or ""),
        shuffle_buffer=int(config["shuffle_buffer"]),
        augment_on_gpu=augment_on_gpu,
        drop_remainder=bool(config.get("drop_remainder")),
        input_threads=int(config.get("input_threads") or 0),
        deterministic=bool(config.get("deterministic", True)),
        host_cache=bool(config.get("host_cache")),
    )
    views = len(index.train_paths) * index.repeats
    steps = views // batch_size if config.get("drop_remainder") else math.ceil(views / batch_size)
    logger.info(
        "Steps this epoch: %s (%s augmented views, batch %s).",
        f"{steps:,}",
        f"{views:,}",
        batch_size,
    )
    if config.get("host_cache") and not augment_on_gpu:
        logger.info(
            "The warp runs on the CPU. The compiled step is the network only. "
            "The first epoch of this process also copies the training images into RAM and compiles the graph. "
            "Judge the pace from ms/step on the epoch after that."
        )
    else:
        logger.info("The first epoch in this process also compiles the graph.")

    model = build_model(
        name=config["model"],
        input_shape=(image_size, image_size, 1),
        num_classes=index.num_classes,
        weights=config.get("weights"),
        freeze_base=bool(config["freeze_base"]),
    )
    resume_path = _resume_checkpoint(config)
    if resume_path is not None:
        _load_matching_weights(model, resume_path)
    if augment_on_gpu:
        model = attach_gpu_augmentation(model, image_size, config.get("augmentation") or {})
        logger.info("Augmentation is one composed warp on the device.")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(float(config["learning_rate"])),
        loss="sparse_categorical_crossentropy",
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
        jit_compile=bool(config.get("jit_compile")),
    )
    logger.info("Parameters: %s", f"{model.count_params():,}")

    run_dir = make_run_dir(config)
    logger.info("Writing this run to %s", run_dir)
    write_json(run_dir / "config.json", _public_config(config, device, tf.__version__))
    write_json(run_dir / "label_map.json", {"classes": index.class_names})
    write_json(
        run_dir / "split_summary.json",
        {
            "num_classes": index.num_classes,
            "train_images": len(index.train_paths),
            "val_images": len(index.val_paths),
            "test_images": len(index.test_paths),
            "repeats": index.repeats,
            "train_counts": index.train_counts,
            "test_counts": index.test_counts,
            "warnings": index.warnings,
        },
    )
    (run_dir / "model_summary.txt").write_text(_summary_text(model), encoding="utf-8")

    monitor = "val_accuracy" if val_ds is not None else "accuracy"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(run_dir / "checkpoint.keras"),
            monitor=monitor,
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor,
            mode="max",
            patience=int(config["patience"]),
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss" if val_ds is not None else "loss",
            factor=0.5,
            patience=max(1, int(config["patience"]) // 2),
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(str(run_dir / "training_log.csv"), append=resume_path is not None),
    ]
    initial_epoch = _completed_epochs(run_dir / "training_log.csv") if resume_path is not None else 0
    if initial_epoch:
        logger.info(
            "Continuing from epoch %s. The saved weights are already loaded. %s epochs remain in this plan.",
            initial_epoch + 1,
            max(0, int(config["epochs"]) - initial_epoch),
        )
    target_seconds = config.get("target_epoch_seconds")
    if target_seconds:
        callbacks.append(make_epoch_time_hint(tf, float(target_seconds)))

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        initial_epoch=initial_epoch,
        epochs=int(config["epochs"]),
        callbacks=callbacks,
        shuffle=False,
        verbose=1,
    )
    history_path = run_dir / "history.json"
    write_json(history_path, {key: [float(value) for value in values] for key, values in history.history.items()})
    plot_history(history.history, run_dir / "training_curves.png")

    model.save(str(run_dir / "best.keras"))
    metrics = {
        "model": config["model"],
        "device": device,
        "tensorflow": tf.__version__,
        "parameters": int(model.count_params()),
        "num_classes": index.num_classes,
        "image_size": image_size,
        "batch_size": batch_size,
        "epochs_requested": int(config["epochs"]),
        "epochs_trained": len(history.history.get("loss", [])),
        "repeats": index.repeats,
        "train_images": len(index.train_paths),
        "test_images": len(index.test_paths),
        "run_dir": str(run_dir),
    }
    if test_ds is None:
        logger.warning("The test split is empty, so test accuracy was not computed.")
        metrics["test_accuracy"] = None
    else:
        top_k = min(5, index.num_classes)
        metric_name = "top5" if top_k == 5 else "topk"
        model.compile(
            optimizer=tf.keras.optimizers.Adam(float(config["learning_rate"])),
            loss="sparse_categorical_crossentropy",
            metrics=[
                tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
                tf.keras.metrics.SparseTopKCategoricalAccuracy(k=top_k, name=metric_name),
            ],
        )
        results = model.evaluate(test_ds, verbose=1, return_dict=True)
        metrics["test_accuracy"] = float(results["accuracy"])
        metrics["top_k"] = top_k
        metrics["test_topk_accuracy"] = float(results[metric_name])
        if top_k == 5:
            metrics["test_top5_accuracy"] = metrics["test_topk_accuracy"]
        logger.info(
            "Test accuracy: %.4f | top-%d accuracy: %.4f",
            metrics["test_accuracy"],
            top_k,
            metrics["test_topk_accuracy"],
        )
    write_json(run_dir / "metrics.json", metrics)
    logger.info("Saved model and metrics to %s", run_dir)


def parse_config(argv: list[str] | None) -> dict:
    parser = argparse.ArgumentParser(description="Train a Traditional Chinese handwriting recognizer.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data-dir", dest="data_dir", default=None)
    parser.add_argument("--output-dir", dest="output_dir", default=None)
    parser.add_argument("--model", choices=MODEL_NAMES, default=None)
    parser.add_argument("--image-size", dest="image_size", type=int, default=None)
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--learning-rate", dest="learning_rate", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--train-per-class", dest="train_per_class", type=int, default=None)
    parser.add_argument("--val-ratio", dest="val_ratio", type=float, default=None)
    parser.add_argument("--target-samples-per-class", dest="target_samples_per_class", type=int, default=None)
    parser.add_argument("--max-classes", dest="max_classes", type=int, default=None)
    parser.add_argument("--classes-file", dest="classes_file", default=None)
    parser.add_argument("--weights", default=None, help="imagenet or none")
    parser.add_argument("--freeze-base", dest="freeze_base", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--cache", default=None, help="'memory', a file prefix, or empty to disable")
    parser.add_argument("--shuffle-buffer", dest="shuffle_buffer", type=int, default=None)
    parser.add_argument(
        "--cpu-threads",
        dest="cpu_threads",
        type=int,
        default=None,
        help="Intra-op threads for a new process. Default 10. Inter-op stays at 1.",
    )
    parser.add_argument("--run-name", dest="run_name", default=None)
    parser.add_argument(
        "--initial-weights",
        dest="initial_weights",
        default=None,
        help="Saved .keras model to load before training.",
    )
    parser.add_argument("--inter-op-threads", dest="inter_op_threads", type=int, default=None)
    args = parser.parse_args(argv)

    file_config = {}
    config_path = Path(args.config)
    if config_path.exists():
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"{config_path} must contain a YAML mapping")
        file_config = loaded
    elif args.config != "configs/default.yaml":
        raise FileNotFoundError(f"Config file not found: {config_path}")

    config = _deep_merge(DEFAULTS, file_config)
    cli_keys = set()
    for key, value in vars(args).items():
        if key == "config" or value is None:
            continue
        config[key] = value
        cli_keys.add(key)
    overrides = config.get("model_overrides") or {}
    chosen = overrides.get(config.get("model")) if isinstance(overrides, dict) else None
    if isinstance(chosen, dict):
        for key, value in chosen.items():
            if key not in cli_keys:
                config[key] = value
    if config.get("max_classes") in (0, "0"):
        config["max_classes"] = None
    return config


def configure_device(
    tf,
    cpu_threads: int | None = None,
    inter_op_threads: int | None = None,
    gpu_performance: bool = False,
) -> str:
    """Use the GPU with memory growth and mixed precision, or stay on the CPU."""

    if cpu_threads is not None:
        _apply_cpu_threads(tf, int(cpu_threads), inter_op_threads)
    tf.config.set_soft_device_placement(True)
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        tf.keras.mixed_precision.set_global_policy("float32")
        if gpu_performance:
            raise SystemExit(
                "This config expects a CUDA GPU, and TensorFlow sees none. "
                "Train from Ubuntu with the project virtualenv active. "
                "On the RTX 4060, install the CUDA libraries with: "
                'python -m pip install "tensorflow[and-cuda]" '
                "and then run: python scripts/check_gpu.py. "
                "An RTX 5070 Ti also needs a TensorFlow build that includes sm_120 kernels."
            )
        return "CPU"
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as exc:
            logger.warning("Could not set memory growth on %s: %s", gpu, exc)
        try:
            details = tf.config.experimental.get_device_details(gpu)
        except Exception:
            details = {}
        logger.info("GPU %s details: %s", gpu.name, details)
    try:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
    except Exception as exc:
        logger.warning("Mixed precision is unavailable (%s). Using float32.", exc)
        tf.keras.mixed_precision.set_global_policy("float32")
    if gpu_performance:
        _probe_gpu(tf)
    names = ", ".join(device.name for device in gpus)
    return f"GPU ({names})"


def _probe_gpu(tf) -> None:
    """Fail before a long run if the Blackwell kernels are missing."""

    try:
        with tf.device("/GPU:0"):
            sample = tf.random.normal((512, 512))
            _ = tf.matmul(sample, sample).numpy()
    except Exception as exc:
        raise SystemExit(
            "The GPU is listed, but a test matmul failed. "
            "An RTX 5070 Ti is Blackwell, compute capability 12.0. "
            "Official TensorFlow 2.21 wheels often ship without sm_120 kernels, "
            "and a PTX fallback can die with CUDA_ERROR_INVALID_PTX.\n"
            f"Original error: {exc}\n"
            "Use Ubuntu or WSL2 and a wheel built for sm_120. "
            "An unofficial TensorFlow 2.21 Linux build is at "
            "https://github.com/Syraxius/TensorflowDockerBuilder/releases/tag/v2.21.0-sm120-cuda128-py310-py313\n"
            "If the error names the allocator, retry with TF_GPU_ALLOCATOR=default."
        ) from exc
    logger.info("GPU matmul succeeded. The card can run this training step.")


def _apply_cpu_threads(tf, cpu_threads: int, inter_op_threads: int | None = None) -> None:
    """Request the thread pools. The environment set at launch is what sticks."""

    inter = 1 if inter_op_threads is None else int(inter_op_threads)
    try:
        tf.config.threading.set_intra_op_parallelism_threads(cpu_threads)
        tf.config.threading.set_inter_op_parallelism_threads(inter)
    except RuntimeError as exc:
        logger.warning("CPU thread pools were already fixed (%s).", exc)


def make_epoch_time_hint(tf, target_seconds: float):
    """Callback that logs epoch wall time. Epoch 1 includes the XLA compile."""

    class EpochTimeHint(tf.keras.callbacks.Callback):
        def __init__(self, seconds: float):
            super().__init__()
            self.target_seconds = float(seconds)
            self._started = 0.0
            self._epochs_seen = 0

        def on_epoch_begin(self, epoch, logs=None):
            self._started = time.perf_counter()

        def on_epoch_end(self, epoch, logs=None):
            elapsed = time.perf_counter() - self._started
            self._epochs_seen += 1
            logger.info("Epoch %s took %.1f seconds.", epoch + 1, elapsed)
            if self._epochs_seen == 1:
                logger.info(
                    "This first epoch includes graph compile and the one-time image load. "
                    "Judge the one-minute target from the next epoch."
                )
            elif elapsed > self.target_seconds * 1.15:
                logger.warning(
                    "Epoch took %.0f seconds, above the %.0f second target. "
                    "The card is already on mixed precision. Raise the batch size if memory remains.",
                    elapsed,
                    self.target_seconds,
                )

    return EpochTimeHint(target_seconds)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import tensorflow as tf

    tf.keras.utils.set_random_seed(seed)


def _resume_checkpoint(config: dict) -> Path | None:
    """Checkpoint to continue, or None for a new run.

    An explicit ``--initial-weights`` wins. Otherwise the named run's own
    ``checkpoint.keras`` is used, so stopping and starting ``train_all`` keeps
    the epochs that already finished.
    """

    explicit = config.get("initial_weights") or None
    if explicit:
        path = Path(str(explicit))
        if not path.exists():
            raise SystemExit(f"Initial weights were not found: {path}")
        return path
    run_name = config.get("run_name")
    if not run_name or config.get("max_classes"):
        return None
    candidate = Path(config["output_dir"]) / str(config["model"]) / str(run_name) / "checkpoint.keras"
    if candidate.exists():
        return candidate
    return None


def _completed_epochs(log_path: Path) -> int:
    if not log_path.exists():
        return 0
    rows = [line for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return max(0, len(rows) - 1)


def _checkpoint_is_wrapped(path: Path) -> bool:
    """True when the file is the full training model, not the bare classifier."""

    if path.suffix != ".keras":
        return False
    import zipfile

    with zipfile.ZipFile(path) as zipped:
        if "config.json" not in zipped.namelist():
            return False
        text = zipped.read("config.json").decode("utf-8")
    return "keep_clean" in text or "aug_composed" in text or "aug_rotation" in text


def _load_matching_weights(model, path: Path) -> None:
    """Load a checkpoint, including one saved from the older five-warp wrapper."""

    import tensorflow as tf

    if not _checkpoint_is_wrapped(path):
        model.load_weights(str(path))
        logger.info("Loaded initial weights from %s", path)
        return

    loaded = tf.keras.models.load_model(path, compile=False, safe_mode=False)
    donor = _named_submodel(loaded, model.name)
    if donor is None:
        raise SystemExit(f"No weights named {model.name} were found in {path}")
    copied = 0
    for layer in donor.layers:
        try:
            target = model.get_layer(layer.name)
        except ValueError:
            continue
        weights = layer.get_weights()
        if not weights:
            continue
        current = target.get_weights()
        if [tuple(item.shape) for item in current] != [tuple(item.shape) for item in weights]:
            continue
        target.set_weights(weights)
        copied += 1
    if copied == 0:
        raise SystemExit(f"No matching layers could be copied from {path}")
    logger.info("Copied %s layers from %s", copied, path)


def _named_submodel(model, name: str):
    import tensorflow as tf

    matches = [layer for layer in getattr(model, "layers", []) if isinstance(layer, tf.keras.Model) and layer.name == name]
    if len(matches) == 1:
        return matches[0]
    if getattr(model, "name", None) == name:
        return model
    return matches[0] if matches else None


def make_run_dir(config: dict) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    name = config.get("run_name") or stamp
    run_dir = Path(config["output_dir"]) / str(config["model"]) / str(name)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def plot_history(history: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history.get("loss", []), label="train")
    if "val_loss" in history:
        axes[0].plot(history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    accuracy = history.get("accuracy", [])
    axes[1].plot(accuracy, label="train")
    if "val_accuracy" in history:
        axes[1].plot(history["val_accuracy"], label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def configure_logging() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8")
            except Exception:
                pass
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _summary_text(model) -> str:
    lines: list[str] = []
    model.summary(print_fn=lines.append)
    return "\n".join(lines) + "\n"


def _public_config(config: dict, device: str, tf_version: str) -> dict:
    public = {key: value for key, value in config.items()}
    public["device"] = device
    public["tensorflow"] = tf_version
    return public


def _optional_int(value):
    if value is None or value == "":
        return None
    return int(value)


def _deep_merge(base: dict, extra: dict) -> dict:
    merged = dict(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


if __name__ == "__main__":
    main()
