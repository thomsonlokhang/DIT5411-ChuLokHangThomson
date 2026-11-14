"""Train one character recognizer.

Run from the repository root:

    python -m src.train --config configs/laptop_debug.yaml --data-dir data/cleaned_data
    python -m src.train --model mobilenet_v2 --max-classes 50 --epochs 10
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

from src.cpu_threads import INTER_OP_THREADS, install_cpu_thread_env, resolve_cpu_threads

# TensorFlow reads the thread pool when it is imported. Set it first, and only
# for a real training launch. Importing this module must not change a process
# that is already running.
if __name__ == "__main__":
    install_cpu_thread_env(resolve_cpu_threads(sys.argv[1:]))

from src.data_pipeline import build_split_index, make_datasets
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
    device = configure_device(tf, cpu_threads=int(config["cpu_threads"]))
    set_seed(int(config["seed"]))
    logger.info(
        "TensorFlow %s | device: %s | policy: %s | CPU threads: %s intra-op, %s inter-op",
        tf.__version__,
        device,
        tf.keras.mixed_precision.global_policy().name,
        int(config["cpu_threads"]),
        INTER_OP_THREADS,
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

    image_size = int(config["image_size"])
    batch_size = int(config["batch_size"])
    train_ds, val_ds, test_ds = make_datasets(
        index,
        image_size=image_size,
        batch_size=batch_size,
        seed=int(config["seed"]),
        augmentation=config.get("augmentation") or {},
        cache=str(config.get("cache") or ""),
        shuffle_buffer=int(config["shuffle_buffer"]),
    )

    model = build_model(
        name=config["model"],
        input_shape=(image_size, image_size, 1),
        num_classes=index.num_classes,
        weights=config.get("weights"),
        freeze_base=bool(config["freeze_base"]),
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(float(config["learning_rate"])),
        loss="sparse_categorical_crossentropy",
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    initial_weights = config.get("initial_weights") or None
    if initial_weights:
        model.load_weights(str(initial_weights))
        logger.info("Loaded initial weights from %s", initial_weights)
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
        tf.keras.callbacks.CSVLogger(str(run_dir / "training_log.csv")),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
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
    for key, value in vars(args).items():
        if key == "config" or value is None:
            continue
        config[key] = value
    if config.get("max_classes") in (0, "0"):
        config["max_classes"] = None
    return config


def configure_device(tf, cpu_threads: int | None = None) -> str:
    """Use the GPU with memory growth and mixed precision, or stay on the CPU."""

    if cpu_threads is not None:
        _apply_cpu_threads(tf, int(cpu_threads))
    tf.config.set_soft_device_placement(True)
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        tf.keras.mixed_precision.set_global_policy("float32")
        return "CPU"
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as exc:
            logger.warning("Could not set memory growth on %s: %s", gpu, exc)
    try:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
    except Exception as exc:
        logger.warning("Mixed precision is unavailable (%s). Using float32.", exc)
        tf.keras.mixed_precision.set_global_policy("float32")
    names = ", ".join(device.name for device in gpus)
    return f"GPU ({names})"


def _apply_cpu_threads(tf, cpu_threads: int) -> None:
    """Request the thread pools. The environment set at launch is what sticks."""

    try:
        tf.config.threading.set_intra_op_parallelism_threads(cpu_threads)
        tf.config.threading.set_inter_op_parallelism_threads(INTER_OP_THREADS)
    except RuntimeError as exc:
        logger.warning("CPU thread pools were already fixed (%s).", exc)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import tensorflow as tf

    tf.keras.utils.set_random_seed(seed)


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
