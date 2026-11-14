"""Evaluate one saved run, or compare several runs, on the held-out test split.

The test images are rebuilt from the label map saved during training, using
the same first-40 rule. Augmentation is not applied.

Examples, from the repository root:

    python -m src.evaluate --run-dir runs/custom_cnn/20261111-120000
    python -m src.evaluate --compare runs/custom_cnn/20261111-120000 runs/mobilenet_v2/20261111-130000
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger("dit5411.evaluate")


def main(argv: list[str] | None = None) -> None:
    configure_logging()
    args = parse_args(argv)
    run_dirs = [Path(path) for path in (args.run_dir or [])]
    compare_dirs = [Path(path) for path in (args.compare or [])]
    if not run_dirs and not compare_dirs:
        raise SystemExit("Pass --run-dir, --compare, or both.")

    to_evaluate = list(run_dirs)
    if args.reevaluate:
        for path in compare_dirs:
            if path not in to_evaluate:
                to_evaluate.append(path)
    if to_evaluate:
        import tensorflow as tf

        from src.train import configure_device

        device = configure_device(tf)
        logger.info("TensorFlow %s | device: %s", tf.__version__, device)
        for run_dir in to_evaluate:
            evaluate_run(run_dir, data_dir=args.data_dir)

    rows = [load_metrics(path) for path in (compare_dirs or run_dirs)]
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    write_reports(rows, report_dir)
    logger.info("Wrote the comparison to %s", report_dir)


def evaluate_run(run_dir: Path, data_dir: str | None = None) -> dict:
    import tensorflow as tf

    from src.data_pipeline import build_split_index, make_datasets

    run_dir = run_dir.resolve()
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    label_map = json.loads((run_dir / "label_map.json").read_text(encoding="utf-8"))
    class_names = label_map["classes"]
    resolved_data = data_dir or config["data_dir"]
    logger.info("Evaluating %s on %s", run_dir.name, resolved_data)

    index = build_split_index(
        data_dir=resolved_data,
        train_per_class=int(config["train_per_class"]),
        val_ratio=float(config.get("val_ratio") or 0.0),
        class_names=class_names,
        target_samples_per_class=int(config.get("target_samples_per_class") or 40),
    )
    if index.test_paths == []:
        raise RuntimeError(f"{run_dir} has an empty test split.")

    _train, _val, test_ds = make_datasets(
        index,
        image_size=int(config["image_size"]),
        batch_size=int(config["batch_size"]),
        seed=int(config.get("seed") or 5411),
        augmentation=config.get("augmentation") or {},
        cache="",
        repeat_factor=1,
        which=("test",),
    )
    model_path = run_dir / "best.keras"
    if not model_path.exists():
        model_path = run_dir / "checkpoint.keras"
    model = tf.keras.models.load_model(model_path)
    top_k = min(5, index.num_classes)
    metric_name = "top5" if top_k == 5 else "topk"
    model.compile(
        loss="sparse_categorical_crossentropy",
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=top_k, name=metric_name),
        ],
    )
    results = model.evaluate(test_ds, verbose=1, return_dict=True)
    metrics = load_metrics(run_dir) if (run_dir / "metrics.json").exists() else {}
    metrics.update(
        {
            "model": config["model"],
            "run_dir": str(run_dir),
            "num_classes": index.num_classes,
            "image_size": int(config["image_size"]),
            "parameters": int(model.count_params()),
            "top_k": top_k,
            "test_accuracy": float(results["accuracy"]),
            "test_topk_accuracy": float(results[metric_name]),
            "test_images": len(index.test_paths),
        }
    )
    if top_k == 5:
        metrics["test_top5_accuracy"] = metrics["test_topk_accuracy"]
    else:
        metrics.pop("test_top5_accuracy", None)
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info(
        "%s test accuracy %.4f | top-%d %.4f",
        config["model"],
        metrics["test_accuracy"],
        top_k,
        metrics["test_topk_accuracy"],
    )
    return metrics


def load_metrics(run_dir: Path) -> dict:
    path = Path(run_dir) / "metrics.json"
    if not path.exists():
        raise FileNotFoundError(f"No metrics.json in {run_dir}. Train the model or pass --reevaluate.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["run_dir"] = str(run_dir)
    return payload


def write_reports(rows: list[dict], report_dir: Path) -> None:
    columns = [
        "model",
        "num_classes",
        "image_size",
        "parameters",
        "epochs_trained",
        "test_accuracy",
        "top_k",
        "test_topk_accuracy",
        "test_top5_accuracy",
        "device",
        "run_dir",
    ]
    csv_path = report_dir / "comparison.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    lines = [
        "| Model | Classes | Image | Parameters | Epochs | Test accuracy | Top-k | Top-5 | Device |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| {model} | {classes} | {size} | {params} | {epochs} | {acc} | {topk} | {top5} | {device} |".format(
                model=row.get("model", ""),
                classes=row.get("num_classes", ""),
                size=row.get("image_size", ""),
                params=row.get("parameters", ""),
                epochs=row.get("epochs_trained", ""),
                acc=_format_metric(row.get("test_accuracy")),
                topk=_format_metric(row.get("test_topk_accuracy")),
                top5=_format_metric(row.get("test_top5_accuracy")),
                device=row.get("device", ""),
            )
        )
    (report_dir / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    plot_comparison(rows, report_dir / "comparison.png")


def plot_comparison(rows: list[dict], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = []
    for index, row in enumerate(rows, start=1):
        name = str(row.get("model", "model"))
        if sum(1 for item in rows if item.get("model") == row.get("model")) > 1:
            name = f"{name} #{index}"
        labels.append(name)
    accuracy = [float(row.get("test_accuracy") or 0.0) for row in rows]
    top5 = [
        float(row.get("test_top5_accuracy") if row.get("test_top5_accuracy") is not None else row.get("test_topk_accuracy") or 0.0)
        for row in rows
    ]
    positions = list(range(len(labels)))
    width = 0.36
    figure, axis = plt.subplots(figsize=(max(6, len(labels) * 1.6), 4))
    axis.bar([pos - width / 2 for pos in positions], accuracy, width, label="Test accuracy")
    axis.bar([pos + width / 2 for pos in positions], top5, width, label="Top-k accuracy")
    axis.set_xticks(positions)
    axis.set_xticklabels(labels, rotation=15, ha="right")
    axis.set_ylim(0, 1)
    axis.set_ylabel("Accuracy")
    axis.set_title("Held-out test comparison")
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate or compare handwriting recognizers.")
    parser.add_argument("--run-dir", action="append", default=None, help="Run directory to evaluate.")
    parser.add_argument("--compare", nargs="+", default=None, help="Run directories to put in one table.")
    parser.add_argument("--data-dir", dest="data_dir", default=None, help="Override the dataset path stored in the run.")
    parser.add_argument("--report-dir", dest="report_dir", default="reports")
    parser.add_argument(
        "--reevaluate",
        action="store_true",
        help="Recompute test metrics for every directory passed to --compare.",
    )
    return parser.parse_args(argv)


def configure_logging() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8")
            except Exception:
                pass
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")


def _format_metric(value) -> str:
    if value is None:
        return ""
    return f"{float(value):.4f}"


if __name__ == "__main__":
    main()
