"""Score the finished recognizers on the held-out test split and pick the best.

    python scripts/evaluate_final.py --data-dir "data/cleaned_data(50_50)"

Looks under runs/<model>/ for rtx4060_<model> or rtx5070ti_<model>, preferring
a run with 13,065 classes. Otherwise it uses the newest run that saved a model.
The test images are the files after the first 40, and they are not augmented. Recognition rate in
the table is exact-match test accuracy. Top-5 is reported beside it.

Writes reports/final/comparison.md, comparison.csv, and comparison.png.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MODEL_ORDER = (
    "custom_cnn",
    "residual_cnn",
    "mobilenet_v2",
    "efficientnet_b0",
    "mobilenet_v3_small",
)
RUN_PREFIXES = ("rtx4060_", "rtx5070ti_")


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate the finished handwriting runs on the test split.")
    parser.add_argument("--data-dir", dest="data_dir", default=None)
    parser.add_argument("--output-dir", dest="output_dir", default="runs")
    parser.add_argument("--report-dir", dest="report_dir", default="reports/final")
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument(
        "--from-metrics",
        action="store_true",
        help="Use the test accuracy already stored in metrics.json. Skip a new pass over the test images.",
    )
    args = parser.parse_args(argv)

    models = args.models or list(MODEL_ORDER)
    output_dir = Path(args.output_dir)
    runs = []
    missing = []
    for name in models:
        path = latest_run(output_dir, name, need_weights=not args.from_metrics)
        if path is None:
            missing.append(name)
        else:
            runs.append(path)
    if missing:
        print("No finished run for: " + ", ".join(missing))
    if not runs:
        raise SystemExit("Nothing to evaluate. Train first with scripts/train_all.py.")

    if args.from_metrics:
        from src.evaluate import load_metrics, write_reports

        rows = [load_metrics(path) for path in runs]
        report_dir = Path(args.report_dir)
        report_dir.mkdir(parents=True, exist_ok=True)
        write_reports(rows, report_dir)
    else:
        from src.evaluate import main as evaluate_main

        command = ["--reevaluate", "--report-dir", args.report_dir, "--compare", *[str(path) for path in runs]]
        if args.data_dir:
            command.extend(["--data-dir", args.data_dir])
        evaluate_main(command)
        rows = []
        for path in runs:
            metrics_path = path / "metrics.json"
            if metrics_path.exists():
                rows.append(json.loads(metrics_path.read_text(encoding="utf-8")))

    scored = [row for row in rows if row.get("test_accuracy") is not None]
    if not scored:
        print(f"Comparison written to {args.report_dir}. No test accuracy was recorded.")
        return 0
    best = max(scored, key=lambda row: float(row["test_accuracy"]))
    print(f"Comparison written to {args.report_dir}")
    print(
        "Best test recognition rate: {model} {accuracy:.2%} (top-5 {top5:.2%}).".format(
            model=best.get("model"),
            accuracy=float(best["test_accuracy"]),
            top5=float(best.get("test_top5_accuracy") or best.get("test_topk_accuracy") or 0.0),
        )
    )
    if int(best.get("num_classes") or 0) not in (0, 13065):
        print(
            f"This best model was scored on {best.get('num_classes')} classes. "
            "The assignment asks for 13,065 output neurons."
        )
    return 0


def latest_run(output_dir: Path, model: str, need_weights: bool = True) -> Path | None:
    named = []
    for prefix in RUN_PREFIXES:
        path = output_dir / model / f"{prefix}{model}"
        if _usable(path, need_weights):
            named.append(path)
    chosen = _prefer_full_set(named)
    if chosen is not None:
        return chosen
    root = output_dir / model
    if not root.is_dir():
        return None
    found = [path for path in root.iterdir() if path.is_dir() and _usable(path, need_weights)]
    return _prefer_full_set(found)


def _prefer_full_set(paths: list[Path]) -> Path | None:
    if not paths:
        return None
    full = [path for path in paths if _class_count(path) == 13065]
    pool = full or paths
    # A finished run has best.keras. An interrupted run has only checkpoint.keras.
    return max(pool, key=lambda path: ((path / "best.keras").exists(), _stamp(path)))


def _stamp(path: Path) -> float:
    metrics = path / "metrics.json"
    if metrics.exists():
        return metrics.stat().st_mtime
    return path.stat().st_mtime


def _class_count(path: Path) -> int | None:
    for name in ("metrics.json", "split_summary.json"):
        file_path = path / name
        if not file_path.exists():
            continue
        try:
            loaded = json.loads(file_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        value = loaded.get("num_classes") if isinstance(loaded, dict) else None
        if isinstance(value, int):
            return value
    return None


def _usable(path: Path, need_weights: bool) -> bool:
    if not path.is_dir():
        return False
    has_weights = (path / "best.keras").exists() or (path / "checkpoint.keras").exists()
    has_metrics = (path / "metrics.json").exists()
    if need_weights:
        return has_weights
    return has_metrics


if __name__ == "__main__":
    sys.exit(main())
