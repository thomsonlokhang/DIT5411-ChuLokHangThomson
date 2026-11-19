"""Write the assignment README from the finished test scores.

    python scripts/write_report.py --output ASSIGNMENT_README.md

The brief asks the GitHub readme to include the augmentation method, a
description of several models, their recognition accuracy on the testing set,
and references. This script fills the accuracy table from metrics.json and
leaves a blank cell when a model has not been trained. It does not invent a score.

Default output is ASSIGNMENT_README.md so the setup guide in README.md stays
in place. Pass --output README.md when you want this report to be the GitHub page.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
MODEL_ORDER = (
    "custom_cnn",
    "residual_cnn",
    "mobilenet_v2",
    "efficientnet_b0",
    "mobilenet_v3_small",
)
MODEL_TEXT = {
    "custom_cnn": (
        "A network written for this assignment. The input is one grayscale image. "
        "Three stages use 32, 64, then 128 filters. Each stage is two 3×3 convolutions, "
        "batch normalisation, ReLU, 2×2 max pooling, and dropout 0.2. Global average "
        "pooling feeds a 512-unit ReLU layer, dropout 0.4, and a float32 softmax. "
        "The softmax has one neuron per character."
    ),
    "residual_cnn": (
        "A second architecture, also written for this assignment. A 3×3 stem produces "
        "32 channels. Three residual stages follow, at 32, 64, and 128 channels. The "
        "64 and 128 stages downsample with stride 2, and a projection shortcut is used "
        "when the shape changes. The classifier head matches the custom CNN. The shortcut "
        "keeps early stroke features available to the later layers."
    ),
    "mobilenet_v2": (
        "ImageNet MobileNetV2 (alpha 1.0) without its original classifier. The grayscale "
        "glyph is stacked into three channels, scaled to [0, 255], then mapped to [-1, 1] "
        "with the MobileNetV2 preprocessing. A new 512-unit head and a float32 softmax "
        "replace the ImageNet head. The base is fine-tuned rather than frozen, because "
        "ImageNet features were not trained on ink."
    ),
    "efficientnet_b0": (
        "ImageNet EfficientNetB0 without its original classifier. Grayscale is stacked "
        "into three channels and scaled to [0, 255]. Current Keras EfficientNet rescales "
        "inside the network. The new head is a 512-unit layer, dropout, and a float32 "
        "softmax over the characters. The base is fine-tuned."
    ),
    "mobilenet_v3_small": (
        "ImageNet MobileNetV3Small without its original classifier. It uses the same "
        "[0, 255] input convention as EfficientNet, then a new 512-unit head and a "
        "float32 softmax. The base is fine-tuned. This is a third pretrained architecture, "
        "smaller than EfficientNetB0, so the comparison is not two copies of one network."
    ),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write the assignment report from saved test metrics.")
    parser.add_argument("--output", default="ASSIGNMENT_README.md")
    parser.add_argument("--config", default=None)
    parser.add_argument("--data-dir", dest="data_dir", default=None)
    parser.add_argument("--output-dir", dest="output_dir", default="runs")
    parser.add_argument("--name", default="Chu Lok Hang Thomson")
    parser.add_argument("--github", default="https://github.com/thomsonlokhang/DIT5411-ChuLokHangThomson")
    args = parser.parse_args(argv)

    runs = {name: _latest_run(Path(args.output_dir), name) for name in MODEL_ORDER}
    rows = []
    for name in MODEL_ORDER:
        path = runs[name]
        metrics = _read_json(path / "metrics.json") if path else None
        run_config = _read_json(path / "config.json") if path else None
        rows.append({"name": name, "path": path, "metrics": metrics, "config": run_config})

    config_path = args.config or _infer_config_path(rows) or "configs/rtx5070ti.yaml"
    config = _load_yaml(ROOT / config_path)
    data_dir = args.data_dir or _infer_data_dir(rows) or "data/cleaned_data"
    text = render(
        args.name,
        args.github,
        config,
        rows,
        config_path=config_path,
        data_dir=data_dir,
    )
    destination = Path(args.output)
    if not destination.is_absolute():
        destination = ROOT / destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text, encoding="utf-8")
    trained = sum(1 for row in rows if row["metrics"] and row["metrics"].get("test_accuracy") is not None)
    print(f"Wrote {destination} ({trained} of {len(MODEL_ORDER)} models have a test accuracy).")
    return 0


def render(
    student: str,
    github: str,
    config: dict,
    rows: list[dict],
    config_path: str = "configs/rtx5070ti.yaml",
    data_dir: str = "data/cleaned_data",
) -> str:
    scored = [row for row in rows if row["metrics"] and row["metrics"].get("test_accuracy") is not None]
    best = max(scored, key=lambda row: float(row["metrics"]["test_accuracy"])) if scored else None
    class_counts = sorted({int(row["metrics"]["num_classes"]) for row in scored if row["metrics"].get("num_classes")})
    lines = [
        "# DIT5411 Machine Learning — Character Recognition",
        "",
        f"- Student: {student}",
        "- Course: THEi DIT5411 Machine Learning",
        "- Assignment period: 2026-11-11 to 2026-11-23",
        f"- Code: {github}",
        "",
        "## Objective",
        "",
        "Build a TensorFlow program that recognises Traditional Chinese handwritten characters. "
        "The official split uses the first 40 samples of each character for training and the "
        "remaining samples for testing. The network has several convolutional layers and a "
        "softmax with one output neuron per character. Because each character supplies only "
        "40 training images, training uses augmented views, at least 200 per character. "
        "Several architectures are trained, and the one with the highest recognition rate "
        "on the testing set is the selected model.",
        "",
        "## Dataset and split",
        "",
        "The images come from the Tegaki-based Traditional Chinese Handwriting Dataset. "
        "Files are ordered by the sample number in the filename (`永_2` before `永_10`), "
        "then cut. The list is not shuffled before the cut, so the test membership does not "
        "depend on Windows or Linux directory order. Every one of those 40 images is used "
        "for training (`val_ratio` 0). Images after the first 40 are the test set. "
        "A character with fewer than 40 files keeps all of them for training and contributes "
        "no test images.",
        "",
        _class_note(class_counts),
        "",
        "## Augmented image creation",
        "",
        _augmentation_section(config, rows),
        "",
        "### Does rotation raise accuracy?",
        "",
        "Yes, when the angle stays small and the upright glyph is still shown. Handwritten "
        "characters are slightly tilted, so a rotation of ±15° teaches a variation that "
        "appears in real writing. The test images are upright and unwarped. If every "
        "training view is rotated, sheared, and scaled, the network fits that warped look "
        "and its test score drops. An earlier run that warped every view reached about 76% "
        "on those warped training images and about 20.8% exact accuracy on the clean test set. "
        "Large angles such as 90° or 180° are not used: they are not how these characters "
        "are written, and some shapes can be confused with a different character. "
        "The originals on disk are never rotated. Each view is drawn at random during training.",
        "",
        "## Models",
        "",
        "Five architectures share the same grayscale input, the same split, and the same "
        "augmentation. The assignment asks for several models with different layers. "
        "The two networks written for this project and the three ImageNet networks are "
        "different families. All of them end in a float32 softmax so mixed precision on "
        "the GPU does not compute the loss in float16.",
        "",
    ]
    for row in rows:
        metrics = row["metrics"] or {}
        lines.append(f"### {row['name']}")
        lines.append("")
        lines.append(MODEL_TEXT[row["name"]])
        lines.append("")
        if metrics:
            lines.append(
                "This run used {size}×{size} input, batch size {batch}, learning rate {lr}, "
                "{epochs} epoch(s), and {params} parameters.".format(
                    size=metrics.get("image_size", "the configured"),
                    batch=metrics.get("batch_size", "the configured"),
                    lr=_learning_rate(row),
                    epochs=_epochs_trained(row),
                    params=f"{int(metrics['parameters']):,}" if metrics.get("parameters") else "the model's",
                )
            )
            lines.append("")
        else:
            lines.append("No finished test score is stored for this architecture yet.")
            lines.append("")

    lines.extend(
        [
            "## Recognition accuracy on the testing set",
            "",
            "The test images are not augmented. Recognition rate means exact-match accuracy: "
            "the predicted character is the true character. Top-5 counts a success when the "
            "true character is one of the five highest probabilities. That second number "
            "matters once the label set is in the thousands, and the selection below uses "
            "the exact-match rate, which is what the brief calls the recognition rate.",
            "",
            "| Model | Classes | Image | Parameters | Epochs | Test recognition rate | Top-5 | Device |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in rows:
        metrics = row["metrics"] or {}
        lines.append(
            "| {name} | {classes} | {size} | {params} | {epochs} | {acc} | {top5} | {device} |".format(
                name=row["name"],
                classes=_cell(metrics.get("num_classes")),
                size=_cell(metrics.get("image_size")),
                params=f"{int(metrics['parameters']):,}" if metrics.get("parameters") else "—",
                epochs=_epochs_trained(row),
                acc=_percent(metrics.get("test_accuracy")),
                top5=_percent(metrics.get("test_top5_accuracy")),
                device=_cell(metrics.get("device")) or "—",
            )
        )
    lines.append("")
    if best is None:
        lines.append(
            "No test accuracy has been recorded yet. After training, run "
            "`python scripts/evaluate_final.py` and then run this script again."
        )
    else:
        metrics = best["metrics"]
        lines.append(
            "The best model on the testing set is **{name}**, with a recognition rate of {acc} "
            "(top-5 {top5}). The weights and the log are in `{path}`.".format(
                name=best["name"],
                acc=_percent(metrics.get("test_accuracy")),
                top5=_percent(metrics.get("test_top5_accuracy")),
                path=_display_path(best["path"]),
            )
        )
        if int(metrics.get("num_classes") or 0) not in (0, 13065):
            lines.append("")
            lines.append(
                f"This table is for {metrics.get('num_classes')} classes. "
                "The brief asks for a softmax of 13,065 neurons. Train again with the full "
                "`cleaned_data` folder before treating this number as the submitted result."
            )
    lines.extend(
        [
            "",
            "## How to reproduce",
            "",
            "From the repository root, with the project virtualenv active:",
            "",
            "```bash",
            "python scripts/check_gpu.py",
            f"python scripts/train_all.py --config {config_path} --data-dir {_shell_quote(data_dir)}",
            "python scripts/evaluate_final.py --from-metrics",
            f"python scripts/write_report.py --config {config_path} --data-dir {_shell_quote(data_dir)}",
            "```",
            "",
            "`train_all.py` trains `custom_cnn`, `residual_cnn`, `mobilenet_v2`, `efficientnet_b0`, "
            "and `mobilenet_v3_small` one after another. One GPU trains one model at a time.",
            "",
            "## References",
            "",
            "- Po-Chuan Chen, Traditional Chinese Handwriting Dataset, 2020. "
            "https://github.com/AI-FREE-Team/Traditional-Chinese-Handwriting-Dataset",
            "- Full 13,065-character image release used by the loader: "
            "https://github.com/chenkenanalytic/handwritting_data_all",
            "- Darshita Jain, Introduction to OpenCV (rotation, shearing, and scaling, the geometric "
            "transforms cited in the assignment brief). "
            "https://github.com/darshitajain/Introduction-to-OpenCV",
            "- Kim Analytics, Handwritten Digit Recognition using Keras and TensorFlow "
            "(format reference named in the brief). "
            "https://github.com/kimanalytics/Handwritten-Digit-Recognition-using-Keras-and-TensorFlow",
            "- Martín Abadi et al., TensorFlow: large-scale machine learning on heterogeneous systems, 2015. "
            "https://www.tensorflow.org/",
            "",
        ]
    )
    return "\n".join(lines)


def _augmentation_section(config: dict, rows: list[dict]) -> str:
    factors = _factors_from(config, rows)
    target = int(factors.get("target_samples_per_class") or config.get("target_samples_per_class") or 200)
    train_per_class = int(factors.get("train_per_class") or config.get("train_per_class") or 40)
    repeats = max(1, -(-target // train_per_class))
    clean = factors.get("clean_probability")
    if clean is None:
        clean_sentence = (
            "The saved run did not record a clean-image probability, so check that run's "
            "`config.json` before quoting a fraction."
        )
    else:
        clean_sentence = (
            f"A fraction {float(clean):.2f} of each training batch is left as the original "
            "resized glyph. The rest receives the random transform."
        )
    described = _described_run(rows)
    gpu_augment = bool((described or {}).get("augment_on_gpu"))
    if gpu_augment:
        where = (
            "On the RTX 5070 Ti configuration these layers run inside the training step on the GPU. "
            "Evaluation turns training mode off, so the saved network does not warp the test image."
        )
    else:
        where = (
            "These layers run in the training data pipeline. The test pipeline does not apply them."
        )
    return "\n".join(
        [
            "Augmented images are created on the fly with TensorFlow image layers. "
            "Nothing is written back to the dataset folder. Each training image is resized "
            "to grayscale and scaled to [0, 1]. A training view can then receive:",
            "",
            f"- rotation of ±{factors.get('rotation_degrees', 15)} degrees",
            f"- translation of ±{factors.get('translation', 0.10)} of the image size, on both axes",
            f"- zoom of ±{factors.get('zoom', 0.15)}",
            f"- a second anisotropic scale of about ±{factors.get('scale', 0.10)}, wider on the x axis",
            f"- shear of ±{factors.get('shear_x', 0.25)} horizontally and ±{factors.get('shear_y', 0.15)} vertically",
            "",
            "Empty pixels are filled with white (1.0). A black fill would draw a box around the ink "
            "and give the model a shortcut that is not part of the character. "
            f"The {train_per_class} real training images of a full class are repeated {repeats} times "
            f"per epoch, which is {train_per_class * repeats} views and meets the requirement of at least "
            f"{target} samples per character. Each repeat draws a new random transform, so the repeats "
            "are not five copies of one picture. The test set skips this stack.",
            "",
            clean_sentence,
            "",
            where,
        ]
    )


def _described_run(rows: list[dict]) -> dict | None:
    for row in rows:
        run_config = row.get("config") or {}
        if run_config.get("augmentation"):
            return run_config
    return None


def _factors_from(config: dict, rows: list[dict]) -> dict:
    """Prefer a finished run's own factors so the report matches the scored model.

    A missing ``clean_probability`` stays missing. It is not filled from the
    YAML, because an older run warped every view and the report must say so.
    """

    for row in rows:
        run_config = row.get("config") or {}
        augmentation = run_config.get("augmentation") or {}
        if augmentation:
            factors = dict(augmentation)
            factors["target_samples_per_class"] = run_config.get(
                "target_samples_per_class", config.get("target_samples_per_class")
            )
            factors["train_per_class"] = run_config.get("train_per_class", config.get("train_per_class"))
            return factors
    factors = dict(config.get("augmentation") or {})
    factors["target_samples_per_class"] = config.get("target_samples_per_class")
    factors["train_per_class"] = config.get("train_per_class")
    return factors


def _class_note(class_counts: list[int]) -> str:
    if not class_counts:
        return (
            "The brief states 13,065 characters and a softmax of 13,065 neurons. "
            "The accuracy table below is filled after a run stores `metrics.json`."
        )
    if class_counts == [13065]:
        return "The runs in the table below use the full set of 13,065 characters."
    shown = ", ".join(str(count) for count in class_counts)
    return (
        f"The runs summarised below use {shown} character classes. "
        "The brief also states a total of 13,065 characters and 13,065 output neurons. "
        "A result on the smaller common-character release is not a substitute for that full set."
    )


def _epochs_trained(row: dict) -> str:
    """Prefer the CSV row count. A resumed run stores only the new epochs in metrics."""

    path = row.get("path")
    if path is not None:
        log = Path(path) / "training_log.csv"
        if log.exists():
            rows = [line for line in log.read_text(encoding="utf-8").splitlines() if line.strip()]
            if len(rows) > 1:
                return str(len(rows) - 1)
    metrics = row.get("metrics") or {}
    return _cell(metrics.get("epochs_trained"))


def _learning_rate(row: dict) -> str:
    run_config = row.get("config") or {}
    value = run_config.get("learning_rate")
    if value is None:
        return "the configured"
    return str(value)


def _latest_run(output_dir: Path, model: str) -> Path | None:
    named = []
    for prefix in ("rtx4060_", "rtx5070ti_"):
        path = output_dir / model / f"{prefix}{model}"
        if (path / "metrics.json").exists():
            named.append(path)
    chosen = _prefer_metrics(named)
    if chosen is not None:
        return chosen
    root = output_dir / model
    if not root.is_dir():
        return None
    found = [path for path in root.iterdir() if path.is_dir() and (path / "metrics.json").exists()]
    return _prefer_metrics(found)


def _prefer_metrics(paths: list[Path]) -> Path | None:
    if not paths:
        return None
    full = [path for path in paths if _class_count(path) == 13065]
    pool = full or list(paths)
    return max(pool, key=lambda path: (path / "metrics.json").stat().st_mtime)


def _class_count(path: Path) -> int | None:
    loaded = _read_json(path / "metrics.json")
    if not loaded:
        return None
    value = loaded.get("num_classes")
    return value if isinstance(value, int) else None


def _read_json(path: Path | None) -> dict | None:
    if path is None or not path.exists():
        return None
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else None


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return loaded if isinstance(loaded, dict) else {}


def _percent(value) -> str:
    if value is None:
        return "—"
    return f"{float(value) * 100:.2f}%"


def _cell(value) -> str:
    if value is None or value == "":
        return "—"
    return str(value)


def _infer_config_path(rows: list[dict]) -> str | None:
    for row in rows:
        path = row.get("path")
        if path is None:
            continue
        folder = path.name
        if folder.startswith("rtx4060_"):
            return "configs/rtx4060.yaml"
        if folder.startswith("rtx5070ti_"):
            return "configs/rtx5070ti.yaml"
    return None


def _infer_data_dir(rows: list[dict]) -> str | None:
    for row in rows:
        data_dir = (row.get("config") or {}).get("data_dir")
        if data_dir:
            return str(data_dir)
    return None


def _shell_quote(value: str) -> str:
    if value and all(char.isalnum() or char in "/._-" for char in value):
        return value
    return "'" + value.replace("'", "'\"'\"'") + "'"


def _display_path(path: Path | None) -> str:
    if path is None:
        return ""
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


if __name__ == "__main__":
    sys.exit(main())
