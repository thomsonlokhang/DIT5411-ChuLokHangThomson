"""Check the split rule and run one tiny training job on synthetic characters.

This does not download the real dataset. It writes a few dozen images under
artifacts/smoke_data, which is gitignored, then trains custom_cnn for one epoch.

    python scripts/smoke_test.py
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import tensorflow as tf

from src.data_pipeline import (
    build_augmenter,
    build_split_index,
    make_datasets,
    sample_sort_key,
    shear_images,
)
from src.evaluate import main as evaluate_main
from src.models import MODEL_NAMES, build_model
from src.train import main as train_main


def main() -> None:
    os.chdir(ROOT)
    test_numeric_order()
    data_dir = write_synthetic_dataset(ROOT / "artifacts" / "smoke_data")
    test_official_split(data_dir)
    test_flat_split(ROOT / "artifacts" / "smoke_flat")
    test_short_class(ROOT / "artifacts" / "smoke_short")
    test_shear_identity()
    test_models_build()
    test_augmentation_batch(data_dir)
    run_dir = train_once(data_dir)
    evaluate_main(
        [
            "--run-dir",
            str(run_dir),
            "--data-dir",
            str(data_dir),
            "--report-dir",
            str(ROOT / "artifacts" / "smoke_reports"),
        ]
    )
    print("Smoke test passed.")


def test_numeric_order() -> None:
    paths = [Path("甲_10.png"), Path("甲_2.png"), Path("甲_1.png")]
    ordered = [path.name for path in sorted(paths, key=sample_sort_key)]
    assert ordered == ["甲_1.png", "甲_2.png", "甲_10.png"], ordered


def write_synthetic_dataset(root: Path) -> Path:
    if root.exists():
        shutil.rmtree(root)
    specs = {
        "甲": (45, "horizontal"),
        "乙": (45, "vertical"),
        "丙": (45, "diagonal"),
    }
    for label, (count, kind) in specs.items():
        for number in range(1, count + 1):
            image = render_glyph(kind, number)
            path = root / label / f"{label}_{number}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(tf.io.encode_png(image).numpy())
    return root


def render_glyph(kind: str, number: int) -> np.ndarray:
    image = np.full((32, 32, 1), 255, dtype=np.uint8)
    shift = (number * 3) % 7
    if kind == "horizontal":
        row = 8 + shift
        image[row : row + 3, 4:28] = 0
    elif kind == "vertical":
        col = 8 + shift
        image[4:28, col : col + 3] = 0
    else:
        for offset in range(3):
            index = np.arange(4, 28)
            image[index, np.clip(index + shift - 4 + offset, 0, 31)] = 0
    return image


def test_official_split(data_dir: Path) -> None:
    index = build_split_index(data_dir, train_per_class=40, target_samples_per_class=200)
    assert index.num_classes == 3
    assert index.repeats == 5
    assert index.class_names == sorted(index.class_names)
    for name in index.class_names:
        numbers = _numbers_for(index.train_paths, name)
        assert numbers == list(range(1, 41)), (name, numbers[:5], numbers[-5:])
        test_numbers = _numbers_for(index.test_paths, name)
        assert test_numbers == [41, 42, 43, 44, 45], test_numbers
    # Writing files out of order must not change the cut.
    shuffled = data_dir / "甲"
    files = list(shuffled.iterdir())
    assert files
    print(f"Split check passed for {index.num_classes} classes, repeat x{index.repeats}.")


def test_flat_split(root: Path) -> None:
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    for number in (1, 2, 10, 11):
        image = render_glyph("horizontal", number)
        path = root / f"丁_{number}.png"
        path.write_bytes(tf.io.encode_png(image).numpy())
    index = build_split_index(root, train_per_class=3, target_samples_per_class=3)
    assert index.class_names == ["丁"]
    assert _numbers_for(index.train_paths, "丁") == [1, 2, 10]
    assert _numbers_for(index.test_paths, "丁") == [11]


def test_short_class(root: Path) -> None:
    if root.exists():
        shutil.rmtree(root)
    folder = root / "戊"
    folder.mkdir(parents=True)
    for number in range(1, 6):
        image = render_glyph("vertical", number)
        (folder / f"戊_{number}.png").write_bytes(tf.io.encode_png(image).numpy())
    index = build_split_index(root, train_per_class=40, target_samples_per_class=200)
    assert len(index.train_paths) == 5
    assert index.test_paths == []
    assert index.warnings


def test_shear_identity() -> None:
    image = np.zeros((1, 16, 16, 1), dtype=np.float32)
    image[:, 4:12, 4:12] = 1.0
    result = shear_images(tf.constant(image), tf.constant([0.0]), tf.constant([0.0]), fill_value=0.0)
    np.testing.assert_allclose(result.numpy(), image, atol=1e-5)


def test_models_build() -> None:
    for name in MODEL_NAMES:
        sample = tf.zeros((2, 32, 32, 1))
        model = build_model(name, (32, 32, 1), num_classes=4, weights=None, freeze_base=True)
        output = model(sample, training=False)
        assert tuple(output.shape) == (2, 4), (name, output.shape)
        print(f"Built {name} ({model.count_params():,} parameters)")
        if name == "mobilenet_v2":
            path = ROOT / "artifacts" / "smoke_mobilenet.keras"
            path.parent.mkdir(parents=True, exist_ok=True)
            model.save(str(path))
            loaded = tf.keras.models.load_model(str(path))
            reloaded = loaded(sample[:1], training=False)
            assert tuple(reloaded.shape) == (1, 4)
            path.unlink()
        tf.keras.backend.clear_session()


def test_augmentation_batch(data_dir: Path) -> None:
    index = build_split_index(data_dir, train_per_class=40, target_samples_per_class=40)
    train_ds, _val, _test = make_datasets(
        index,
        image_size=32,
        batch_size=8,
        augmentation={"rotation_degrees": 20, "translation": 0.1, "zoom": 0.2, "scale": 0.1, "shear_x": 0.3, "shear_y": 0.1},
        which=("train",),
        repeat_factor=1,
    )
    preview = build_augmenter(32, {})
    print("Augmentation layers:", ", ".join(f"{layer.name}:{layer.__class__.__name__}" for layer in preview.layers))
    images, labels = next(iter(train_ds))
    assert images.shape[1:] == (32, 32, 1)
    assert float(tf.reduce_min(images)) >= -1e-5
    assert float(tf.reduce_max(images)) <= 1.0 + 1e-5
    assert int(tf.reduce_max(labels)) < index.num_classes


def train_once(data_dir: Path) -> Path:
    output = ROOT / "artifacts" / "smoke_runs"
    if output.exists():
        shutil.rmtree(output)
    train_main(
        [
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(output),
            "--model",
            "custom_cnn",
            "--epochs",
            "1",
            "--image-size",
            "32",
            "--batch-size",
            "16",
            "--target-samples-per-class",
            "40",
            "--learning-rate",
            "0.001",
            "--patience",
            "1",
            "--run-name",
            "smoke",
            "--weights",
            "none",
        ]
    )
    run_dir = output / "custom_cnn" / "smoke"
    assert (run_dir / "best.keras").exists()
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "training_curves.png").exists()
    return run_dir


def _numbers_for(paths: list[str], label: str) -> list[int]:
    numbers = []
    for path in paths:
        file_path = Path(path)
        stem_label = file_path.stem.rsplit("_", 1)[0]
        if file_path.parent.name != label and stem_label != label:
            continue
        numbers.append(int(file_path.stem.rsplit("_", 1)[1]))
    return numbers


if __name__ == "__main__":
    main()
