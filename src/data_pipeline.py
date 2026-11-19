"""Load the Tegaki Traditional Chinese handwriting set and build tf.data pipelines.

The official split is positional. For every character the files are ordered by
the sample number in the filename, then the first ``train_per_class`` images
(default 40) become the training pool and the remainder become the test set.
That list is never shuffled before the cut. Shuffling happens later, and only
inside the training pipeline, so the membership of the test set stays fixed.

Augmentation is applied on the fly after the split. The training set is repeated
until each class with a full training pool contributes at least
``target_samples_per_class`` views per epoch (default 200). Images are not
written to disk.
"""

from __future__ import annotations

import logging
import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import tensorflow as tf

logger = logging.getLogger("dit5411.data")

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff", ".webp"}
_SAMPLE_NUMBER = re.compile(r"(\d+)$")
_FLAT_NAME = re.compile(r"^(.*)_(\d+)$")


@dataclass
class SplitIndex:
    """Paths and integer labels after the official first-N cut."""

    class_names: list[str]
    train_paths: list[str]
    train_labels: list[int]
    val_paths: list[str]
    val_labels: list[int]
    test_paths: list[str]
    test_labels: list[int]
    train_per_class: int
    val_ratio: float
    repeats: int
    train_counts: dict[str, int]
    test_counts: dict[str, int]
    warnings: list[str] = field(default_factory=list)

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    def summary(self) -> str:
        short_warnings = self.warnings[:8]
        extra = len(self.warnings) - len(short_warnings)
        warning_text = ""
        if short_warnings:
            warning_text = "\n".join(f"  - {line}" for line in short_warnings)
            if extra > 0:
                warning_text += f"\n  - ... {extra} more"
            warning_text = "\nWarnings:\n" + warning_text
        return (
            f"Classes: {self.num_classes}\n"
            f"Train images: {len(self.train_paths)} | "
            f"val images: {len(self.val_paths)} | "
            f"test images: {len(self.test_paths)}\n"
            f"Augmented views per epoch: {len(self.train_paths) * self.repeats} "
            f"(repeat x{self.repeats})"
            f"{warning_text}"
        )


def sample_sort_key(path: Path) -> tuple[int, str]:
    """Order samples by the trailing number in the filename, then by name.

    ``char_2.png`` sorts before ``char_10.png``. Filesystem listing order is
    not used, because it changes between Windows and Linux.
    """

    match = _SAMPLE_NUMBER.search(path.stem)
    number = int(match.group(1)) if match else 10**18
    return (number, path.name)


def class_name_from_file(path: Path) -> str:
    """Recover the character from a flat filename such as ``永_12.png``."""

    match = _FLAT_NAME.match(path.stem)
    if match and match.group(1):
        return match.group(1)
    return path.stem


def _is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def _images_in(directory: Path) -> list[Path]:
    return [path for path in directory.iterdir() if _is_image(path)]


def resolve_data_dir(data_dir: str | Path) -> Path:
    """Accept either the character root or a parent that contains ``cleaned_data``."""

    root = Path(data_dir).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(
            f"Dataset directory does not exist: {root}\n"
            "Download and extract the Traditional Chinese Handwriting Dataset, "
            "then pass its cleaned_data folder with --data-dir. See README.md."
        )
    if not root.is_dir():
        raise NotADirectoryError(f"Dataset path is not a directory: {root}")

    named = ("cleaned_data", "cleaned_data(50_50)")
    has_images = any(_is_image(path) for path in root.iterdir())
    has_class_dirs = any(
        path.is_dir() and _images_in(path) for path in root.iterdir() if not path.name.startswith(".")
    )
    if has_images or has_class_dirs:
        return root
    for name in named:
        candidate = root / name
        if candidate.is_dir():
            return candidate
    return root


def discover_classes(data_dir: str | Path) -> dict[str, list[Path]]:
    """Map each character to its image paths.

    Two layouts are accepted:

    * ``root/<character>/<character>_<n>.png`` (full 13,065-class release)
    * ``root/<character>_<n>.png`` (flat files before they are moved into folders)
    """

    root = resolve_data_dir(data_dir)
    grouped: dict[str, list[Path]] = {}
    class_dirs = [
        path
        for path in root.iterdir()
        if path.is_dir() and not path.name.startswith(".") and path.name != "__MACOSX"
    ]
    used_folders = False
    for directory in class_dirs:
        images = _images_in(directory)
        if not images:
            continue
        used_folders = True
        grouped[directory.name] = images

    if used_folders:
        return grouped

    for image in _images_in(root):
        grouped.setdefault(class_name_from_file(image), []).append(image)
    return grouped


def _read_class_list(path: str | Path) -> list[str]:
    names: list[str] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        name = line.strip()
        if name and not name.startswith("#"):
            names.append(name)
    if not names:
        raise ValueError(f"No class names found in {path}")
    return names


def build_split_index(
    data_dir: str | Path,
    train_per_class: int = 40,
    val_ratio: float = 0.0,
    max_classes: int | None = None,
    classes_file: str | Path | None = None,
    class_names: list[str] | None = None,
    target_samples_per_class: int = 200,
) -> SplitIndex:
    """Build the official split without shuffling samples first.

    ``class_names``, when supplied, is the exact label order from a training
    run. Evaluation uses it so class 0 still means the same character.
    """

    if train_per_class < 1:
        raise ValueError("train_per_class must be at least 1")
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("val_ratio must be in [0, 1)")

    discovered = discover_classes(data_dir)
    if not discovered:
        root = resolve_data_dir(data_dir)
        raise FileNotFoundError(
            f"No character images found under {root}. "
            "Expected one folder per character, or flat files named <character>_<number>.png."
        )

    warnings: list[str] = []
    if class_names is None:
        if classes_file:
            requested = _read_class_list(classes_file)
            missing = [name for name in requested if name not in discovered]
            if missing:
                raise FileNotFoundError(
                    f"{len(missing)} names in {classes_file} are not in the dataset. "
                    f"Examples: {missing[:5]}"
                )
            class_names = requested
        else:
            class_names = sorted(discovered)
        if max_classes is not None:
            if max_classes < 1:
                raise ValueError("max_classes must be positive")
            class_names = class_names[:max_classes]
    else:
        missing = [name for name in class_names if name not in discovered]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} classes from this training run are missing on disk. "
                f"Examples: {missing[:5]}"
            )

    train_paths: list[str] = []
    train_labels: list[int] = []
    val_paths: list[str] = []
    val_labels: list[int] = []
    test_paths: list[str] = []
    test_labels: list[int] = []
    train_counts: dict[str, int] = {}
    test_counts: dict[str, int] = {}

    for label, name in enumerate(class_names):
        ordered = sorted(discovered[name], key=sample_sort_key)
        pool = ordered[:train_per_class]
        test = ordered[train_per_class:]
        if len(ordered) < train_per_class:
            warnings.append(
                f"{name}: only {len(ordered)} samples, so all of them are training "
                f"and this class has no test images"
            )
        elif len(test) == 0:
            warnings.append(f"{name}: exactly {train_per_class} samples, so the test split is empty")

        val_count = 0
        if val_ratio > 0 and len(pool) > 1:
            val_count = min(len(pool) - 1, int(len(pool) * val_ratio))
        train = pool[:-val_count] if val_count else pool
        validation = pool[-val_count:] if val_count else []

        train_paths.extend(str(path) for path in train)
        train_labels.extend([label] * len(train))
        val_paths.extend(str(path) for path in validation)
        val_labels.extend([label] * len(validation))
        test_paths.extend(str(path) for path in test)
        test_labels.extend([label] * len(test))
        train_counts[name] = len(train)
        test_counts[name] = len(test)

    if not train_paths:
        raise RuntimeError("The training split is empty.")

    repeats = max(1, math.ceil(target_samples_per_class / train_per_class))
    return SplitIndex(
        class_names=list(class_names),
        train_paths=train_paths,
        train_labels=train_labels,
        val_paths=val_paths,
        val_labels=val_labels,
        test_paths=test_paths,
        test_labels=test_labels,
        train_per_class=train_per_class,
        val_ratio=val_ratio,
        repeats=repeats,
        train_counts=train_counts,
        test_counts=test_counts,
        warnings=warnings,
    )


def shear_projective_transforms(shx: tf.Tensor, shy: tf.Tensor, height: tf.Tensor, width: tf.Tensor) -> tf.Tensor:
    """Eight coefficients that invert a centre shear, matching ``shear_images``.

    ``ImageProjectiveTransformV3`` maps each output pixel back to an input
    pixel. The coefficients are that inverse, so a factor of zero is identity.
    """

    shx = tf.cast(shx, tf.float32)
    shy = tf.cast(shy, tf.float32)
    cx = (tf.cast(width, tf.float32) - 1.0) / 2.0
    cy = (tf.cast(height, tf.float32) - 1.0) / 2.0
    det = 1.0 - shx * shy
    det = tf.where(tf.abs(det) < 1e-4, tf.ones_like(det), det)
    a0 = 1.0 / det
    a1 = -shx / det
    a2 = cx - (cx - shx * cy) / det
    b0 = -shy / det
    b1 = 1.0 / det
    b2 = cy - (cy - shy * cx) / det
    zeros = tf.zeros_like(a0)
    return tf.stack([a0, a1, a2, b0, b1, b2, zeros, zeros], axis=-1)


def shear_images(images: tf.Tensor, shx: tf.Tensor, shy: tf.Tensor, fill_value: float) -> tf.Tensor:
    """Shear a batch around the image centre and fill empty pixels.

    ``shx`` and ``shy`` are per-image shear factors. Zero shear returns the
    input pixels. This is the portable path used when ``RandomShear`` is absent.
    """

    images = tf.convert_to_tensor(images, dtype=tf.float32)
    shape = tf.shape(images)
    batch = shape[0]
    height = shape[1]
    width = shape[2]
    xs = tf.cast(tf.range(width), tf.float32)
    ys = tf.cast(tf.range(height), tf.float32)
    x_out, y_out = tf.meshgrid(xs, ys)
    x_out = tf.broadcast_to(x_out, [batch, height, width])
    y_out = tf.broadcast_to(y_out, [batch, height, width])
    cx = (tf.cast(width, tf.float32) - 1.0) / 2.0
    cy = (tf.cast(height, tf.float32) - 1.0) / 2.0
    shx = tf.reshape(tf.cast(shx, tf.float32), [batch, 1, 1])
    shy = tf.reshape(tf.cast(shy, tf.float32), [batch, 1, 1])
    det = 1.0 - shx * shy
    det = tf.where(tf.abs(det) < 1e-4, tf.ones_like(det), det)
    x_in = cx + ((x_out - cx) - shx * (y_out - cy)) / det
    y_in = cy + ((y_out - cy) - shy * (x_out - cx)) / det
    return _bilinear_sample(images, x_in, y_in, fill_value)


def _sample_affine(images: tf.Tensor, coeff: tf.Tensor, fill_value: float) -> tf.Tensor:
    """Resample a batch with one output-to-input affine map.

    Gather and arithmetic stay inside XLA. ``ImageProjectiveTransformV3`` has
    no XLA GPU kernel, so a compiled training step cannot call it.
    """

    shape = tf.shape(images)
    batch = shape[0]
    height = shape[1]
    width = shape[2]
    xs = tf.cast(tf.range(width), tf.float32)
    ys = tf.cast(tf.range(height), tf.float32)
    x_out, y_out = tf.meshgrid(xs, ys)
    x_out = tf.broadcast_to(x_out, [batch, height, width])
    y_out = tf.broadcast_to(y_out, [batch, height, width])
    a0, a1, a2, b0, b1, b2, c0, c1 = tf.unstack(tf.cast(coeff, tf.float32), axis=-1)

    def column(value: tf.Tensor) -> tf.Tensor:
        return value[:, None, None]

    denom = column(c0) * x_out + column(c1) * y_out + 1.0
    x_in = (column(a0) * x_out + column(a1) * y_out + column(a2)) / denom
    y_in = (column(b0) * x_out + column(b1) * y_out + column(b2)) / denom
    return _bilinear_sample(images, x_in, y_in, fill_value)


def _bilinear_sample(
    images: tf.Tensor,
    x_coords: tf.Tensor,
    y_coords: tf.Tensor,
    fill_value: float,
) -> tf.Tensor:
    shape = tf.shape(images)
    batch = shape[0]
    height = shape[1]
    width = shape[2]
    x0 = tf.floor(x_coords)
    y0 = tf.floor(y_coords)
    x1 = x0 + 1.0
    y1 = y0 + 1.0
    wa = (x1 - x_coords) * (y1 - y_coords)
    wb = (x_coords - x0) * (y1 - y_coords)
    wc = (x1 - x_coords) * (y_coords - y0)
    wd = (x_coords - x0) * (y_coords - y0)

    def gather(ix: tf.Tensor, iy: tf.Tensor) -> tf.Tensor:
        valid = (
            (ix >= 0)
            & (iy >= 0)
            & (ix < tf.cast(width, tf.float32))
            & (iy < tf.cast(height, tf.float32))
        )
        ix_c = tf.clip_by_value(tf.cast(ix, tf.int32), 0, width - 1)
        iy_c = tf.clip_by_value(tf.cast(iy, tf.int32), 0, height - 1)
        batch_index = tf.tile(tf.range(batch)[:, None, None], [1, height, width])
        index = tf.stack([batch_index, iy_c, ix_c], axis=-1)
        sampled = tf.gather_nd(images, index)
        fill = tf.ones_like(sampled) * tf.cast(fill_value, images.dtype)
        return tf.where(valid[..., None], sampled, fill)

    return (
        gather(x0, y0) * wa[..., None]
        + gather(x1, y0) * wb[..., None]
        + gather(x0, y1) * wc[..., None]
        + gather(x1, y1) * wd[..., None]
    )


@tf.keras.utils.register_keras_serializable(package="dit5411")
class FallbackRandomShear(tf.keras.layers.Layer):
    """Random shear that stays active only while ``training=True``."""

    def __init__(self, x_factor: float = 0.2, y_factor: float = 0.0, fill_value: float = 1.0, **kwargs):
        # Stay in float32 under mixed_float16. The shear op rejects a float16 fill.
        kwargs["dtype"] = "float32"
        super().__init__(**kwargs)
        self.x_factor = float(x_factor)
        self.y_factor = float(y_factor)
        self.fill_value = float(fill_value)

    def call(self, images, training=None):
        if training is False:
            return images
        # Both cond branches are traced on the same tensor, so a dynamic rank
        # check cannot add a batch axis. Model and dataset inputs are NHWC.
        if images.shape.rank == 3:
            result = self._shear_batch(images[tf.newaxis, ...])[0]
        else:
            result = self._shear_batch(images)
        result.set_shape(images.shape)
        return result

    def compute_output_shape(self, input_shape):
        return tuple(input_shape)

    def compute_output_spec(self, inputs, training=None):
        return tf.keras.KerasTensor(shape=inputs.shape, dtype=self.compute_dtype)

    def _shear_batch(self, batched):
        batch = tf.shape(batched)[0]
        shx = tf.random.uniform([batch], -self.x_factor, self.x_factor)
        shy = tf.random.uniform([batch], -self.y_factor, self.y_factor)
        shape = tf.shape(batched)
        # ImageProjectiveTransformV3 requires a float32 fill even when the
        # activations arriving from mixed_float16 are float16.
        pixels = tf.cast(batched, tf.float32)
        sheared = tf.raw_ops.ImageProjectiveTransformV3(
            images=pixels,
            transforms=shear_projective_transforms(shx, shy, shape[1], shape[2]),
            output_shape=tf.cast(shape[1:3], tf.int32),
            fill_value=tf.constant(self.fill_value, dtype=tf.float32),
            interpolation="BILINEAR",
            fill_mode="CONSTANT",
        )
        return tf.cast(sheared, batched.dtype)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "x_factor": self.x_factor,
                "y_factor": self.y_factor,
                "fill_value": self.fill_value,
            }
        )
        return config


def _layer_kwargs(layer_cls, preferred: dict) -> dict:
    try:
        import inspect

        signature = inspect.signature(layer_cls.__init__)
    except (TypeError, ValueError):
        return preferred
    if any(param.kind == param.VAR_KEYWORD for param in signature.parameters.values()):
        return preferred
    return {key: value for key, value in preferred.items() if key in signature.parameters}


def build_augmenter(image_size: int, augmentation: dict | None = None) -> tf.keras.Sequential:
    """Stack rotation, translation, zoom, anisotropic scale, and shear.

    Empty pixels are filled with 1.0. Handwriting ink is dark on a white page,
    and the default Keras fill of 0 would paint black borders into the glyph.
    """

    options = {
        "rotation_degrees": 15,
        "translation": 0.10,
        "zoom": 0.15,
        "scale": 0.10,
        "shear_x": 0.25,
        "shear_y": 0.15,
    }
    if augmentation:
        options.update(augmentation)

    fill = {"fill_mode": "constant", "fill_value": 1.0}
    rotation = options["rotation_degrees"] / 360.0
    layers = [
        tf.keras.layers.RandomRotation(
            factor=rotation,
            **_layer_kwargs(tf.keras.layers.RandomRotation, fill),
            name="aug_rotation",
        ),
        tf.keras.layers.RandomTranslation(
            height_factor=options["translation"],
            width_factor=options["translation"],
            **_layer_kwargs(tf.keras.layers.RandomTranslation, fill),
            name="aug_translation",
        ),
        tf.keras.layers.RandomZoom(
            height_factor=options["zoom"],
            width_factor=options["zoom"],
            **_layer_kwargs(tf.keras.layers.RandomZoom, fill),
            name="aug_zoom",
        ),
        tf.keras.layers.RandomZoom(
            height_factor=options["scale"],
            width_factor=options["scale"] * 1.5,
            **_layer_kwargs(tf.keras.layers.RandomZoom, fill),
            name="aug_scale",
        ),
    ]
    # Keras RandomShear accepts a magnitude in [0, 1] and then saves the signed
    # interval [-factor, factor]. Loading that config raises ValueError, so a
    # model that contains it cannot be reloaded for evaluation. The project
    # layer stores the magnitude and applies the same centre shear.
    layers.append(
        FallbackRandomShear(
            x_factor=options["shear_x"],
            y_factor=options["shear_y"],
            fill_value=1.0,
            name="aug_shear",
        )
    )
    return tf.keras.Sequential(layers, name="train_augmentation")


@tf.keras.utils.register_keras_serializable(package="dit5411")
class KeepCleanFraction(tf.keras.layers.Layer):
    """Leave a fraction of each training batch unwarped.

    The test glyphs are upright. A model that only sees rotated and sheared
    ink learns that appearance and then misses the clean test images.
    """

    def __init__(self, clean_probability: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.clean_probability = float(clean_probability)

    def call(self, inputs, training=None):
        original, transformed = inputs
        transformed = tf.clip_by_value(transformed, 0.0, 1.0)
        if training is False or training is None:
            return original
        if self.clean_probability <= 0:
            mixed = transformed
        else:
            keep = tf.random.uniform([tf.shape(original)[0], 1, 1, 1]) < self.clean_probability
            mixed = tf.where(keep, original, transformed)
        if training is True:
            mixed.set_shape(original.shape)
            return mixed
        flag = tf.cast(tf.reshape(training, []), mixed.dtype)
        result = flag * mixed + (1.0 - flag) * original
        result.set_shape(original.shape)
        return result

    def get_config(self):
        config = super().get_config()
        config.update({"clean_probability": self.clean_probability})
        return config


def _coeff_matrix(coeff: tf.Tensor) -> tf.Tensor:
    """Turn the 8 projective coefficients into a batch of 3x3 maps."""

    a0, a1, a2, b0, b1, b2, c0, c1 = tf.unstack(tf.cast(coeff, tf.float32), axis=-1)
    ones = tf.ones_like(a0)
    row0 = tf.stack([a0, a1, a2], axis=-1)
    row1 = tf.stack([b0, b1, b2], axis=-1)
    row2 = tf.stack([c0, c1, ones], axis=-1)
    return tf.stack([row0, row1, row2], axis=-2)


def _matrix_coeff(matrix: tf.Tensor) -> tf.Tensor:
    return tf.stack(
        [
            matrix[:, 0, 0],
            matrix[:, 0, 1],
            matrix[:, 0, 2],
            matrix[:, 1, 0],
            matrix[:, 1, 1],
            matrix[:, 1, 2],
            matrix[:, 2, 0],
            matrix[:, 2, 1],
        ],
        axis=-1,
    )


def _compose_coeffs(*coeffs: tf.Tensor) -> tf.Tensor:
    """Combine output-to-input maps. The first map is the earliest image warp."""

    combined = _coeff_matrix(coeffs[0])
    for coeff in coeffs[1:]:
        combined = tf.matmul(combined, _coeff_matrix(coeff))
    return _matrix_coeff(combined)


def _rotation_coeff(angle_deg: tf.Tensor, height: tf.Tensor, width: tf.Tensor) -> tf.Tensor:
    angle = tf.cast(angle_deg, tf.float32) * (math.pi / 180.0)
    cos_theta = tf.math.cos(angle)
    sin_theta = tf.math.sin(angle)
    cx = 0.5 * (tf.cast(width, tf.float32) - 1.0)
    cy = 0.5 * (tf.cast(height, tf.float32) - 1.0)
    a0 = cos_theta
    a1 = -sin_theta
    b0 = sin_theta
    b1 = cos_theta
    a2 = cx - cx * a0 - cy * a1
    b2 = cy - cx * b0 - cy * b1
    zeros = tf.zeros_like(a0)
    return tf.stack([a0, a1, a2, b0, b1, b2, zeros, zeros], axis=-1)


def _translation_coeff(dx: tf.Tensor, dy: tf.Tensor) -> tf.Tensor:
    zeros = tf.zeros_like(dx)
    ones = tf.ones_like(dx)
    return tf.stack([ones, zeros, -dx, zeros, ones, -dy, zeros, zeros], axis=-1)


def _zoom_coeff(zx: tf.Tensor, zy: tf.Tensor, height: tf.Tensor, width: tf.Tensor) -> tf.Tensor:
    x_offset = ((tf.cast(width, tf.float32) - 1.0) / 2.0) * (1.0 - zx)
    y_offset = ((tf.cast(height, tf.float32) - 1.0) / 2.0) * (1.0 - zy)
    zeros = tf.zeros_like(zx)
    return tf.stack([zx, zeros, x_offset, zeros, zy, y_offset, zeros, zeros], axis=-1)


@tf.keras.utils.register_keras_serializable(package="dit5411")
class ComposedAffineAugment(tf.keras.layers.Layer):
    """One rotation, shift, zoom, scale, and shear, applied as a single resample.

    Five separate warps made each 4060 step wait on image resampling. The same
    factors are composed into one map, then sampled once. ``training=False``
    returns the pixels unchanged, so the test set stays clean.
    """

    def __init__(
        self,
        rotation_degrees: float = 15.0,
        translation: float = 0.10,
        zoom: float = 0.15,
        scale: float = 0.10,
        shear_x: float = 0.25,
        shear_y: float = 0.15,
        fill_value: float = 1.0,
        **kwargs,
    ):
        kwargs["dtype"] = "float32"
        super().__init__(**kwargs)
        self.rotation_degrees = float(rotation_degrees)
        self.translation = float(translation)
        self.zoom = float(zoom)
        self.scale = float(scale)
        self.shear_x = float(shear_x)
        self.shear_y = float(shear_y)
        self.fill_value = float(fill_value)

    def call(self, images, training=None):
        if training is False or training is None:
            return images
        warped = self._warp(images)
        if training is True:
            return warped
        flag = tf.cast(tf.reshape(training, []), warped.dtype)
        original = tf.cast(images, warped.dtype)
        return flag * warped + (1.0 - flag) * original

    def _warp(self, images):
        pixels = images
        squeezed = pixels.shape.rank == 3
        if squeezed:
            pixels = pixels[tf.newaxis, ...]
        pixels = tf.cast(pixels, tf.float32)
        shape = tf.shape(pixels)
        batch = shape[0]
        height = shape[1]
        width = shape[2]
        height_f = tf.cast(height, tf.float32)
        width_f = tf.cast(width, tf.float32)

        angle = tf.random.uniform([batch], -self.rotation_degrees, self.rotation_degrees)
        dx = tf.random.uniform([batch], -self.translation, self.translation) * width_f
        dy = tf.random.uniform([batch], -self.translation, self.translation) * height_f
        zoom_x = tf.random.uniform([batch], 1.0 - self.zoom, 1.0 + self.zoom)
        zoom_y = tf.random.uniform([batch], 1.0 - self.zoom, 1.0 + self.zoom)
        scale_x = tf.random.uniform([batch], 1.0 - self.scale * 1.5, 1.0 + self.scale * 1.5)
        scale_y = tf.random.uniform([batch], 1.0 - self.scale, 1.0 + self.scale)
        shx = tf.random.uniform([batch], -self.shear_x, self.shear_x)
        shy = tf.random.uniform([batch], -self.shear_y, self.shear_y)

        transform = _compose_coeffs(
            _rotation_coeff(angle, height, width),
            _translation_coeff(dx, dy),
            _zoom_coeff(zoom_x, zoom_y, height, width),
            _zoom_coeff(scale_x, scale_y, height, width),
            shear_projective_transforms(shx, shy, height, width),
        )
        warped = _sample_affine(pixels, transform, self.fill_value)
        warped = tf.cast(warped, images.dtype)
        if squeezed:
            warped = warped[0]
        warped.set_shape(images.shape)
        return warped

    def compute_output_shape(self, input_shape):
        return tuple(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "rotation_degrees": self.rotation_degrees,
                "translation": self.translation,
                "zoom": self.zoom,
                "scale": self.scale,
                "shear_x": self.shear_x,
                "shear_y": self.shear_y,
                "fill_value": self.fill_value,
            }
        )
        return config


def attach_gpu_augmentation(model, image_size: int, augmentation: dict | None = None):
    """Run one composed training warp on the GPU, inside the compiled train step.

    The warp is an identity when ``training=False``, so evaluation of the saved
    model does not change the test set. ``model`` itself is unchanged and is
    called after the mix.
    """

    options = dict(augmentation or {})
    clean_probability = float(options.pop("clean_probability", 0.0) or 0.0)
    if not 0.0 <= clean_probability < 1.0:
        raise ValueError("clean_probability must be in [0, 1)")
    inputs = tf.keras.Input(shape=(image_size, image_size, 1), name="image")
    warped = ComposedAffineAugment(
        rotation_degrees=float(options.get("rotation_degrees", 15.0)),
        translation=float(options.get("translation", 0.10)),
        zoom=float(options.get("zoom", 0.15)),
        scale=float(options.get("scale", 0.10)),
        shear_x=float(options.get("shear_x", 0.25)),
        shear_y=float(options.get("shear_y", 0.15)),
        name="aug_composed",
    )(inputs)
    mixed = KeepCleanFraction(clean_probability=clean_probability, name="keep_clean")([inputs, warped])
    outputs = model(mixed)
    return tf.keras.Model(inputs, outputs, name=model.name)


def _load_image(path: tf.Tensor, label: tf.Tensor, image_size: int):
    raw = tf.io.read_file(path)
    image = tf.io.decode_image(raw, channels=1, expand_animations=False)
    image = tf.image.resize(image, [image_size, image_size], method="bilinear")
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.reshape(image, [image_size, image_size, 1])
    return image, tf.cast(label, tf.int32)


def _project_batch(pixels: tf.Tensor, options: dict) -> tf.Tensor:
    """One bilinear resample. The native op has a CPU kernel and no XLA kernel."""

    with tf.device("/CPU:0"):
        shape = tf.shape(pixels)
        batch = shape[0]
        height = shape[1]
        width = shape[2]
        height_f = tf.cast(height, tf.float32)
        width_f = tf.cast(width, tf.float32)
        rotation = float(options.get("rotation_degrees", 15.0))
        translation = float(options.get("translation", 0.10))
        zoom = float(options.get("zoom", 0.15))
        scale = float(options.get("scale", 0.10))
        shear_x = float(options.get("shear_x", 0.25))
        shear_y = float(options.get("shear_y", 0.15))
        angle = tf.random.uniform([batch], -rotation, rotation)
        dx = tf.random.uniform([batch], -translation, translation) * width_f
        dy = tf.random.uniform([batch], -translation, translation) * height_f
        zoom_x = tf.random.uniform([batch], 1.0 - zoom, 1.0 + zoom)
        zoom_y = tf.random.uniform([batch], 1.0 - zoom, 1.0 + zoom)
        scale_x = tf.random.uniform([batch], 1.0 - scale * 1.5, 1.0 + scale * 1.5)
        scale_y = tf.random.uniform([batch], 1.0 - scale, 1.0 + scale)
        shx = tf.random.uniform([batch], -shear_x, shear_x)
        shy = tf.random.uniform([batch], -shear_y, shear_y)
        transform = _compose_coeffs(
            _rotation_coeff(angle, height, width),
            _translation_coeff(dx, dy),
            _zoom_coeff(zoom_x, zoom_y, height, width),
            _zoom_coeff(scale_x, scale_y, height, width),
            shear_projective_transforms(shx, shy, height, width),
        )
        warped = tf.raw_ops.ImageProjectiveTransformV3(
            images=pixels,
            transforms=transform,
            output_shape=tf.cast(shape[1:3], tf.int32),
            fill_value=tf.constant(1.0, dtype=tf.float32),
            interpolation="BILINEAR",
            fill_mode="CONSTANT",
        )
        warped.set_shape(pixels.shape)
        return warped


def apply_composed_warp(images: tf.Tensor, options: dict, clean_probability: float) -> tf.Tensor:
    """Rotate, shift, zoom, scale, and shear each dirty image in one CPU resample.

    Half of a typical batch stays upright. Those images are not resampled.
    ``ImageProjectiveTransformV3`` stays outside the compiled training step:
    that op has no XLA GPU kernel, and a hand-written gather resample filled
    the 4060 for several minutes per epoch.
    """

    with tf.device("/CPU:0"):
        pixels = tf.cast(images, tf.float32)
        probability = float(clean_probability)
        if probability <= 0.0:
            warped = _project_batch(pixels, options)
        else:
            keep = tf.random.uniform([tf.shape(pixels)[0]]) < probability
            dirty = tf.boolean_mask(pixels, tf.logical_not(keep))
            # The projective op needs a non-empty batch. One extra image covers
            # the rare all-clean batch and is discarded by the slice.
            warped_dirty = _project_batch(tf.concat([dirty, pixels[:1]], axis=0), options)
            warped_dirty = warped_dirty[: tf.shape(dirty)[0]]
            warped = tf.tensor_scatter_nd_update(pixels, tf.where(tf.logical_not(keep)), warped_dirty)
        warped = tf.clip_by_value(warped, 0.0, 1.0)
        warped.set_shape(images.shape)
        return warped


def _cpu_augment_batch(images, labels, image_size: int, options: dict, clean_probability: float):
    transformed = apply_composed_warp(images, options, clean_probability)
    transformed.set_shape([None, image_size, image_size, 1])
    return transformed, labels


def _fill_uint8(dataset, images_np: np.ndarray, labels_np: np.ndarray) -> int:
    offset = 0
    for batch_images, batch_labels in dataset.batch(4096).prefetch(1):
        chunk = batch_images.numpy()
        count = int(chunk.shape[0])
        images_np[offset : offset + count] = np.clip(np.rint(chunk * 255.0), 0, 255).astype(np.uint8)
        labels_np[offset : offset + count] = np.asarray(batch_labels.numpy(), dtype=np.int32)
        offset += count
    return offset


def _load_uint8_store(
    paths: list[str],
    labels: list[int],
    image_size: int,
    cache: str,
    split_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Copy one split into a uint8 array. The float32 disk cache is only read."""

    count = len(paths)
    gib = count * image_size * image_size / (1024**3)
    logger.info(
        "Copying %s %s images into host RAM as uint8 (%.2f GB). This happens once in this process.",
        f"{count:,}",
        split_name,
        gib,
    )
    started = time.perf_counter()
    images_np = np.empty((count, image_size, image_size, 1), dtype=np.uint8)
    labels_np = np.empty((count,), dtype=np.int32)

    def source(use_cache: bool):
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))
        ds = ds.map(
            lambda path, label: _load_image(path, label, image_size),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        if not use_cache or not cache:
            return ds
        if cache == "memory":
            return ds.cache()
        cache_path = Path(f"{cache}_{split_name}")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        return ds.cache(str(cache_path))

    try:
        loaded = _fill_uint8(source(True), images_np, labels_np)
    except (tf.errors.AlreadyExistsError, tf.errors.InvalidArgumentError) as exc:
        logger.warning(
            "The disk cache could not be reread (%s). Decoding the image files once. The cache file is left in place.",
            exc,
        )
        loaded = _fill_uint8(source(False), images_np, labels_np)
    if loaded != count:
        raise RuntimeError(f"Host cache loaded {loaded} {split_name} images, expected {count}.")
    images_np.setflags(write=False)
    logger.info("Host RAM cache is ready in %.1f seconds.", time.perf_counter() - started)
    return images_np, labels_np


def _host_train_dataset(
    images_np: np.ndarray,
    labels_np: np.ndarray,
    image_size: int,
    batch_size: int,
    seed: int,
    repeats: int,
    shuffle_buffer: int,
    drop_remainder: bool,
    aug_config: dict,
    clean_probability: float,
    use_cpu_augment: bool,
):
    """Shuffle image indexes, then warp on the CPU. The uint8 store is not a TF constant.

    A single ``tf.constant`` of this array would cross the 2 GB tensor limit.
    """

    with tf.device("/CPU:0"):
        label_tensor = tf.constant(labels_np, dtype=tf.int32)
    count = int(images_np.shape[0])
    dataset = tf.data.Dataset.range(count)
    dataset = dataset.repeat(repeats)
    buffer = min(max(count * repeats, 1), max(1, shuffle_buffer))
    dataset = dataset.shuffle(buffer, seed=seed, reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)

    def lookup(indices):
        def _take(idx):
            chosen = np.asarray(idx, dtype=np.int64)
            return np.ascontiguousarray(images_np[chosen])

        uint8 = tf.py_function(_take, [indices], tf.uint8)
        uint8.set_shape([None, image_size, image_size, 1])
        with tf.device("/CPU:0"):
            labels = tf.gather(label_tensor, indices)
            pixels = tf.cast(uint8, tf.float32) / 255.0
            if use_cpu_augment:
                pixels = apply_composed_warp(pixels, aug_config, clean_probability)
        pixels.set_shape([None, image_size, image_size, 1])
        return pixels, labels

    # One warp at a time uses the CPU cores. prefetch overlaps that warp with the GPU step.
    return dataset.map(lookup, num_parallel_calls=1).prefetch(2)


def make_datasets(
    index: SplitIndex,
    image_size: int,
    batch_size: int,
    seed: int = 5411,
    augmentation: dict | None = None,
    cache: str = "",
    shuffle_buffer: int = 8192,
    repeat_factor: int | None = None,
    which: tuple[str, ...] = ("train", "val", "test"),
    augment_on_gpu: bool = False,
    drop_remainder: bool = False,
    input_threads: int = 0,
    deterministic: bool = True,
    host_cache: bool = False,
):
    """Return ``(train_ds, val_ds, test_ds)``.

    ``val_ds`` is ``None`` when the training pool was not held out. The test
    pipeline never repeats and never augments. ``host_cache`` keeps the training
    images in host RAM so later epochs do not reread the disk cache.
    """

    if image_size < 8:
        raise ValueError("image_size must be at least 8")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    repeats = index.repeats if repeat_factor is None else max(1, repeat_factor)
    aug_config = dict(augmentation or {})
    clean_probability = float(aug_config.pop("clean_probability", 0.0) or 0.0)
    if not 0.0 <= clean_probability < 1.0:
        raise ValueError("clean_probability must be in [0, 1)")
    use_cpu_augment = "train" in which and not augment_on_gpu
    autotune = tf.data.AUTOTUNE

    def pipeline(paths: list[str], labels: list[int], training: bool, split_name: str):
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))
        ds = ds.map(
            lambda path, label: _load_image(path, label, image_size),
            num_parallel_calls=autotune,
        )
        if cache == "memory":
            ds = ds.cache()
        elif cache:
            cache_path = Path(f"{cache}_{split_name}")
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            ds = ds.cache(str(cache_path))
        if training:
            ds = ds.repeat(repeats)
            buffer = min(max(len(paths) * repeats, 1), max(1, shuffle_buffer))
            ds = ds.shuffle(buffer, seed=seed, reshuffle_each_iteration=True)
        ds = ds.batch(batch_size, drop_remainder=bool(drop_remainder) if training else False)
        if training and use_cpu_augment:
            ds = ds.map(
                lambda images, batch_labels: _cpu_augment_batch(
                    images, batch_labels, image_size, aug_config, clean_probability
                ),
                num_parallel_calls=autotune,
            )
        return ds.prefetch(autotune)

    def with_runtime_options(dataset):
        if dataset is None or (int(input_threads) <= 0 and deterministic):
            return dataset
        options = tf.data.Options()
        options.deterministic = bool(deterministic)
        if int(input_threads) > 0:
            options.threading.private_threadpool_size = int(input_threads)
            options.threading.max_intra_op_parallelism = int(input_threads)
        return dataset.with_options(options)

    train_ds = None
    if "train" in which:
        if host_cache and index.train_paths:
            host_images, host_labels = _load_uint8_store(
                index.train_paths,
                index.train_labels,
                image_size,
                str(cache or ""),
                "train",
            )
            train_ds = _host_train_dataset(
                host_images,
                host_labels,
                image_size=image_size,
                batch_size=batch_size,
                seed=seed,
                repeats=repeats,
                shuffle_buffer=shuffle_buffer,
                drop_remainder=bool(drop_remainder),
                aug_config=aug_config,
                clean_probability=clean_probability,
                use_cpu_augment=use_cpu_augment,
            )
        else:
            train_ds = pipeline(index.train_paths, index.train_labels, training=True, split_name="train")
    val_ds = (
        pipeline(index.val_paths, index.val_labels, training=False, split_name="val")
        if "val" in which and index.val_paths
        else None
    )
    test_ds = (
        pipeline(index.test_paths, index.test_labels, training=False, split_name="test")
        if "test" in which and index.test_paths
        else None
    )
    return with_runtime_options(train_ds), with_runtime_options(val_ds), with_runtime_options(test_ds)
