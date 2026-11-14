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

import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import tensorflow as tf

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


class FallbackRandomShear(tf.keras.layers.Layer):
    """Random shear that stays active only while ``training=True``."""

    def __init__(self, x_factor: float = 0.2, y_factor: float = 0.0, fill_value: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.x_factor = float(x_factor)
        self.y_factor = float(y_factor)
        self.fill_value = float(fill_value)

    def call(self, images, training=None):
        if training is False:
            return images
        rank = tf.rank(images)

        def run(batched):
            batch = tf.shape(batched)[0]
            shx = tf.random.uniform([batch], -self.x_factor, self.x_factor)
            shy = tf.random.uniform([batch], -self.y_factor, self.y_factor)
            return shear_images(batched, shx, shy, self.fill_value)

        return tf.cond(
            tf.equal(rank, 3),
            lambda: run(images[tf.newaxis, ...])[0],
            lambda: run(images),
        )

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
    shear_cls = getattr(tf.keras.layers, "RandomShear", None)
    if shear_cls is not None:
        shear_kwargs = _layer_kwargs(
            shear_cls,
            {
                "x_factor": options["shear_x"],
                "y_factor": options["shear_y"],
                **fill,
                "name": "aug_shear",
            },
        )
        try:
            layers.append(shear_cls(**shear_kwargs))
        except TypeError:
            layers.append(
                FallbackRandomShear(
                    x_factor=options["shear_x"],
                    y_factor=options["shear_y"],
                    fill_value=1.0,
                    name="aug_shear",
                )
            )
    else:
        layers.append(
            FallbackRandomShear(
                x_factor=options["shear_x"],
                y_factor=options["shear_y"],
                fill_value=1.0,
                name="aug_shear",
            )
        )
    return tf.keras.Sequential(layers, name="train_augmentation")


def _load_image(path: tf.Tensor, label: tf.Tensor, image_size: int):
    raw = tf.io.read_file(path)
    image = tf.io.decode_image(raw, channels=1, expand_animations=False)
    image = tf.image.resize(image, [image_size, image_size], method="bilinear")
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.reshape(image, [image_size, image_size, 1])
    return image, tf.cast(label, tf.int32)


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
):
    """Return ``(train_ds, val_ds, test_ds)``.

    ``val_ds`` is ``None`` when the training pool was not held out. The test
    pipeline never repeats and never augments.
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
    augmenter = build_augmenter(image_size, aug_config) if "train" in which else None
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
        ds = ds.batch(batch_size, drop_remainder=False)
        if training:

            def augment(images, batch_labels):
                transformed = augmenter(images, training=True)
                transformed = tf.clip_by_value(transformed, 0.0, 1.0)
                if clean_probability > 0:
                    keep = tf.random.uniform([tf.shape(images)[0], 1, 1, 1]) < clean_probability
                    # Augmentation layers follow mixed_float16 on the GPU.
                    # Loaded images stay float32, and tf.where requires both.
                    transformed = tf.cast(transformed, images.dtype)
                    transformed = tf.where(keep, images, transformed)
                transformed.set_shape([None, image_size, image_size, 1])
                return transformed, batch_labels

            ds = ds.map(augment, num_parallel_calls=autotune)
        return ds.prefetch(autotune)

    train_ds = (
        pipeline(index.train_paths, index.train_labels, training=True, split_name="train")
        if "train" in which
        else None
    )
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
    return train_ds, val_ds, test_ds
