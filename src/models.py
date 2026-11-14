"""Character classifiers that share one grayscale input shape.

Every builder returns a Keras model. The last layer is a float32 softmax so
mixed precision on the RTX 4060 does not compute the loss in float16.

Transfer models expand the single grayscale channel to three channels and scale
pixels to the range the ImageNet weights expect:

* MobileNetV2 has no internal rescaling. Its ``preprocess_input`` maps
  ``[0, 255]`` to ``[-1, 1]``.
* EfficientNetB0 and MobileNetV3 include a rescaling layer in current Keras
  and expect ``[0, 255]``. Older TensorFlow builds do not, so those builders
  call ``preprocess_input`` only when the constructor has no
  ``include_preprocessing`` argument.
"""

from __future__ import annotations

import inspect

import numpy as np
import tensorflow as tf

TRANSFER_MODELS = ("mobilenet_v2", "efficientnet_b0", "mobilenet_v3_small")
MODEL_NAMES = ("custom_cnn", "residual_cnn", *TRANSFER_MODELS)


def build_model(
    name: str,
    input_shape: tuple[int, int, int],
    num_classes: int,
    weights: str | None = "imagenet",
    freeze_base: bool = False,
    dropout: float = 0.4,
) -> tf.keras.Model:
    if num_classes < 2:
        raise ValueError("num_classes must be at least 2")
    if name not in MODEL_NAMES:
        raise ValueError(f"Unknown model {name!r}. Choose one of: {', '.join(MODEL_NAMES)}")

    builders = {
        "custom_cnn": build_custom_cnn,
        "residual_cnn": build_residual_cnn,
        "mobilenet_v2": build_mobilenet_v2,
        "efficientnet_b0": build_efficientnet_b0,
        "mobilenet_v3_small": build_mobilenet_v3_small,
    }
    kwargs = {}
    if name in TRANSFER_MODELS:
        if min(input_shape[0], input_shape[1]) < 32:
            raise ValueError(f"{name} needs an image size of at least 32. Got {input_shape}.")
        kwargs = {"weights": weights, "freeze_base": freeze_base}
    return builders[name](input_shape, num_classes, dropout=dropout, **kwargs)


def build_custom_cnn(
    input_shape: tuple[int, int, int],
    num_classes: int,
    dropout: float = 0.4,
) -> tf.keras.Model:
    """Three-stage CNN, shallow enough to iterate on CPU and deep enough to compare."""

    inputs = tf.keras.layers.Input(shape=input_shape, name="image")
    x = inputs
    for filters in (32, 64, 128):
        x = _conv_block(x, filters)
        x = tf.keras.layers.MaxPooling2D(name=f"pool_{filters}")(x)
        x = tf.keras.layers.Dropout(0.2, name=f"drop_{filters}")(x)
    outputs = _classifier(x, num_classes, dropout, width=512)
    return tf.keras.Model(inputs, outputs, name="custom_cnn")


def build_residual_cnn(
    input_shape: tuple[int, int, int],
    num_classes: int,
    dropout: float = 0.4,
) -> tf.keras.Model:
    """Small residual network. The shortcut keeps early stroke features available."""

    inputs = tf.keras.layers.Input(shape=input_shape, name="image")
    x = tf.keras.layers.Conv2D(32, 3, padding="same", use_bias=False, name="stem_conv")(inputs)
    x = tf.keras.layers.BatchNormalization(name="stem_bn")(x)
    x = tf.keras.layers.Activation("relu", name="stem_relu")(x)
    x = _residual_stage(x, 32, name="res32")
    x = _residual_stage(x, 64, stride=2, name="res64")
    x = _residual_stage(x, 128, stride=2, name="res128")
    outputs = _classifier(x, num_classes, dropout, width=512)
    return tf.keras.Model(inputs, outputs, name="residual_cnn")


def build_mobilenet_v2(
    input_shape: tuple[int, int, int],
    num_classes: int,
    dropout: float = 0.3,
    weights: str | None = "imagenet",
    freeze_base: bool = False,
) -> tf.keras.Model:
    inputs, features = _backbone_features(
        constructor=tf.keras.applications.MobileNetV2,
        preprocess=tf.keras.applications.mobilenet_v2.preprocess_input,
        input_shape=input_shape,
        weights=weights,
        freeze_base=freeze_base,
        constructor_extra={"alpha": 1.0},
    )
    outputs = _classifier(features, num_classes, dropout, width=512)
    return tf.keras.Model(inputs, outputs, name="mobilenet_v2")


def build_efficientnet_b0(
    input_shape: tuple[int, int, int],
    num_classes: int,
    dropout: float = 0.3,
    weights: str | None = "imagenet",
    freeze_base: bool = False,
) -> tf.keras.Model:
    inputs, features = _backbone_features(
        constructor=tf.keras.applications.EfficientNetB0,
        preprocess=tf.keras.applications.efficientnet.preprocess_input,
        input_shape=input_shape,
        weights=weights,
        freeze_base=freeze_base,
    )
    outputs = _classifier(features, num_classes, dropout, width=512)
    return tf.keras.Model(inputs, outputs, name="efficientnet_b0")


def build_mobilenet_v3_small(
    input_shape: tuple[int, int, int],
    num_classes: int,
    dropout: float = 0.3,
    weights: str | None = "imagenet",
    freeze_base: bool = False,
) -> tf.keras.Model:
    inputs, features = _backbone_features(
        constructor=tf.keras.applications.MobileNetV3Small,
        preprocess=tf.keras.applications.mobilenet_v3.preprocess_input,
        input_shape=input_shape,
        weights=weights,
        freeze_base=freeze_base,
    )
    outputs = _classifier(features, num_classes, dropout, width=512)
    return tf.keras.Model(inputs, outputs, name="mobilenet_v3_small")


def _grayscale_inputs(input_shape: tuple[int, int, int]):
    inputs = tf.keras.layers.Input(shape=input_shape, name="image")
    rgb = tf.keras.layers.Concatenate(name="grayscale_to_rgb")([inputs, inputs, inputs])
    rgb255 = tf.keras.layers.Rescaling(255.0, name="to_255")(rgb)
    return inputs, rgb255


def _backbone_features(
    constructor,
    preprocess,
    input_shape,
    weights,
    freeze_base,
    constructor_extra: dict | None = None,
):
    """Call a 3-channel backbone on grayscale images scaled to ``[0, 255]``.

    The backbone is built with its own ``input_shape`` and then called, instead
    of being wired through ``input_tensor``. That keeps the grayscale input and
    the ImageNet preprocessing in one graph on both Keras 2 and Keras 3.
    """

    height, width, _channels = input_shape
    signature = inspect.signature(constructor)
    constructor_args = {
        "include_top": False,
        "weights": _weight_arg(weights),
        "input_shape": (height, width, 3),
    }
    if constructor_extra:
        constructor_args.update(constructor_extra)
    # Current EfficientNet and MobileNetV3 rescale inside the model and expect
    # pixels in [0, 255]. MobileNetV2 does not, so it still needs preprocess_input.
    model_rescales = "include_preprocessing" in signature.parameters
    if model_rescales:
        constructor_args["include_preprocessing"] = True
    base = constructor(**constructor_args)
    base.trainable = not freeze_base

    inputs, rgb255 = _grayscale_inputs(input_shape)
    if model_rescales:
        features = base(rgb255)
    else:
        features = base(_match_preprocess(preprocess, rgb255))
    return inputs, features


def _match_preprocess(preprocess, rgb255):
    """Apply the same scaling as ``preprocess_input`` with serializable layers.

    A Lambda around ``preprocess_input`` often fails to reload. The ImageNet
    helpers used here are either a no-op or the TensorFlow ``x / 127.5 - 1`` map.
    """

    sample = np.linspace(0.0, 255.0, 5, dtype=np.float32)
    try:
        output = np.asarray(preprocess(sample.copy()), dtype=np.float32).reshape(-1)
    except Exception:
        output = None
    if output is not None and np.allclose(output, sample, atol=1e-3):
        return rgb255
    tf_mode = sample / 127.5 - 1.0
    if output is not None and np.allclose(output, tf_mode, atol=1e-2):
        return tf.keras.layers.Rescaling(1.0 / 127.5, offset=-1.0, name="preprocess_input")(rgb255)
    return tf.keras.layers.Lambda(preprocess, name="preprocess_input")(rgb255)


def _weight_arg(weights: str | None):
    if weights is None or str(weights).lower() in {"none", "null", "random"}:
        return None
    return weights


def _conv_block(x, filters: int):
    for index in (1, 2):
        x = tf.keras.layers.Conv2D(
            filters,
            3,
            padding="same",
            use_bias=False,
            name=f"conv_{filters}_{index}",
        )(x)
        x = tf.keras.layers.BatchNormalization(name=f"bn_{filters}_{index}")(x)
        x = tf.keras.layers.Activation("relu", name=f"relu_{filters}_{index}")(x)
    return x


def _residual_stage(x, filters: int, stride: int = 1, name: str = "res"):
    shortcut = x
    y = tf.keras.layers.Conv2D(filters, 3, strides=stride, padding="same", use_bias=False, name=f"{name}_conv1")(x)
    y = tf.keras.layers.BatchNormalization(name=f"{name}_bn1")(y)
    y = tf.keras.layers.Activation("relu", name=f"{name}_relu1")(y)
    y = tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"{name}_conv2")(y)
    y = tf.keras.layers.BatchNormalization(name=f"{name}_bn2")(y)
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = tf.keras.layers.Conv2D(
            filters, 1, strides=stride, padding="same", use_bias=False, name=f"{name}_proj"
        )(shortcut)
        shortcut = tf.keras.layers.BatchNormalization(name=f"{name}_proj_bn")(shortcut)
    y = tf.keras.layers.Add(name=f"{name}_add")([y, shortcut])
    return tf.keras.layers.Activation("relu", name=f"{name}_out")(y)


def _classifier(x, num_classes: int, dropout: float, width: int):
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = tf.keras.layers.Dense(width, activation="relu", name="dense")(x)
    x = tf.keras.layers.Dropout(dropout, name="head_dropout")(x)
    # float32 softmax keeps the loss stable when the GPU policy is mixed_float16.
    return tf.keras.layers.Dense(num_classes, activation="softmax", dtype="float32", name="predictions")(x)
