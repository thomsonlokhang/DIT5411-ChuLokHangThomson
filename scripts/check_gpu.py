"""Confirm TensorFlow can run work on the NVIDIA GPU before a long training job.

    python scripts/check_gpu.py

Exit code 0 means a GPU matmul finished. An RTX 5070 Ti needs a TensorFlow
build with sm_120 kernels. Native Windows wheels do not provide that.
"""

from __future__ import annotations

import sys


def main() -> int:
    try:
        import tensorflow as tf
    except Exception as exc:
        print(f"TensorFlow did not import: {exc}")
        return 1

    print(f"TensorFlow {tf.__version__}")
    gpus = tf.config.list_physical_devices("GPU")
    print(f"GPUs: {gpus or 'none'}")
    if not gpus:
        print(
            "No CUDA device is visible. An RTX 5070 Ti on native Windows will look "
            "like this with current TensorFlow wheels. Use Ubuntu or WSL2 and a "
            "build that contains sm_120 kernels, then run this script again."
        )
        return 1

    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as exc:
            print(f"Memory growth was already set ({exc}).")
        try:
            details = tf.config.experimental.get_device_details(gpu)
        except Exception as exc:
            details = {"error": str(exc)}
        print(f"{gpu.name}: {details}")

    try:
        with tf.device("/GPU:0"):
            sample = tf.random.normal((1024, 1024))
            _ = tf.matmul(sample, sample).numpy()
    except Exception as exc:
        print("A GPU matmul failed.")
        print(exc)
        print(
            "An RTX 5070 Ti is compute capability 12.0. If the message says the "
            "kernels are missing, or CUDA_ERROR_INVALID_PTX, this TensorFlow build "
            "cannot run the card. Install a wheel compiled for sm_120 and CUDA 12.8 "
            "or newer. An unofficial TensorFlow 2.21 Linux build is published at "
            "https://github.com/Syraxius/TensorflowDockerBuilder/releases/tag/v2.21.0-sm120-cuda128-py310-py313"
        )
        print("If the message names the allocator, retry with TF_GPU_ALLOCATOR=default.")
        return 1

    print("GPU_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
