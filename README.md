# DIT5411 Traditional Chinese Handwritten Character Recognition

TensorFlow / Keras recognizer for the [Traditional Chinese Handwriting Dataset](https://github.com/AI-FREE-Team/Traditional-Chinese-Handwriting-Dataset) (Tegaki-based, 13,065 character classes, 684,677 images). The same code runs on:

- **This laptop** — AMD Ryzen AI 9 365, 32 GB RAM, Radeon 880M. Native Windows has no CUDA device, so TensorFlow uses the CPU.
- **Desktop** — NVIDIA RTX 4060 8 GB. Use the GPU from WSL2. Native Windows GPU builds stop at TensorFlow 2.10, so that path is documented separately.

Images are resized to 64×64 grayscale. Training uses an on-the-fly augmented view of the official split, and the test set is never augmented.

## Decisions that the assignment leaves open

- **“First 40” means the 40 lowest sample numbers in the filename**, for example `永_1.png` then `永_2.png` then `永_10.png`. Directory listing order changes between Windows and Linux, so it is not used. The list is sorted, then cut. It is not shuffled before the cut.
- **Every one of those 40 images is a training image** (`val_ratio: 0`). Early stopping watches training accuracy. Set `--val-ratio 0.1` only when you want the tail of those 40 held out for checkpoint selection. The test split stays the images after the first 40.
- **One epoch shows 200 augmented views per full class.** Forty real images repeated 5 times, with a new random transform on every view. Nothing is written to disk.
- **Default image size is 64.** `--image-size 48` is the faster CPU setting. Transfer models need at least 32.
- **Five models** share that input: a custom CNN, a small residual CNN, MobileNetV2, EfficientNetB0, and MobileNetV3Small.

## Layout

```
src/data_pipeline.py    dataset discovery, first-40 split, tf.data, augmentation
src/models.py           the five classifiers
src/train.py            device detection, training, test metrics
src/evaluate.py         reload a run, or compare several runs
configs/default.yaml    full-assignment hyperparameters
configs/laptop_debug.yaml
scripts/extract_dataset.py
scripts/smoke_test.py   synthetic end-to-end check, no real dataset
notebooks/01_check_split.ipynb
```

`data/`, `runs/`, `reports/`, `artifacts/`, and `.venv/` are gitignored.

## 1. Dataset

The images are not in this repository. The dataset is CC BY-NC-SA 4.0. Cite it if you use it:

```
@misc{AI.FREE2020,
  author = {Po-Chuan Chen},
  title = {Traditional Chinese Handwriting Dataset},
  year = {2020},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/AI-FREE-Team/Traditional-Chinese-Handwriting-Dataset}}
}
```

### Full set (required for the assignment)

13,065 classes, 300×300 images, average about 52 samples per class.

```powershell
git clone https://github.com/chenkenanalytic/handwritting_data_all.git D:\handwritting_data_all
.\.venv\Scripts\python.exe scripts\extract_dataset.py --parts-dir D:\handwritting_data_all --output D:\handwriting-data
```

The script joins `all_data.zip.001` … `all_data.zip.012` and decodes filenames as Big5. A plain unzip on Windows often turns the character folders into mojibake. After extraction you should have:

```
D:\handwriting-data\cleaned_data\<character>\<character>_1.png
```

Delete `D:\handwriting-data\all_data.zip` after a successful extract if you need the disk space back. The combined archive is large.

The loader also accepts flat files named `<character>_<number>.png`. Pass `--organize` to the extract script when the common-character zips come out flat.

### Smaller common-character set (debug only)

4,803 classes at 50×50. Useful for a first real run. It is not a substitute for the 13,065-class result.

```powershell
git clone https://github.com/AI-FREE-Team/Traditional-Chinese-Handwriting-Dataset.git D:\tch-common
Get-ChildItem D:\tch-common\data\*.zip | ForEach-Object {
  .\.venv\Scripts\python.exe scripts\extract_dataset.py --zip $_.FullName --output D:\handwriting-data-common --organize
}
```

Those four zips are the 50×50 common-character release. If the folder names look garbled, run the same command with `--encoding utf-8`. Some strokes are hard to read at 50×50. Use the full 300×300 release for the result you report.

## 2. Environment on this AMD laptop

Use **Python 3.13**. The `python` command on this PC is the Microsoft Store alias, and Python 3.14 does not have a TensorFlow wheel.

```powershell
cd D:\DIT5411-ChuLokHangThomson
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -c "import tensorflow as tf; print(tf.__version__, tf.config.list_physical_devices('GPU'))"
```

An empty GPU list is the expected CPU result. Confirm the pipeline before the real download finishes:

```powershell
.\.venv\Scripts\python.exe scripts\smoke_test.py
```

That writes synthetic characters under `artifacts/`, trains `custom_cnn` for one epoch, and evaluates the held-out images.

## 3. Environment on the RTX 4060 desktop

Current TensorFlow GPU packages are Linux packages. On Windows, install Ubuntu in WSL2, install the NVIDIA driver on Windows, then inside Ubuntu:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install "tensorflow[and-cuda]"
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

`src/train.py` turns on GPU memory growth and `mixed_float16` when a GPU is visible. The softmax stays in float32. Batch size 32 at 64×64 fits an 8 GB card for these models. If a transfer model runs out of memory, use `--batch-size 16`.

### Native Windows GPU (only if you stay on TensorFlow 2.10)

TensorFlow 2.11 and later do not ship CUDA builds for native Windows. A 2.10 environment needs Python 3.10, CUDA 11.2, and cuDNN 8.1:

```powershell
py -3.10 -m venv .venv-tf210
.\.venv-tf210\Scripts\python.exe -m pip install "tensorflow<2.11" numpy PyYAML matplotlib Pillow
```

The training code stays on the public Keras APIs used by 2.10 and by current 2.x. Prefer WSL2 when you can, so the desktop and the laptop share one requirements file.

## 4. Train

From the repository root:

```powershell
# Laptop: 20 classes, 48x48, short run
.\.venv\Scripts\python.exe -m src.train --config configs\laptop_debug.yaml --data-dir D:\handwriting-data\cleaned_data

# One architecture on the full label set
.\.venv\Scripts\python.exe -m src.train --data-dir D:\handwriting-data\cleaned_data --model custom_cnn --epochs 15

# The other architectures. Fine-tuning uses a smaller learning rate.
.\.venv\Scripts\python.exe -m src.train --data-dir D:\handwriting-data\cleaned_data --model residual_cnn
.\.venv\Scripts\python.exe -m src.train --data-dir D:\handwriting-data\cleaned_data --model mobilenet_v2 --learning-rate 0.0001
.\.venv\Scripts\python.exe -m src.train --data-dir D:\handwriting-data\cleaned_data --model efficientnet_b0 --learning-rate 0.0001
.\.venv\Scripts\python.exe -m src.train --data-dir D:\handwriting-data\cleaned_data --model mobilenet_v3_small --learning-rate 0.0001
```

The first transfer-learning run downloads ImageNet weights. `--weights none` skips that download. `--freeze-base` trains only the new classifier head, which is the practical CPU setting for MobileNet and EfficientNet.

Useful flags:

| Flag | Role |
| --- | --- |
| `--max-classes 50` | First 50 characters in Unicode order, for debugging |
| `--classes-file classes.txt` | One character per line, in the label order you want |
| `--image-size 48` | Smaller tensors for the CPU |
| `--target-samples-per-class 200` | Augmented views per class per epoch. 40 real images ⇒ repeat ×5 |
| `--cache memory` | Keep decoded tensors in RAM. Safe for a few hundred classes |
| `--val-ratio 0.1` | Hold out the tail of the 40 training samples for early stopping |

A run is saved under `runs/<model>/<timestamp>/`:

```
best.keras  checkpoint.keras  config.json  label_map.json
split_summary.json  history.json  training_log.csv
training_curves.png  metrics.json  model_summary.txt
```

`metrics.json` includes the test accuracy and top-5 accuracy computed once after training. Top-5 matters once the label set is in the thousands.

Full-data scale, with the defaults: 13,065 × 40 × 5 = about 2.6 million augmented training views per epoch, and about 160 thousand test images. That epoch is a desktop-GPU job. On this laptop, use `configs/laptop_debug.yaml` or `--max-classes` until the code path is proven, then run the full comparison on the 4060.

## 5. Evaluate and compare

```powershell
.\.venv\Scripts\python.exe -m src.evaluate --run-dir runs\custom_cnn\<timestamp> --data-dir D:\handwriting-data\cleaned_data

.\.venv\Scripts\python.exe -m src.evaluate --compare runs\custom_cnn\<timestamp> runs\residual_cnn\<timestamp> runs\mobilenet_v2\<timestamp> runs\efficientnet_b0\<timestamp> runs\mobilenet_v3_small\<timestamp>
```

The comparison writes `reports/comparison.md`, `reports/comparison.csv`, and `reports/comparison.png`. Compare runs that used the same characters, image size, and split. A 20-class debug accuracy is not comparable to a 13,065-class accuracy.

`--reevaluate` recomputes test metrics. The reloaded label map fixes the class order, so class 0 remains the same character as in training.

## 6. Augmentation

Training images are resized to grayscale and scaled to `[0, 1]`. Each training view then gets:

- rotation, default ±15°
- translation, default ±10%
- zoom, default ±15%
- a second, anisotropic scale (width a bit stronger than height)
- shear on both axes

Empty pixels are filled with white (`1.0`). The Keras default fill is black, which would paint a box around the ink and give the model a shortcut. Test and validation batches skip this stack. Factors live under `augmentation` in the YAML config.

Repeating the 40 real images is what produces “at least 200 samples per class”. The repeats are shuffled, and the random layers run after the batch, so the five views of one source image are different. Classes with fewer than 40 files keep every file for training, get a warning in `split_summary.json`, and contribute no test images.

## 7. Models

All five take `(image_size, image_size, 1)` and a softmax over the classes in the run.

| Name | Idea |
| --- | --- |
| `custom_cnn` | Three conv stages (32, 64, 128), global average pooling, dense head |
| `residual_cnn` | Stem plus three residual stages, same head |
| `mobilenet_v2` | ImageNet MobileNetV2. Grayscale is stacked into 3 channels, scaled to `[0, 255]`, then mapped to `[-1, 1]` |
| `efficientnet_b0` | ImageNet EfficientNetB0. Current Keras rescales `[0, 255]` inside the network |
| `mobilenet_v3_small` | ImageNet MobileNetV3Small, same `[0, 255]` convention as EfficientNet |

The custom and residual networks are the ones that finish a debug run on this CPU in a reasonable time. The transfer networks are larger and, with the base unfrozen, are much faster on the 4060. Frozen-base transfer (`--freeze-base`) is a fair CPU baseline and a weak handwriting model, because ImageNet features were not trained on ink.

## 8. What differs between the two machines

- **Device.** The laptop log line says `device: CPU` and the policy is `float32`. The desktop log line names the GPU and uses `mixed_float16`.
- **Wall time.** The same epoch is limited by host RAM and CPU throughput here, and by the GPU there. Do the architecture comparison on the desktop once a `--max-classes` run has succeeded on both.
- **Batch size.** 32 is the default for both. Drop to 16 if the 4060 runs out of memory while fine-tuning EfficientNet.
- **Accuracy.** A model trained on the CPU and a model trained on the GPU can differ slightly under mixed precision. Record the device from `metrics.json` next to each number. Use the same seed (`5411`) when you want the data order to match.

## 9. Reproduce a reported number

1. Extract the full dataset with `scripts/extract_dataset.py` so folder names are real characters.
2. Train with `configs/default.yaml` and an explicit `--model`. Leave `--max-classes` unset for the final table.
3. Keep `runs/<model>/<timestamp>/` (at least `best.keras`, `config.json`, `label_map.json`, `metrics.json`, `training_curves.png`).
4. Rebuild the table with `python -m src.evaluate --compare ...`.

The test membership is a function of the filenames and `train_per_class`. It does not depend on the shuffle seed. The seed affects weight init and the order of augmented training views.

## Troubleshooting

- **`python` opens the Store.** Use `py -3.13` or `.\.venv\Scripts\python.exe`.
- **No images found.** Pass the `cleaned_data` directory, not the zip. The loader also descends into `cleaned_data` if you pass its parent.
- **Garbled folder names.** Re-extract with `scripts/extract_dataset.py`. The training script will not guess the encoding of a folder that was already unpacked incorrectly.
- **A class has no test score.** It had 40 or fewer source images. Those counts are listed in `split_summary.json`.
- **GPU list is empty on the desktop.** You are on native Windows with TensorFlow 2.11+, or WSL2 cannot see the NVIDIA driver. The program still trains on CPU.
- **Out of memory.** Lower `--batch-size`, use `--image-size 48`, or set `--freeze-base` for a transfer model.
