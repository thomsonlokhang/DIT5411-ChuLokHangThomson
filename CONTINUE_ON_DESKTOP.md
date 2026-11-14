# Continue this training run on the RTX 4060 desktop (WSL2 Ubuntu)

The laptop run was stopped on 14 November 2026 during epoch 2 of 8. Epoch 1 had already finished and its weights are saved. Copy this project after that stop, then continue on the desktop from the checkpoint below.

Do not use the Windows `.venv` inside Ubuntu. Build a new Linux environment.

## What to resume

| Item | Value |
| --- | --- |
| Checkpoint | `runs/custom_cnn/common_custom_cnn_v2/checkpoint.keras` |
| Finished epochs in that run | 1 of 8 |
| Epoch 1 training accuracy | 81.8% |
| Epoch 1 training loss | 0.650 |
| Learning rate | 0.0002 |
| Image size | 48 |
| Batch size | 32 |
| Data folder | `data/cleaned_data(50_50)` |
| Epochs still to run | 7 |

`best.keras` for this continued run does not exist yet. It is written only after the full run finishes. Load `checkpoint.keras`.

The older finished run is still in `runs/custom_cnn/common_custom_cnn`. That file is the model from before this continued run. Do not pass it to `--initial-weights` if you want to keep the epoch 1 result above.

## 1. Copy the project off the laptop

Copy the whole `DIT5411-ChuLokHangThomson` folder to the desktop. Include:

- `src`, `configs`, `scripts`, `requirements.txt`, this file
- `data/cleaned_data(50_50)` (the images are not in git)
- `runs/custom_cnn/common_custom_cnn_v2/checkpoint.keras`

You can skip `.venv`. It is a Windows environment and will not run in Ubuntu.

The GitHub repo has no commits yet, so `git clone` does not contain this code or the images.

## 2. Install the NVIDIA driver on Windows

Install a current GeForce Game Ready or Studio driver on the desktop. Do not install a second NVIDIA driver inside Ubuntu. WSL2 uses the Windows driver.

## 3. Install Ubuntu

In Windows PowerShell on the desktop:

```powershell
wsl --install -d Ubuntu
```

Restart if Windows asks. Open **Ubuntu** from the Start menu and create the Linux username.

## 4. Check that Ubuntu sees the RTX 4060

Inside the Ubuntu window:

```bash
nvidia-smi
```

The table should name the RTX 4060. If the command is missing, update the Windows NVIDIA driver, then in PowerShell run `wsl --shutdown`, and open Ubuntu again.

## 5. Put the project on the Linux disk

Reading tens of thousands of images from `/mnt/c` or `/mnt/d` is slow. Copy the folder into the Ubuntu home directory.

If the copied folder is at `D:\DIT5411-ChuLokHangThomson` on the desktop:

```bash
mkdir -p ~/work
cp -a /mnt/d/DIT5411-ChuLokHangThomson ~/work/DIT5411-ChuLokHangThomson
cd ~/work/DIT5411-ChuLokHangThomson
rm -rf .venv
```

If the desktop drive letter is not `D:`, change `/mnt/d` to that letter in lowercase, for example `/mnt/e`.

Check the images and the checkpoint:

```bash
ls "data/cleaned_data(50_50)" | head
ls -lh runs/custom_cnn/common_custom_cnn_v2/checkpoint.keras
```

## 6. Create the Linux environment

`python3 --version` must be 3.10, 3.11, 3.12, or 3.13. Python 3.14 has no TensorFlow wheel.

```bash
cd ~/work/DIT5411-ChuLokHangThomson
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install "tensorflow[and-cuda]"
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

The last line must print a GPU device. An empty list means training would stay on the CPU.

Every new Ubuntu terminal needs the environment again:

```bash
cd ~/work/DIT5411-ChuLokHangThomson
source .venv/bin/activate
```

## 7. Short GPU check

This trains 20 characters for one epoch. It does not replace the real run.

```bash
python -m src.train \
  --data-dir "data/cleaned_data(50_50)" \
  --model custom_cnn \
  --image-size 48 \
  --batch-size 32 \
  --epochs 1 \
  --max-classes 20 \
  --target-samples-per-class 80 \
  --cache memory \
  --run-name gpu_check
```

The log should name the GPU and say `mixed_float16`.

## 8. Continue the real run

From `~/work/DIT5411-ChuLokHangThomson` with the environment active:

```bash
python -m src.train \
  --data-dir "data/cleaned_data(50_50)" \
  --model custom_cnn \
  --image-size 48 \
  --batch-size 32 \
  --epochs 7 \
  --learning-rate 0.0002 \
  --target-samples-per-class 200 \
  --cache memory \
  --patience 4 \
  --initial-weights runs/custom_cnn/common_custom_cnn_v2/checkpoint.keras \
  --run-name common_custom_cnn_gpu
```

Keep the quotes around the data directory. `(50_50)` is part of the folder name.

Keep `--image-size 48`. Those weights were trained at 48×48.

`--epochs 7` is the remaining part of the 8-epoch plan. The new run counts those 7 as its own epochs. Results go to:

`runs/custom_cnn/common_custom_cnn_gpu`

When it finishes, read `metrics.json` for the test accuracy and top-5 accuracy. The final weights are `best.keras`. The curves are `training_curves.png`.

Half of each training batch is left as the original character (`clean_probability: 0.5` in `configs/default.yaml`). The other half is still rotated, shifted, zoomed, and sheared. Leave that file as it is for this continued run.

## 9. If you would rather start over on the GPU

A fresh run does not load the laptop checkpoint and can use the 64×64 default:

```bash
python -m src.train \
  --data-dir "data/cleaned_data(50_50)" \
  --model custom_cnn \
  --epochs 15 \
  --target-samples-per-class 200 \
  --cache memory \
  --run-name common_custom_cnn_gpu_fresh
```

Do not pass `--initial-weights` on that command.

## Notes

- Batch 32 fits in the 8 GB 4060 for this model. If a later MobileNet or EfficientNet run runs out of memory, use `--batch-size 16`.
- `cpu_threads: 10` only sizes the CPU pool that prepares images. The 4060 does the network math.
- One epoch on the laptop was about 70 minutes. On the 4060 the same epoch is often a few minutes to about 15 minutes. The exact pace is the `ms/step` line in the log.
