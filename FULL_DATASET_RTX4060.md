# Full dataset on this RTX 4060

Assignment: THEi DIT5411 Machine Learning, Character Recognition, due 2026-11-23.

The markable result is the full Traditional Chinese Handwriting Dataset: **13,065 characters**, **684,677 images**, and a model whose last layer has **13,065 neurons**. On this PC the full set is already extracted at:

```text
/root/work/handwritting-data/cleaned_data
```

Pass that path as `--data-dir`. The folder `data/cleaned_data(50_50)` is the 4,803-character practice set. Leave it there.

This PC is an RTX 4060 with 8 GB, driver 621.24. Ubuntu in WSL2 can see the card. The project virtualenv is Python 3.13.18 and TensorFlow 2.21.0, and that TensorFlow install does not see the GPU until the CUDA libraries below are installed. The system `python3` is 3.14 and has no TensorFlow wheel, so every command uses the virtualenv.

Do the commands in **Ubuntu**, from the project directory. From Windows PowerShell, enter Ubuntu with:

```powershell
wsl
```

Then:

```bash
cd /root/work/DIT5411-ChuLokHangThomson
source .venv/bin/activate
```

Every new Ubuntu window needs those two lines again. The prompt should show `(.venv)`.

`D:` is full. The Ubuntu disk has about 944 GB free. Keep the download and the extracted images on the Ubuntu disk.

## 1. Let TensorFlow see the RTX 4060

```bash
cd /root/work/DIT5411-ChuLokHangThomson
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install "tensorflow[and-cuda]"
python scripts/check_gpu.py
```

The last line of a good check is `GPU_OK`. The GPU list must name the RTX 4060. An empty list means later training would stay on the CPU.

This card is compute capability 8.9. The normal TensorFlow CUDA install is the right one. The special Blackwell wheel in the 5070 Ti notes is for that other machine.

## 2. Download the twelve zip parts

The full set is published as twelve pieces, `all_data.zip.001` through `all_data.zip.012`. Parts `.001` to `.011` are 95 MB each. Part `.012` is about 42 MB. Together they are about 1.1 GB.

On this PC the twelve parts, and the joined `all_data.zip`, are already in the project:

```text
/root/work/DIT5411-ChuLokHangThomson/handwritting_data_all
```

Check them with:

```bash
ls -lh /root/work/DIT5411-ChuLokHangThomson/handwritting_data_all/all_data.zip.*
```

You want twelve files, with the sizes above. A part that is only a few kilobytes did not finish downloading. A fresh clone, if you ever need one, goes outside the image folder:

```bash
cd /root/work
git clone https://github.com/chenkenanalytic/handwritting_data_all.git
```

## 3. Join the parts (the `cat`) and extract them

The dataset README tells you to do this:

```bash
cat all_data.zip* > all_data.zip
unzip all_data.zip -d some_folder
```

`cat` only sticks the twelve pieces into one zip. It does not create the character folders. A normal `unzip` also reads the Chinese filenames as the wrong encoding, so the folder names become garbage. The dataset page itself says that happens. Training would then learn those garbage names.

This PC is already past that step. The images are here:

```text
/root/work/handwritting-data/cleaned_data/<character>/<character>_0.png
```

The published zip stores numbered folders (`cleaned_data/10000/颮_0.png`) and puts the character in the file name. Those folders have been renamed to the character, which is the label the trainer reads. Sample numbers start at `_0`. The trainer sorts by that number and keeps the lowest 40 as the training pool.

The project link `data/cleaned_data` points at the same folder. Training commands below pass the absolute path so a shell working directory cannot pick up the 4,803-character set.

To extract again, write into the same parent and then rename the numbered folders before training:

```bash
cd /root/work/DIT5411-ChuLokHangThomson
source .venv/bin/activate
python scripts/extract_dataset.py \
  --zip /root/work/DIT5411-ChuLokHangThomson/handwritting_data_all/all_data.zip \
  --output /root/work/handwritting-data
ln -sfn /root/work/handwritting-data/cleaned_data data/cleaned_data
```

Run the join once. A second `cat all_data.zip* > all_data.zip` can pull the new zip back into itself and fill the disk. The extract script only reads files named `all_data.zip.*`, so it does not do that.

Keep `/root/work/handwritting-data/cleaned_data`. Training reads those PNGs.

## 4. Check the extract before training

```bash
cd /root/work/DIT5411-ChuLokHangThomson
source .venv/bin/activate
python - <<'PY'
from pathlib import Path
root = Path("/root/work/handwritting-data/cleaned_data")
folders = [p for p in root.iterdir() if p.is_dir()]
print("folders", len(folders))
print("example folders", sorted(p.name for p in folders)[:8])
print("yi files", len(list((root / "一").glob("*.png"))))
PY
```

Use that absolute path. `Path("~/work/...")` does not expand `~`, and Python then looks for a folder whose name starts with a tilde.

A good extract prints:

- `folders 13065`
- example folder names that are real characters, such as `一` or `丁`
- `yi files 51`

Counting every PNG is slow. The full set has 684,677 images. Three folder names in the published zip are odd (`p`, `p嶒`, `濈u`). They are part of the 13,065 classes.

If the count is 13065 and the names look right, go on. If the folder names are only numbers, the numbered zip folders were not renamed yet. The loader will treat `1598` as the class name instead of the character in the file name.

You do not split the files by hand. The trainer takes the 40 lowest sample numbers of each character (`永_1.png`, `永_2.png`, `永_10.png`, and so on) as the training pool and keeps every later file as the test set. Each epoch then shows 200 augmented views of each full class: the 40 real images, five times, with a new rotation, shift, zoom, scale, and shear on every view. Half of each batch is left upright, because the test images are upright. Nothing is written back into the image folders.

## 5. What `configs/rtx4060.yaml` is set to

The file is already in the project. It is the file this PC should pass to training.

| Setting | Value | Why |
| --- | --- | --- |
| Image size | 64 | Assignment runs use 64. The older checkpoint was 48 and is a different model. |
| Epochs | 15, patience 4 | Stops when training accuracy stops rising. |
| `custom_cnn` | batch 2048, lr 0.008 | The warp is on the CPU, so this batch is the bare network on the 8 GB card. |
| `residual_cnn` | batch 384, lr 0.0035 | A bit larger than the custom CNN. |
| `mobilenet_v2` | batch 128, lr 0.0002 | ImageNet fine-tune. |
| `efficientnet_b0` | batch 64, lr 0.00014 | Heaviest of the five on this card. |
| `mobilenet_v3_small` | batch 128, lr 0.0002 | ImageNet fine-tune. |
| Cache | `artifacts/tfcache/full` | Float tensors stay on disk. Training copies them into RAM once as uint8, about 2 GB. |
| One model at a time | yes | The 8 GB card trains one network. |

The learning rates are the laptop rates scaled by the square root of the batch-size increase. `configs/rtx5070ti.yaml` is for the 16 GB school card. Leave that file as it is.

## 6. Short GPU check on 20 characters

This proves the full-image pipeline and the GPU. It is not the result you submit.

```bash
cd /root/work/DIT5411-ChuLokHangThomson
source .venv/bin/activate
python -m src.train \
  --config configs/rtx4060.yaml \
  --data-dir /root/work/handwritting-data/cleaned_data \
  --model custom_cnn \
  --max-classes 20 \
  --epochs 1 \
  --cache memory \
  --run-name gpu_check_full
```

`--cache memory` is only for this 20-character check. The full run must keep the disk cache from the YAML, so the check does not write over `artifacts/tfcache/full`.

A good log names the GPU and says `mixed_float16`. The run is saved at `runs/custom_cnn/gpu_check_full`.

## 7. Train the five models

Plug the PC in. In Windows, set the power plan so the machine does not sleep. Sleep pauses Ubuntu in the middle of an epoch.

Preview the five commands:

```bash
cd /root/work/DIT5411-ChuLokHangThomson
source .venv/bin/activate
python scripts/train_all.py \
  --config configs/rtx4060.yaml \
  --data-dir /root/work/handwritting-data/cleaned_data \
  --dry-run
```

You should see five commands, with run names `rtx4060_custom_cnn`, `rtx4060_residual_cnn`, `rtx4060_mobilenet_v2`, `rtx4060_efficientnet_b0`, and `rtx4060_mobilenet_v3_small`.

If a `custom_cnn` run is already going, stop it with Ctrl+C first. The process that is already running will not pick up this file. Starting the same command again keeps the saved checkpoint and continues at the next epoch. The warp now runs on the CPU, and the GPU step is only the network. `custom_cnn` uses batch 2048. The first minute of the new process copies the training images from the disk cache into RAM. Time the epoch after that, from `ms/step`. About 47 ms/step at batch 2048 is one minute.

Start the real training:

```bash
python scripts/train_all.py \
  --config configs/rtx4060.yaml \
  --data-dir /root/work/handwritting-data/cleaned_data
```

What happens:

1. `scripts/check_gpu.py` runs. Training starts only if that check passes.
2. The models run one after another, in the order above.
3. The first epoch of each process copies the training images into RAM and compiles the graph, so it is slower than the next epoch. Leave `artifacts/tfcache/full` in place. That file is the source of the RAM copy.
4. Each finished model is scored on the test images. Test images are not augmented.
5. Weights and scores land in `runs/<model>/rtx4060_<model>/`.

Read `ms/step` in the log. Multiply that by the step count on the same line to see how long one epoch takes. `custom_cnn` at batch 2048 has about 1,274 steps, so 47 ms/step is about one minute. EfficientNet-B0 and the two MobileNets do more math on this 8 GB card, so their epochs can stay above one minute. The full job is five models and up to 15 epochs each. Leave this command running.

Start a new model only after the previous one has exited. A second training process on this card splits the 8 GB and is likely to run out of memory.

`custom_cnn` continues from `runs/custom_cnn/rtx4060_custom_cnn/checkpoint.keras` when that file is already there. The other four models start from their own initial weights, and the three pretrained networks start from ImageNet. Do not pass `--initial-weights` from `runs/custom_cnn/common_custom_cnn_v2` or `common_custom_cnn_gpu`. Those weights are the 4,803-character set at 48×48. The full model has a different output layer.

### If one model stops

The models that already finished stay on disk. Run only the names that are missing:

```bash
python scripts/train_all.py \
  --config configs/rtx4060.yaml \
  --data-dir /root/work/handwritting-data/cleaned_data \
  --models efficientnet_b0 mobilenet_v3_small
```

If the log says the GPU ran out of memory, open `configs/rtx4060.yaml` and lower **that model's** `batch_size` under `model_overrides`, then rerun with `--models` and only that name. For `custom_cnn`, use batch 1024 and learning rate 0.0056. Halving the batch is the change to make. For EfficientNet, 64 then 32 then 16. The checkpoint is still valid.

If Ubuntu itself is killed, Windows gave WSL about 15 GB of the 32 GB in this PC. Create `C:\Users\thoms\.wslconfig` with:

```ini
[wsl2]
memory=24GB
swap=8GB
```

From PowerShell run `wsl --shutdown`, open Ubuntu again, activate the virtualenv, and resume with `--models` for anything that has no `metrics.json`.

## 8. Collect the test scores

When all five folders contain `metrics.json`, build the comparison from those scores:

```bash
cd /root/work/DIT5411-ChuLokHangThomson
source .venv/bin/activate
python scripts/evaluate_final.py --from-metrics
python scripts/write_report.py --config configs/rtx4060.yaml
```

Open:

- `reports/final/comparison.md`
- `reports/final/comparison.csv`
- `reports/final/comparison.png`
- `ASSIGNMENT_README.md`

The recognition rate in the table is exact-match test accuracy. Top-5 is next to it. The best model is the one with the highest test accuracy.

Each `metrics.json` must say `"num_classes": 13065`. The comparison script says so when a score came from the smaller set. A 20-character check, and the older 4,803-character runs, stay out of that table once the `rtx4060_<model>` results exist.

For the GitHub deliverable, keep for each model:

- `best.keras`
- `config.json`
- `label_map.json`
- `metrics.json`
- `training_curves.png`

The README the assignment asks for should describe the augmentation, the five architectures, and the test accuracy of each. `ASSIGNMENT_README.md` is that write-up. The dataset citation is already in the project `README.md`.

## 9. Checks while it runs

| What you see | What it means |
| --- | --- |
| `GPU_OK` | TensorFlow can multiply matrices on the 4060. |
| `device: GPU` and `mixed_float16` | The real run is on the card. |
| `This run has 4803 classes` | The command is still pointed at `data/cleaned_data(50_50)`. Point `--data-dir` at `/root/work/handwritting-data/cleaned_data`. |
| `folders 13065` | The extract is the full set. |
| Folder names are numbers | Rename each numbered folder to the character in its file names before training. |
| `out of memory` or `OOM` | Lower that model's batch in `configs/rtx4060.yaml` and rerun that model. |
| `GPU list is empty` | The virtualenv is not active, or step 1 was skipped. |

The first full epoch also fills `artifacts/tfcache/full`. If you extract the dataset again, delete that cache folder before the next full training run so the new images are decoded again.
