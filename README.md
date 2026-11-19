# DIT5411 Machine Learning — Character Recognition

- Student: Chu Lok Hang Thomson
- Course: THEi DIT5411 Machine Learning
- Assignment period: 2026-11-11 to 2026-11-23
- Code: https://github.com/thomsonlokhang/DIT5411-ChuLokHangThomson

## Objective

Build a TensorFlow program that recognises Traditional Chinese handwritten characters. The official split uses the first 40 samples of each character for training and the remaining samples for testing. The network has several convolutional layers and a softmax with one output neuron per character. Because each character supplies only 40 training images, training uses augmented views, at least 200 per character. Several architectures are trained, and the one with the highest recognition rate on the testing set is the selected model.

## Dataset and split

The images come from the Tegaki-based Traditional Chinese Handwriting Dataset. Files are ordered by the sample number in the filename (`永_2` before `永_10`), then cut. The list is not shuffled before the cut, so the test membership does not depend on Windows or Linux directory order. Every one of those 40 images is used for training (`val_ratio` 0). Images after the first 40 are the test set. A character with fewer than 40 files keeps all of them for training and contributes no test images.

The runs in the table below use the full set of 13,065 characters.

## Augmented image creation

Augmented images are created on the fly with TensorFlow image layers. Nothing is written back to the dataset folder. Each training image is resized to grayscale and scaled to [0, 1]. A training view can then receive:

- rotation of ±15 degrees
- translation of ±0.1 of the image size, on both axes
- zoom of ±0.15
- a second anisotropic scale of about ±0.1, wider on the x axis
- shear of ±0.25 horizontally and ±0.15 vertically

Empty pixels are filled with white (1.0). A black fill would draw a box around the ink and give the model a shortcut that is not part of the character. The 40 real training images of a full class are repeated 5 times per epoch, which is 200 views and meets the requirement of at least 200 samples per character. Each repeat draws a new random transform, so the repeats are not five copies of one picture. The test set skips this stack.

A fraction 0.50 of each training batch is left as the original resized glyph. The rest receives the random transform.

These layers run in the training data pipeline. The test pipeline does not apply them.

### Does rotation raise accuracy?

Yes, when the angle stays small and the upright glyph is still shown. Handwritten characters are slightly tilted, so a rotation of ±15° teaches a variation that appears in real writing. The test images are upright and unwarped. If every training view is rotated, sheared, and scaled, the network fits that warped look and its test score drops. An earlier run that warped every view reached about 76% on those warped training images and about 20.8% exact accuracy on the clean test set. Large angles such as 90° or 180° are not used: they are not how these characters are written, and some shapes can be confused with a different character. The originals on disk are never rotated. Each view is drawn at random during training.

## Models

Five architectures share the same grayscale input, the same split, and the same augmentation. The assignment asks for several models with different layers. The two networks written for this project and the three ImageNet networks are different families. All of them end in a float32 softmax so mixed precision on the GPU does not compute the loss in float16.

### custom_cnn

A network written for this assignment. The input is one grayscale image. Three stages use 32, 64, then 128 filters. Each stage is two 3×3 convolutions, batch normalisation, ReLU, 2×2 max pooling, and dropout 0.2. Global average pooling feeds a 512-unit ReLU layer, dropout 0.4, and a float32 softmax. The softmax has one neuron per character.

This run used 64×64 input, batch size 2048, learning rate 0.008, 15 epoch(s), and 7,056,169 parameters.

### residual_cnn

A second architecture, also written for this assignment. A 3×3 stem produces 32 channels. Three residual stages follow, at 32, 64, and 128 channels. The 64 and 128 stages downsample with stride 2, and a projection shortcut is used when the shape changes. The classifier head matches the custom CNN. The shortcut keeps early stroke features available to the later layers.

This run used 64×64 input, batch size 384, learning rate 0.0035, 15 epoch(s), and 7,076,521 parameters.

### mobilenet_v2

ImageNet MobileNetV2 (alpha 1.0) without its original classifier. The grayscale glyph is stacked into three channels, scaled to [0, 255], then mapped to [-1, 1] with the MobileNetV2 preprocessing. A new 512-unit head and a float32 softmax replace the ImageNet head. The base is fine-tuned rather than frozen, because ImageNet features were not trained on ink.

This run used 64×64 input, batch size 128, learning rate 0.0002, 15 epoch(s), and 9,616,201 parameters.

### efficientnet_b0

ImageNet EfficientNetB0 without its original classifier. Grayscale is stacked into three channels and scaled to [0, 255]. Current Keras EfficientNet rescales inside the network. The new head is a 512-unit layer, dropout, and a float32 softmax over the characters. The base is fine-tuned.

This run used 64×64 input, batch size 64, learning rate 0.00014, 15 epoch(s), and 11,407,788 parameters.

### mobilenet_v3_small

ImageNet MobileNetV3Small without its original classifier. It uses the same [0, 255] input convention as EfficientNet, then a new 512-unit head and a float32 softmax. The base is fine-tuned. This is a third pretrained architecture, smaller than EfficientNetB0, so the comparison is not two copies of one network.

This run used 64×64 input, batch size 128, learning rate 0.0002, 15 epoch(s), and 7,936,889 parameters.

## Recognition accuracy on the testing set

The test images are not augmented. Recognition rate means exact-match accuracy: the predicted character is the true character. Top-5 counts a success when the true character is one of the five highest probabilities. That second number matters once the label set is in the thousands, and the selection below uses the exact-match rate, which is what the brief calls the recognition rate.

| Model | Classes | Image | Parameters | Epochs | Test recognition rate | Top-5 | Device |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| custom_cnn | 13065 | 64 | 7,056,169 | 15 | 83.53% | 96.93% | GPU (/physical_device:GPU:0) |
| residual_cnn | 13065 | 64 | 7,076,521 | 15 | 79.52% | 95.14% | GPU (/physical_device:GPU:0) |
| mobilenet_v2 | 13065 | 64 | 9,616,201 | 15 | 83.75% | 96.08% | GPU (/physical_device:GPU:0) |
| efficientnet_b0 | 13065 | 64 | 11,407,788 | 15 | 93.24% | 99.07% | GPU (/physical_device:GPU:0) |
| mobilenet_v3_small | 13065 | 64 | 7,936,889 | 15 | 62.55% | 84.24% | GPU (/physical_device:GPU:0) |

The best model on the testing set is **efficientnet_b0**, with a recognition rate of 93.24% (top-5 99.07%). The weights and the log are in `runs/efficientnet_b0/rtx4060_efficientnet_b0`.

## How to reproduce

From the repository root, with the project virtualenv active:

```bash
python scripts/check_gpu.py
python scripts/train_all.py --config configs/rtx4060.yaml --data-dir /root/work/handwritting-data/cleaned_data
python scripts/evaluate_final.py --from-metrics
python scripts/write_report.py --config configs/rtx4060.yaml --data-dir /root/work/handwritting-data/cleaned_data
```

`train_all.py` trains `custom_cnn`, `residual_cnn`, `mobilenet_v2`, `efficientnet_b0`, and `mobilenet_v3_small` one after another. One GPU trains one model at a time.

## References

- Po-Chuan Chen, Traditional Chinese Handwriting Dataset, 2020. https://github.com/AI-FREE-Team/Traditional-Chinese-Handwriting-Dataset
- Full 13,065-character image release used by the loader: https://github.com/chenkenanalytic/handwritting_data_all
- Darshita Jain, Introduction to OpenCV (rotation, shearing, and scaling, the geometric transforms cited in the assignment brief). https://github.com/darshitajain/Introduction-to-OpenCV
- Kim Analytics, Handwritten Digit Recognition using Keras and TensorFlow (format reference named in the brief). https://github.com/kimanalytics/Handwritten-Digit-Recognition-using-Keras-and-TensorFlow
- Martín Abadi et al., TensorFlow: large-scale machine learning on heterogeneous systems, 2015. https://www.tensorflow.org/
