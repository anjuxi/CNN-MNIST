# CNN Handwritten Digit Recognition on MNIST

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)

A CNN for classifying MNIST handwritten digits. The model reaches about **99.4%** test accuracy and supports GPU acceleration and mixed-precision training.

All code is in a single Jupyter Notebook so it's easy to review and reproduce.

## Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Results](#results)
- [Usage](#usage)
- [Files](#files)

## Overview

MNIST is a classic handwritten digit dataset: 60,000 training images and 10,000 test images, each 28×28 grayscale. This project builds a CNN in PyTorch for high-accuracy digit recognition.

## Dataset

Downloaded automatically via `torchvision.datasets.MNIST` and preprocessed:

- Converted to tensors (`ToTensor()`)
- Normalized: mean 0.1307, std 0.3081

```python
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
```

## Model Architecture

| Layer | Parameters |
|---------------------|--------------------------------------------------|
| Conv2d | 1 → 32 channels, 3×3 kernel, padding 1 |
| BatchNorm2d + ReLU | 32 |
| MaxPool2d | 2×2 kernel, stride 2 |
| Conv2d | 32 → 64, 3×3 kernel, padding 1 |
| BatchNorm2d + ReLU | 64 |
| MaxPool2d | 2×2 |
| Conv2d | 64 → 128, 3×3 kernel, padding 1 |
| BatchNorm2d + ReLU | 128 |
| MaxPool2d | 2×2 |
| Dropout | 0.3 |
| Linear | 128×3×3 = 1152 → 256 |
| ReLU + Dropout(0.3) | - |
| Linear | 256 → 10 (10 classes) |

## Training

- **Batch size**: 2048
- **Optimizer**: Adam, learning rate 0.001
- **LR schedule**: StepLR, ×0.1 every 10 epochs
- **Loss**: cross-entropy
- **Epochs**: 50
- **Mixed precision**: `torch.cuda.amp` when a GPU is available
- **Multi-GPU**: uses all available GPUs via `nn.DataParallel`

## Results

### Training

- About **5 minutes** on a Tesla T4 (batch size 2048)
- Best test accuracy: **99.41%** (epoch 12)
- Final test accuracy: **99.38%** (epoch 50)

### Classification report (test set)

```
              precision    recall  f1-score   support
           0       0.99      1.00      0.99       980
           1       0.99      1.00      1.00      1135
           2       0.99      1.00      0.99      1032
           3       0.99      0.99      0.99      1010
           4       0.99      0.99      0.99       982
           5       0.99      0.99      0.99       892
           6       1.00      0.99      0.99       958
           7       0.99      0.99      0.99      1028
           8       0.99      0.99      0.99       974
           9       0.99      0.98      0.99      1009

    accuracy                           0.99     10000
```

### Confusion Matrix

The notebook generates a confusion matrix heatmap after training.

## Usage

### 1. Run Training

Run all cells in the notebook. Each epoch prints loss, accuracy, and time. The best model is saved as `mnist_cnn_best.pth`.

### 2. Predict a Single Image

The notebook provides `predict_single_image`, which predicts on a single MNIST image and returns the top 10 classes with confidence scores.

Example:

```python
pred_label, confidence, probs = predict_single_image(model, image, device)
```

## Files

- `CNN_mnist.ipynb` — full project code: data loading, model definition, training, evaluation, and visualization.

## Contact

Email: `anjuxi.ME@outlook.com`

SIP: `sip:anjuxi@sip.linphone.org`

---
**Author**: Ailan Anjuxi
**License**: MIT
---
