# 手写数字识别 - CNN 模型 (MNIST)

本项目使用卷积神经网络（CNN）对 MNIST 手写数字数据集进行分类识别。模型在测试集上达到了约 **99.36%** 的准确率，并支持 GPU 加速和混合精度训练。

所有项目代码放到一个jupyter notebook文件里，方便观看。

## 目录

- [项目简介](#项目简介)
- [环境要求](#环境要求)
- [数据集](#数据集)
- [模型架构](#模型架构)
- [训练细节](#训练细节)
- [结果与评估](#结果与评估)
- [如何使用](#如何使用)
- [文件说明](#文件说明)

## 项目简介

MNIST（Modified National Institute of Standards and Technology）是一个经典的手写数字数据集，包含 60,000 张训练图片和 10,000 张测试图片，每张图片为 28×28 的灰度图像。本项目利用 PyTorch 构建了一个卷积神经网络，实现了高精度的数字识别。

## 环境要求

- Python 3.12+
- PyTorch 2.0+ (建议使用 CUDA 版本以支持 GPU)
- torchvision
- numpy
- matplotlib
- scikit-learn

安装依赖：
```bash
pip install torch torchvision numpy matplotlib scikit-learn
```

## 数据集

数据通过 `torchvision.datasets.MNIST` 自动下载，并进行了预处理：
- 转换为张量（`ToTensor()`）
- 标准化：均值 0.1307，标准差 0.3081

```python
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
```

## 模型架构

CNN 模型结构如下：

| 层类型               | 参数                                             |
|---------------------|--------------------------------------------------|
| Conv2d              | 输入通道 1 → 输出通道 32，卷积核 3×3，padding 1 |
| BatchNorm2d         | 32                                              |
| ReLU                | -                                                |
| MaxPool2d           | 核大小 2×2，步长 2                              |
| Conv2d              | 32 → 64，卷积核 3×3，padding 1                 |
| BatchNorm2d + ReLU  | -                                                |
| MaxPool2d           | 2×2                                             |
| Conv2d              | 64 → 128，卷积核 3×3，padding 1                |
| BatchNorm2d + ReLU  | -                                                |
| MaxPool2d           | 2×2                                             |
| Dropout             | 0.5                                             |
| Linear              | 128×3×3 = 1152 → 256                            |
| ReLU + Dropout(0.5) | -                                                |
| Linear              | 256 → 10 (输出 10 个类别)                       |

## 训练细节

- **批量大小**：2048
- **优化器**：Adam，学习率 0.001
- **学习率调度**：每 10 个 epoch 乘以 0.1 (`StepLR`)
- **损失函数**：交叉熵损失
- **训练轮数**：50
- **混合精度**：使用 `torch.cuda.amp` 加速训练（如果 GPU 可用）
- **多 GPU**：自动检测并使用所有可用 GPU（通过 `nn.DataParallel`）

## 结果与评估

### 训练过程

- 训练约 5 分钟（在 Tesla T4 GPU 上）
- 最终测试准确率：**99.29%**（第 50 个 epoch）
- 最佳测试准确率：**99.36%**（第 15 个 epoch）

### 分类报告（测试集）

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

### 混淆矩阵

训练完成后会生成混淆矩阵热力图，直观展示每个数字的分类情况。

## 如何使用

### 1. 运行训练

直接执行 Jupyter Notebook 中的所有单元格即可开始训练。训练过程中会输出每个 epoch 的损失、准确率和耗时，并自动保存最佳模型为 `mnist_cnn_best.pth`。

### 2. 测试单张图片

Notebook 提供了 `predict_single_image` 函数，可对任意单张 MNIST 图片进行预测并输出置信度最高的前 3 个类别。

示例：
```python
pred_label, confidence, probs = predict_single_image(model, image, device)
```

### 3. 加载已训练模型

```python
model = CNN()
model.load_state_dict(torch.load('mnist_cnn_best.pth'))
model = model.to(device)
model.eval()
```

## 文件说明

- `mnist.ipynb`：完整的项目代码，包含数据处理、模型定义、训练、评估和可视化。

## 有事直接联系我：
我的SIP电话：`sip:anjuxi@sip.linphone.org`

---

**作者**：Ailan Anjuxi
**许可**：MIT
```
