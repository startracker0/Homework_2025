# 花卉分类任务报告

## 一、模型信息

### 1. 模型名称
- **ResNet-50**（可通过参数选择为 ResNet-18、ResNet-34、ResNet-101）

### 2. 网络结构与修改
- **基础结构**：采用自定义实现的 ResNet 系列结构，分别基于 `BasicBlock`（ResNet-18/34）或 `Bottleneck`（ResNet-50/101）模块构建。
- **构建方式**：通过 `models/resnet.py` 中的 `ResNet` 类完成网络初始化：
  - ResNet-18: `[2, 2, 2, 2]`
  - ResNet-34: `[3, 4, 6, 3]`
  - ResNet-50: `[3, 4, 6, 3]`
  - ResNet-101: `[3, 4, 23, 3]`
- **分类头修改**：根据当前任务中类别数量 `num_classes`，自动设置最后一层全连接层为 `nn.Linear(..., num_classes)`。
- **训练配置**：
  - 损失函数：`CrossEntropyLoss`
  - 优化器：`Adam`
  - 学习率调度器：`ReduceLROnPlateau`（监控验证集 loss，`patience=5`, `factor=0.1`）

---

## 二、数据集处理
- 数据来源：`./flower_data` (如当前目录无该数据集，请下载)
- 数据划分：
  - 训练集：`./flower_data/train`
  - 验证集：`./flower_data/val`
- 预处理策略：
  - **训练集增强**：随机裁剪、旋转、颜色抖动、水平翻转等；
  - **验证集增强**：固定大小中心裁剪，标准化处理。

---

## 三、训练参数

- **设备**：自动检测（优先使用 GPU）
- **批量大小**：`--batch_size 32`
- **训练轮数**：`--epochs 20`
- **学习率**：`--lr 0.001`
- **模型输出路径**：`--output_dir ./outputs`
- **类别数**：根据训练集自动识别

---

## 四、训练与验证过程可视化
输入下示指令进行训练：
```bash
python train.py
```

> 模型训练过程中，损失与准确率变化如下图所示：

![训练过程曲线](./outputs/training_history.png)

---

## 五、测试结果与分析

- 模型在训练过程中自动保存验证集上最佳精度对应的模型为：`./outputs/best_model_resnet50.pth`

输入下示指令进行测试：
```bash
python test.py
```
### 1.测试配置信息

- **使用设备**: `cuda:0`
- **模型类型**: `resnet50`
- **训练时的类别数量**: `5`
- **测试集中的类别数量**: `5`
- **加载模型路径**: `./outputs/best_model_resnet50.pth`
- **测试数据目录**: `./flower_data/test`

### 2. 测试集测试结果
- 模型在测试过程中结果如下图所示，存在一个错误。
![测试结果可视化](./outputs/predictions.png)
