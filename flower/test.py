import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import argparse
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

def imshow(inp, ax, title=None):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    ax.imshow(inp)
    if title is not None:
        ax.set_title(title, fontsize=20)  # 将字体大小从10增加到14
    ax.axis('off')
    
def save_prediction_visuals(dataset, all_preds, all_labels, idx_to_class, num_images=5, save_path='./outputs/predictions.png'):
    plt.figure(figsize=(3 * num_images, 3))
    for i in range(num_images):
        ax = plt.subplot(1, num_images, i + 1)
        img_tensor, _ = dataset[i]
        true_label = idx_to_class[all_labels[i]]
        pred_label = idx_to_class[all_preds[i]]
        title = f'{true_label} → {pred_label}'
        imshow(img_tensor, ax, title=title)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f'✅ 可视化预测结果已保存为: {save_path}')


# 参数设置
parser = argparse.ArgumentParser(description='测试花卉分类模型')
parser.add_argument('--data_dir', type=str, default='./flower_data', help='数据集路径')
parser.add_argument('--model_path', type=str,default='./outputs/best_model_resnet50.pth', help='训练好的模型路径')
parser.add_argument('--model_type', type=str, default='resnet50', choices=['resnet18', 'resnet34', 'resnet50', 'resnet101'], help='ResNet模型类型')
parser.add_argument('--batch_size', type=int, default=32, help='测试批量大小')
parser.add_argument('--class_mapping', type=str, default='./outputs/class_mapping.json', help='类别映射文件')
args = parser.parse_args()

# 设置设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 数据预处理
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 加载类别映射
with open(args.class_mapping, 'r') as f:
    class_to_idx = json.load(f)
    
num_classes = len(class_to_idx)
idx_to_class = {v: k for k, v in class_to_idx.items()}
print(f"训练时的类别数量: {num_classes}")

# 加载测试集
test_dataset = datasets.ImageFolder(os.path.join(args.data_dir, 'test'), transform)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
print(f"测试集中的类别数量: {len(test_dataset.classes)}")

# 加载模型 - 使用训练时的类别数量
print(f"加载 {args.model_type} 模型...")
if args.model_type == 'resnet18':
    model = models.resnet18()
elif args.model_type == 'resnet34':
    model = models.resnet34()
elif args.model_type == 'resnet101':
    model = models.resnet101()
else:
    model = models.resnet50()

# 使用训练时的类别数量初始化最后的全连接层
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)  # 使用训练时的类别数量
model.load_state_dict(torch.load(args.model_path, map_location=device))
model = model.to(device)
model.eval()

# 创建测试集类别到训练集类别的映射
test_class_to_idx = test_dataset.class_to_idx
test_class_mapping = {}
for test_class, test_idx in test_class_to_idx.items():
    if test_class in class_to_idx:
        test_class_mapping[test_idx] = class_to_idx[test_class]
    else:
        print(f"警告: 测试集中的类别 '{test_class}' 在训练集中不存在")

# 测试
correct = 0
total = 0
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in tqdm(test_loader, desc='Testing'):
        inputs = inputs.to(device)
        
        # 跳过在训练集中不存在的类别
        valid_indices = torch.tensor([i for i, label in enumerate(labels) 
                                     if label.item() in test_class_mapping.keys()])
        
        if len(valid_indices) == 0:
            continue
            
        selected_inputs = inputs[valid_indices]
        selected_labels = torch.tensor([test_class_mapping[label.item()] 
                                       for label in labels[valid_indices]], 
                                       device=device)
        
        outputs = model(selected_inputs)
        _, preds = torch.max(outputs, 1)

        total += selected_labels.size(0)
        correct += (preds == selected_labels).sum().item()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(selected_labels.cpu().numpy())

# 输出测试准确率
acc = correct / total if total > 0 else 0
print(f'\n✅ 测试集总样本: {total}')
print(f'✅ 准确预测: {correct}')
print(f'✅ 测试集准确率: {acc:.4f}')

# 打印部分预测结果
print("\n📋 部分预测示例（实际 → 预测）:")
for i in range(min(10, len(all_labels))):
    true_label = idx_to_class[all_labels[i]]
    pred_label = idx_to_class[all_preds[i]]
    print(f'{true_label} → {pred_label}')
    
save_prediction_visuals(test_dataset, all_preds, all_labels, idx_to_class)

