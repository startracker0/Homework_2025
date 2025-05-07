import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import argparse
from models.resnet import ResNet, BasicBlock, Bottleneck

# 命令行参数
parser = argparse.ArgumentParser(description='花卉分类ResNet训练')
parser.add_argument('--batch_size', type=int, default=32, help='批量大小')
parser.add_argument('--epochs', type=int, default=20, help='训练轮数')
parser.add_argument('--lr', type=float, default=0.001, help='学习率')
parser.add_argument('--data_dir', type=str, default='./flower_data', help='数据集路径')
parser.add_argument('--model_type', type=str, default='resnet50', choices=['resnet18', 'resnet34', 'resnet50', 'resnet101'], help='ResNet模型类型')
parser.add_argument('--output_dir', type=str, default='./outputs', help='输出目录')
args = parser.parse_args()

# 创建输出目录
os.makedirs(args.output_dir, exist_ok=True)

# 设置设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 数据预处理
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# 加载数据集
print("加载数据集...")
image_datasets = {
    'train': datasets.ImageFolder(os.path.join(args.data_dir, 'train'), data_transforms['train']),
    'valid': datasets.ImageFolder(os.path.join(args.data_dir, 'val'), data_transforms['valid']),
}

dataloaders = {
    'train': DataLoader(image_datasets['train'], batch_size=args.batch_size, shuffle=True, num_workers=4),
    'valid': DataLoader(image_datasets['valid'], batch_size=args.batch_size, shuffle=False, num_workers=4)
}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}
class_names = image_datasets['train'].classes
num_classes = len(class_names)

print(f"训练集大小: {dataset_sizes['train']}")
print(f"验证集大小: {dataset_sizes['valid']}")
print(f"类别数: {num_classes}")
print(f"类别: {class_names}")


def resnet_custom(num_classes, model_type='resnet50'):
    if model_type == 'resnet18':
        return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)
    elif model_type == 'resnet34':
        return ResNet(BasicBlock, [3, 4, 6, 3], num_classes)
    elif model_type == 'resnet50':
        return ResNet(Bottleneck, [3, 4, 6, 3], num_classes)
    elif model_type == 'resnet101':
        return ResNet(Bottleneck, [3, 4, 23, 3], num_classes)

# 替换加载模型部分
print(f"加载自定义 {args.model_type} 模型...")
model = resnet_custom(num_classes, args.model_type)
model = model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)
# 学习率调度器
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

# 训练模型
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    best_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # 每个epoch有训练和验证阶段
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # 训练模式
            else:
                model.eval()   # 评估模式
                
            running_loss = 0.0
            running_corrects = 0
            
            # 遍历数据
            for inputs, labels in tqdm(dataloaders[phase], desc=phase):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # 零参数梯度
                optimizer.zero_grad()
                
                # 前向传播
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    # 如果是训练阶段，反向传播+优化
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # 统计
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # 保存历史指标
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())
                # 调整学习率
                scheduler.step(epoch_loss)
                
                # 保存最佳模型
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), os.path.join(args.output_dir, f'best_model_{args.model_type}.pth'))
                    print(f'保存最佳模型，精度: {best_acc:.4f}')
        
        print()
    
    # 保存最终模型
    torch.save(model.state_dict(), os.path.join(args.output_dir, f'final_model_{args.model_type}.pth'))
    print(f'完成训练, 最佳验证精度: {best_acc:.4f}')
    
    # 绘制训练过程
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='train')
    plt.plot(history['val_loss'], label='val')
    plt.title('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='train')
    plt.plot(history['val_acc'], label='val')
    plt.title('Accuracy')
    plt.legend()
    
    plt.savefig(os.path.join(args.output_dir, 'training_history.png'))
    plt.show()
    
    return model, history

# 开始训练
print("开始训练...")
model, history = train_model(model, criterion, optimizer, scheduler, num_epochs=args.epochs)

# 保存类别映射
import json
class_to_idx = image_datasets['train'].class_to_idx
with open(os.path.join(args.output_dir, 'class_mapping.json'), 'w') as f:
    json.dump(class_to_idx, f)