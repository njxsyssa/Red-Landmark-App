import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import os

# 1. 配置参数
DATA_DIR = './dataset'   # 对应你截图中的文件夹路径
MODEL_SAVE_PATH = 'red_landmark_model.pth'
BATCH_SIZE = 32
NUM_EPOCHS = 10         # 训练轮数，可根据时间调整，建议至少10轮
LEARNING_RATE = 0.001

def train_model():
    # 2. 数据增强与预处理 [cite: 20, 21]
    # PPT提到：随机旋转，透视变换，高斯噪声(这里用ColorJitter模拟光照差异)
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),       # 尺寸归一化 [cite: 20]
        transforms.RandomHorizontalFlip(),   # 随机水平翻转
        transforms.RandomRotation(15),       # 随机旋转，对应PPT中的角度适应 
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), # 模拟光照变化
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 3. 加载数据集
    # ImageFolder会自动处理 dataset/类名/子文件夹/图片.jpg 的结构
    try:
        full_dataset = datasets.ImageFolder(root=DATA_DIR, transform=train_transforms)
    except FileNotFoundError:
        print(f"错误：找不到文件夹 {DATA_DIR}，请确保路径正确。")
        return

    # 获取类别名称映射 (例如: 0 -> 0_gutian_meeting)
    class_names = full_dataset.classes
    print(f"检测到的类别: {class_names}")

    # 划分训练集和验证集 (8:2)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # 验证集应用不带增强的变换
    val_dataset.dataset.transform = val_transforms

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 4. 定义模型 (使用ResNet18，适合轻量级任务)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"正在使用设备: {device}")
    
    model = models.resnet18(pretrained=True) # 使用预训练权重加速收敛
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names)) # 修改最后一层以匹配你的分类数量(14类)
    model = model.to(device)

    # 5. 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 6. 开始训练
    print("开始训练...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {running_loss/len(train_loader):.4f}")

    # 7. 保存模型
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    # 保存类别名称以便Web端使用
    with open("class_names.txt", "w") as f:
        for name in class_names:
            f.write(name + "\n")
            
    print(f"训练完成！模型已保存为 {MODEL_SAVE_PATH}")

if __name__ == '__main__':
    train_model()
