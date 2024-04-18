import torch
import torchvision
from torch import classes
from torchvision import datasets, transforms

# 定义数据预处理操作
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # 数据增强：随机翻转图片
    transforms.RandomCrop(32, padding=4),  # 数据增强：随机裁剪图片
    transforms.ToTensor(),  # 将PIL.Image或者numpy.ndarray数据类型转化为torch.FloadTensor，并归一化到[0.0, 1.0]
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # 标准化（这里的均值和标准差是CIFAR10数据集的）
])

# 下载并加载训练数据集
trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)

# 下载并加载测试数据集
testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)

import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)  # 输入通道数3，输出通道数6，卷积核大小5
        self.pool = nn.MaxPool2d(2, 2)  # 最大池化，核大小2，步长2
        self.conv2 = nn.Conv2d(6, 16, 5)  # 输入通道数6，输出通道数16，卷积核大小5
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 全连接层，输入维度16*5*5，输出维度120
        self.fc2 = nn.Linear(120, 84)  # 全连接层，输入维度120，输出维度84
        self.fc3 = nn.Linear(84, 10)  # 全连接层，输入维度84，输出维度10（CIFAR10有10类）

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 第一层卷积+ReLU激活函数+池化
        x = self.pool(F.relu(self.conv2(x)))  # 第二层卷积+ReLU激活函数+池化
        x = x.view(-1, 16 * 5 * 5)  # 将特征图展平
        x = F.relu(self.fc1(x))  # 第一层全连接+ReLU激活函数
        x = F.relu(self.fc2(x))  # 第二层全连接+ReLU激活函数
        x = self.fc3(x)  # 第三层全连接
        return x

# 创建网络
net = Net().to(device="cuda:0")

import torch.optim as optim

# 定义损失函数
criterion = nn.CrossEntropyLoss()

# 定义优化器
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(2):  # 在数据集上训练两遍

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # 获取输入数据
        inputs, labels = data

        # 梯度清零
        optimizer.zero_grad()

        # 前向传播
        outputs = net(inputs)

        # 计算损失
        loss = criterion(outputs, labels)

        # 反向传播
        loss.backward()

        # 更新参数
        optimizer.step()

        # 打印统计信息
        running_loss += loss.item()
        if i % 2000 == 1999:  # 每2000个批次打印一次
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

# 加载一些测试图片
dataiter = iter(testloader)
images, labels = dataiter.next()

# # 打印图片
# show(torchvision.utils.make_grid(images))

# 显示真实的标签
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

# 让网络做出预测
outputs = net(images)

# 预测的标签是最大输出的标签
_, predicted = torch.max(outputs, 1)

# 显示预测的标签
print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))

# 在整个测试集上测试网络
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))

# 保存模型
torch.save(net.state_dict(), './save/models/cifar_net.pth')