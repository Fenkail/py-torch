from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision as tv
import torchvision.transforms as transforms
from PIL import Image as im
import time
class Net(nn.Module):
    def __init__(self):
        # 必须在构造函数中执行父类的构造函数
        # 等价于 nn.Module.__init__(self)
        super(Net,self).__init__()
        # self, in_channels, out_channels, kernel_size, stride=1,
        #              padding=0, dilation=1, groups=1, bias=True
        self.conv1 = nn.Conv2d(3,6,5)
        self.conv2 = nn.Conv2d(6,16,5)
        # 全连接层 self, in_features, out_features, bias=True
        self.fc1 = nn.Linear(16*5*5,120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self,x):
        #  卷积 ==》 激活 ==》 池化
        x = F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)),2)
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x= F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()
print(net)
# 第一次运行程序torchvision会自动下载CIFAR-10数据集，
# 大约100M，需花费一定的时间，
# 如果已经下载有CIFAR-10，可通过root参数指定

# 定义对数据的预处理
transform = transforms.Compose([
        transforms.ToTensor(), # 转为Tensor
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # 归一化
                             ])

# 训练集
trainset = tv.datasets.CIFAR10(
                    root='./CIFAR10',
                    train=True,
                    download=True,
                    transform=transform)
# Dataloader 返回的每一条数据拼接为batch,提供多线程优化和数据打乱
trainloader = torch.utils.data.DataLoader(
                    trainset,
                    batch_size=4,
                    shuffle=True,
                    num_workers=2)

# 测试集
testset = tv.datasets.CIFAR10(
                    './CIFAR10',
                    train=False,
                    download=True,
                    transform=transform)

testloader = torch.utils.data.DataLoader(
                    testset,
                    batch_size=4,
                    shuffle=False,
                    num_workers=2)
# tuple
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# iter()  构造一个迭代函数   xxxxx.next() 下一个内容
dataiter = iter(trainloader)
images,labels = dataiter.next()

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

start = time.time()


# 训练过程
for epoch in range(2):
    running_loss = 0.0
    # >> > list(enumerate(seasons, start=1))  # 下标从 1 开始
    # [(1, 'Spring'), (2, 'Summer'), (3, 'Fall'), (4, 'Winter')]
    for i, data in enumerate(trainloader,0):
        # 输入数据
        inputs, labels = data
        inputs, labels = Variable(inputs), Variable(labels)
        # 梯度清零
        optimizer.zero_grad()

        # GPU

        labels = labels.cuda()

        output = net(inputs.cuda())
        loss = criterion(output, labels)
        loss.backward()
        # 更新参数
        optimizer.step()
        # 打印log信息
        running_loss += loss.data[0]
        if i % 2000 == 1999:
            print('[%d, %5d] loss: %.3f  '%(epoch+1, i+1, running_loss/2000))
            running_loss = 0
end = time.time()
correct = 0
total = 0
for data in  testloader:
    images ,labels = data
    output = net(Variable(images))
    _, predicted = torch.max(output.data , 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()

print('10000测试集 准确率为： %d %%' %(100*correct/total))
print('10000测试集 时间为： %.3f s' %(end-start))


# ----------------------CPU------------------------------------
# [1,  2000] loss: 2.200
# [1,  4000] loss: 1.847
# [1,  6000] loss: 1.698
# [1,  8000] loss: 1.587
# [1, 10000] loss: 1.533
# [1, 12000] loss: 1.460
# [2,  2000] loss: 1.391
# [2,  4000] loss: 1.365
# [2,  6000] loss: 1.352
# [2,  8000] loss: 1.322
# [2, 10000] loss: 1.326
# [2, 12000] loss: 1.303
# 10000测试集 准确率为： 54 %
# 10000测试集时间为： 130.51539 s