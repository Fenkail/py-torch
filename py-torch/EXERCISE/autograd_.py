from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Variable 和 Tensor 近乎一样的接口
# 定义网络需要继承nn.Module 并且实现forward方法
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
params = list(net.parameters())
print(len(params))
# 需要将Tensor 封装成Variable 才能自动求导
# torh 的随机  rand [0,1]均匀随机   randn标准正太随机  normal 离散正太随机

input = Variable(torch.randn(1,1,32,32))
out = net(input)
# 从 10转到 【1,10】
target = torch.arange(0,10).view(1,10)
# 均方误差
criterion = nn.MSELoss()
loss = criterion(out,target)
# 定义优化器
optimizer = optim.SGD(net.parameters(),lr=0.01)
optimizer.zero_grad()
loss.backward()
optimizer.step()
