import time
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter

class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        # 输入通道  输出通道
        self.conv1 = nn.Conv2d(1, 10 ,kernel_size =5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size =5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 2)
        self.bn = nn.BatchNorm2d(20)


    def forward(self, x):
        # 最大池化   2为尺寸
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.relu(x) + F.relu(-x)
        x = F.relu(F.max_pool2d(F.relu(self.conv2_drop(self.conv2(x))), 2))
        x = self.bn(x)
        # 以每列320个数进行查看
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training = self.training)
        x = self.fc2(x)
        x = F.softmax(x, dim = 1 )
        return x


def train():
    start = time.time()
    writer = SummaryWriter(comment = 'Net1')

    # dummy_input = torch.randn(13, 1, 28, 28)
    dummy_input = torch.load('input.tensor').cuda()
    dummy_output = torch.Tensor([1,0,0,0,0,0,0,0,0,0,0,0,1]).type(torch.LongTensor).cuda()

    model = Net1().cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.001)

    for epoch in tqdm(range(20000)):
        inputs = dummy_input
        targets = dummy_output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        writer.add_scalar('Train', loss, epoch)


    writer.add_graph(model,inputs)


    torch.save(model, 'diy_gpu.pth')
    torch.save(model.state_dict(), "diy_par_gpu.pth")
    end = time.time()

    print(" 本次训练花费时间： %.2f s" %(end-start))


def test(flag = False):
    x = torch.load('input.tensor').cuda()
    model = torch.load('diy_gpu.pth')
    prediction = model(x)

    print(prediction.shape)
    print( prediction)



if __name__ == '__main__':

    # train()

    test()
    # https://www.jianshu.com/p/46eb3004beca