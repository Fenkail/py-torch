from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision as tv
import torchvision.transforms as transforms

# Tensor的读取与保存
a = torch.Tensor(3, 4 )
a.cuda()
torch.save(a,'a.pth')

b = torch.load('a.pth')

c = torch.load('a.pth',map_location=lambda sto,loc:sto)
# ----------------------------------------------------------
torch.set_default_tensor_type('torch.FloatTensor')
from torchvision.models import AlexNet

model = AlexNet()
model.state_dict().keys()
# model的保存与加载
torch.save(model.state_dict(), 'alexnet.pth')
model.load_state_dict(torch.load('alexnet.pth'))

opt = torch.optim.Adam(model.parameters(),lr=0.1)
# 优化器的参数读取与保存
torch.save(opt.state_dict(), 'opt.pth')
opt.load_state_dict(torch.load('opt.pth'))

