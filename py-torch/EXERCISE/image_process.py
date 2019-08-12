from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision as tv
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image
from torchvision.utils import make_grid, save_image

# 2019年08月02日20:53:49
transform = transforms.Compose([
                transforms.Resize(640),
                transforms.CenterCrop(640),
                transforms.ToTensor(),
                transforms.Normalize(mean=[.5,.5,.5],std=[.5,.5,.5])
])
data = Image.open('/home/kai/图片/person.jpg')
data_new = transform(data)
# PIL的处理办法
# data_new = data.crop((0,0,320,320))
# data_new = transforms.ToPILImage(data_new)

# 工具包的使用  画网格分割
# tv.utils.make_grid()
# 利用Tensor保存图像
tv.utils.save_image(data_new,'1.png')



