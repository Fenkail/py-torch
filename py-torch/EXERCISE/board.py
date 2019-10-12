import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from boardX import SummaryWriter

writer = SummaryWriter()
for epoch in range(100):
    # 第一个参数为保存的名称  第二个是Y轴  第三个是X轴
    writer.add_scalar('aaa/11', np.random.rand(), epoch)
    writer.add_scalars('aaa/222', {'xsins': epoch * np.sin(epoch),
                                   "xconx": epoch * np.cos(epoch)}, epoch)

writer.close()