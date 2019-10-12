import numpy as np
import torch

import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter





x = torch.load('input.tensor')


model = torch.load('diy.pth')
prediction = model(x)


