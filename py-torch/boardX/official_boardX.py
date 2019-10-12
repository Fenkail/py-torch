import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from tensorboardX import SummaryWriter

input = torch.rand([10,10,3])
models = torchvision.models.resnet18()
models.fc = nn.Linear(512,10)

# SummaryWriter().add_graph(models, input)