#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
   Keras Helper Notes
"""

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


# Tensor shape
tensor
a = torch.empty(5, 7, dtype=torch.float)

print(torch.__version__)
