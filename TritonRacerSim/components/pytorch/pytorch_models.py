import torch
import torch.nn as nn
from torch.nn.modules.activation import ReLU
from torch.nn.modules.linear import Linear
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import numpy as np

''' A baseline CNN regression model. Spits out steering and throttle (or speed).'''
def get_baseline_cnn(num_channel__in=3, num_out=2):
    return nn.Sequential(
        nn.Conv2d(num_channel__in, 64, 5), nn.ReLU(),
        nn.Conv2d(64, 64, 3), nn.ReLU(),
        nn.Conv2d(64, 64, 3), nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.GroupNorm(8, 64),

        nn.Conv2d(64, 128, 3), nn.ReLU(),
        nn.Conv2d(128, 128, 3), nn.ReLU(),
        nn.Conv2d(128, 128, 3), nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.GroupNorm(16, 128),

        nn.Flatten(),
        nn.Linear(4096, 2048), nn.ReLU(),
        nn.Linear(2048, 1024), nn.ReLU(),
        nn.Linear(1024, 512),
        nn.Linear(1024, 256), 
        nn.Linear(256, num_out), 
    )