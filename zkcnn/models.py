import torch
import torch.nn as nn
from einops import rearrange

class LeNetCifar(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNetCifar, self).__init__()
        # Input: 3x32x32
        # Conv1: 3 -> 6, kernel 5
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.relu1 = nn.ReLU()
        # MaxPool1: 2x2
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # Conv2: 6 -> 16, kernel 5
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu2 = nn.ReLU()
        # MaxPool2: 2x2
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Conv3: 16 -> 120, kernel 5
        # Input to Conv3 is 16x5x5 (after two pools of 32->14->5)
        # Output is 120x1x1
        self.conv3 = nn.Conv2d(16, 120, 5)
        self.relu3 = nn.ReLU()
        
        # FC layers
        self.fc1 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.relu3(self.conv3(x))
        # Flatten C, H, W into a single dimension using einops
        x = rearrange(x, 'b c h w -> b (c h w)')
        x = self.relu4(self.fc1(x))
        x = self.fc2(x)
        return x
