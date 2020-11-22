import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
class Flatten(nn.Module):
        def forward(self, x):
            return x.view(x.shape[0], -1)

class cnn(nn.Module):
    def __init__(self):
        super(cnn, self).__init__()
        self.pad1 = nn.ConstantPad1d((0, 1), 0)
        self.conv1 = nn.Conv1d(1, 32, 3, 1, 1)
        self.leaky_relu1 = nn.ReLU()
        self.avg_pool1 = nn.AvgPool1d(2, 2)
        self.conv2 = nn.Conv1d(32, 64, 3, 1, 1)
        self.leaky_relu2 = nn.ReLU()
        self.avg_pool2 = nn.AvgPool1d(2, 2)
        self.conv3 = nn.Conv1d(64, 128, 3, 1, 1)
        self.leaky_relu3 = nn.ReLU()
        self.avg_pool3 = nn.AvgPool1d(2, 2)
        self.conv4 = nn.Conv1d(128, 128, 3, 1, 1)
        self.conv5 = nn.Conv1d(128, 128, 3, 1, 1)
        self.leaky_relu4 = nn.ReLU()
        self.avg_pool4 = nn.AvgPool1d(2, 2)

        self.unravel = Flatten()
        self.fc1 = nn.Linear(1920, 32)
        self.fc2 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()




    def forward(self, x):
        x = self.pad1(x)
        x = self.conv1(x)
        x = self.leaky_relu1(x)
        x = self.avg_pool1(x)
        x = self.conv2(x)
        x = self.leaky_relu2(x)
        x = self.avg_pool2(x)
        x = self.conv3(x)
        x = self.leaky_relu3(x)
        x = self.avg_pool3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.leaky_relu4(x)
        x = self.avg_pool4(x)
        x = self.unravel(x)
        x = self.fc1(x)
        x = self.fc2(x)
        out = self.sigmoid(x)
        return out

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.001)
                nn.init.constant_(m.bias, 0)





if __name__ == "__main__":
    pass