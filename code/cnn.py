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
        self.conv1 = nn.Conv1d(1, 2, 3, 1, 1)
        self.leaky_relu1 = nn.ReLU()
        self.avg_pool1 = nn.AvgPool1d(2, 2)
        self.conv2 = nn.Conv1d(2, 4, 3, 1, 1)
        self.leaky_relu2 = nn.ReLU()
        self.conv3 = nn.Conv1d(4, 8, 3, 1, 1)
        self.leaky_relu3 = nn.ReLU()
        self.conv4 = nn.Conv1d(8, 16, 3, 1, 1)
        self.leaky_relu4 = nn.ReLU()

        self.unravel = Flatten()
        self.fc1 = nn.Linear(112, 64)
        self.fc2 = nn.Linear(64, 2)




    def forward(self, x):
        x = self.conv1(x)
        x = self.leaky_relu1(x)
        x = self.conv2(x)
        x = self.leaky_relu2(x)
        x = self.avg_pool1(x)
        x = self.conv3(x)
        x = self.leaky_relu3(x)
        x = self.conv4(x)
        x = self.leaky_relu4(x)
        x = self.unravel(x)
        x = self.fc1(x)
        out = self.fc2(x)
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