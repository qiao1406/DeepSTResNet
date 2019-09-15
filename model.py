import torch.nn.functional as F
from torch import nn


class STResNet(nn.Module):

    def __init__(self, res_layers, interval):
        super(STResNet, self).__init__()
        self.res_layers = res_layers
        self.conv1 = nn.Conv2d(2 * interval, 64, 3)
        self.conv2 = nn.Conv2d(64, 2, 3)

    def forward(self, x):

        # conv layer 1
        x = self.conv1(x)

        # L layers ResUnit
        for l in range(self.res_layers):
            f = self.conv1(F.relu(x))
            f = self.conv1(F.relu(f))
            x = x + f

        # conv layer 2
        x = self.conv2(x)

        return x

