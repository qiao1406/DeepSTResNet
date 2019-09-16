import torch
import torch.nn.functional as func
from torch import nn


class ResUnit(nn.Module):
    """
    Defines a residual unit
    x[l+1] = x[l] + (x{l]-> relu -> conv1 -> relu -> conv1)
    """

    def __init__(self):
        super(ResUnit, self).__init__()
        self.conv1 = nn.Conv2d(64, 64, 3)
        self.conv2 = nn.Conv2d(64, 64, 3)

    # x channel num: 64
    def forward(self, x):
        f = self.conv1(func.relu(x))
        f = self.conv2(func.relu(f))
        x = x + f

        return x


class STResNet(nn.Module):

    def __init__(self, res_layers, interval):
        super(STResNet, self).__init__()
        self.res_layers = res_layers

        self.conv1 = nn.Conv2d(2 * interval, 64, 3)
        self.res_units = nn.ModuleList([ResUnit() for _ in range(res_layers)])
        self.conv2 = nn.Conv2d(64, 2, 3)

    def forward(self, x):

        # conv layer 1
        x = self.conv1(x)

        # L layers ResUnit
        for l in range(self.res_layers):
            x = self.res_units[l](x)

        # conv layer 2
        x = self.conv2(x)

        return x


class RMSELoss(nn.Module):

    def __init__(self):
        super(RMSELoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, y_hat, y):
        return torch.sqrt(self.mse(y_hat, y))
