# -*- coding: utf-8 -*-

from torch import nn

class RLV(nn.Module):
    def __init__(self, channel, reduction=16):
        super(RLV, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, kernel_size = 1, bias=False),
            nn.BatchNorm2d(channel // reduction),
            nn.ReLU(inplace = True),
            nn.Conv2d(channel // reduction, channel, kernel_size = 1, bias=False),
            nn.BatchNorm2d(channel),
            nn.Sigmoid()
        )
        self.relu  = nn.ReLU(inplace = True)
    def forward(self, x1, x2):

        b, c, _, _ = x2.size()  # batchsize, chanels

        y1 = self.avg_pool(x1)
        y2 = self.avg_pool(x2)
        y3 = y1 * 0.3 + y2 * 0.7
        y3 = self.relu(y3)

        y = self.fc(y3).view(b, c, 1, 1)  
        out = x2 * y.expand_as(x2)

        return out
