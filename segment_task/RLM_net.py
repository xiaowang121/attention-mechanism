# -*- coding: utf-8 -*-

from torch import nn
import torch
class RLM(nn.Module):
    def __init__(self, channel, reduction=16):
        super(RLM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size = 1, bias=False),
            nn.ReLU(inplace=True),

            nn.Conv2d(1, 1, kernel_size = 1,  bias=False),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AvgPool2d((1, channel), stride = (1, 1))
    def forward(self, x1, x2):

        b, c, _, _ = x2.size()  # batchsize, chanels
        y1 = self.avg_pool(x1).view(b,c,-1)
        y2 = self.avg_pool(x2).view(b,c,-1)


        y22 = y2.permute(0, 2, 1)
        Y = torch.matmul(y1, y22)
        Y = self.relu(Y)
        Y = Y.view(b,1,c,c)

        y = self.fc(Y)  
        y = self.pool(y)
        y = y.permute(0,2,1,3)

        out = x2 * y.expand_as(x2)

        return out

