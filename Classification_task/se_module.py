from torch import nn


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),

            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):

        b, c, _, _ = x.size()####batchsize, chanels

        y1 = self.avg_pool(x).view(b, c)###[b,c]
        y1 = self.fc(y1).view(b, c, 1, 1)###x的权重矩阵

        return x * y1.expand_as(x)
