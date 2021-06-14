# -*- coding: utf-8 -*-

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from se_module import SELayer
from RLM_net import RLM
from RLV_net import RLV
from Non_Local import NL_local
import torch


def _weights_init(m):
    # classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):  # 进行填充
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

class init_block(nn.Module):  ####residualblock
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(init_block, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias = False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()


        if stride != 1 or in_planes != planes:
            self.shortcut = LambdaLayer(lambda x:
                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
    def forward(self, x):

        out1 = F.relu(self.bn1(self.conv1(x)))

        out = self.bn2(self.conv2(out1))

        out += self.shortcut(x)

        out = F.relu(out)
        return out

class se_block(nn.Module):  ####residualblock
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(se_block, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias = False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        self.se = SELayer(planes)

        if stride != 1 or in_planes != planes:
            self.shortcut = LambdaLayer(lambda x:
                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
    def forward(self, x):

        out1 = F.relu(self.bn1(self.conv1(x)))

        out2 = self.bn2(self.conv2(out1))
        out = self.se(out2)
        out += self.shortcut(x)

        out = F.relu(out)
        return out

class NL_block(nn.Module):  ####residualblock
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(NL_block, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias = False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        self.NL = NL_local(planes)

        if stride != 1 or in_planes != planes:
            self.shortcut = LambdaLayer(lambda x:
                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
    def forward(self, x):

        out1 = F.relu(self.bn1(self.conv1(x)))

        out2 = self.bn2(self.conv2(out1))
        out = self.NL(out2)
        out += self.shortcut(x)

        out = F.relu(out)
        return out

class RLV_block(nn.Module):  ####residualblock
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(RLV_block, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias = False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        self.RLV = RLV(planes)

        if stride != 1 or in_planes != planes:
            self.shortcut = LambdaLayer(lambda x:
                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
    def forward(self, x):

        out1 = F.relu(self.bn1(self.conv1(x)))

        out2 = self.bn2(self.conv2(out1))
        out = self.RLV(out1, out2)
        out += self.shortcut(x)

        out = F.relu(out)
        return out

class RLM_block(nn.Module):  ####residualblock
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(RLM_block, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias = False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        self.RLM = RLM(planes)

        if stride != 1 or in_planes != planes:
            self.shortcut = LambdaLayer(lambda x:
                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
    def forward(self, x):

        out1 = F.relu(self.bn1(self.conv1(x)))

        out2 = self.bn2(self.conv2(out1))
        out = self.RLM(out1, out2)
        out += self.shortcut(x)

        out = F.relu(out)
        return out

class net_resnet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(net_resnet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)  ####list
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)  ###list
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)  ###list

        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.ModuleList(layers)
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))

        for i in range(len(self.layer1)):
            out = self.layer1[i](out)
        for j in range(len(self.layer2)):
            out = self.layer2[j](out)
        for k in range(len(self.layer3)):
            out = self.layer3[k](out)

        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out
###########
def init_resnet20():
    return net_resnet(init_block, [3, 3, 3])

def init_resnet32():
    return net_resnet(init_block, [5, 5, 5])

def init_resnet44():
    return net_resnet(init_block, [7, 7, 7])

def init_resnet56():
    return net_resnet(init_block, [9, 9, 9])

def init_resnet110():
    return net_resnet(init_block, [18, 18, 18])

###########
def se_resnet20():
    return net_resnet(se_block, [3, 3, 3])

def se_resnet32():
    return net_resnet(se_block, [5, 5, 5])

def se_resnet44():
    return net_resnet(se_block, [7, 7, 7])

def se_resnet56():
    return net_resnet(se_block, [9, 9, 9])

def se_resnet110():
    return net_resnet(se_block, [18, 18, 18])

###########
def nl_resnet20():
    return net_resnet(NL_block, [3, 3, 3])

def nl_resnet32():
    return net_resnet(NL_block, [5, 5, 5])

def nl_resnet44():
    return net_resnet(NL_block, [7, 7, 7])

def nl_resnet56():
    return net_resnet(NL_block, [9, 9, 9])

def nl_resnet110():
    return net_resnet(NL_block, [18, 18, 18])

###########
def rlv_resnet20():
    return net_resnet(RLV_block, [3, 3, 3])

def rlv_resnet32():
    return net_resnet(RLV_block, [5, 5, 5])

def rlv_resnet44():
    return net_resnet(RLV_block, [7, 7, 7])

def rlv_resnet56():
    return net_resnet(RLV_block, [9, 9, 9])

def rlv_resnet110():
    return net_resnet(RLV_block, [18, 18, 18])

###########
def rlm_resnet20():
    return net_resnet(RLM_block, [3, 3, 3])

def rlm_resnet32():
    return net_resnet(RLM_block, [5, 5, 5])

def rlm_resnet44():
    return net_resnet(RLM_block, [7, 7, 7])

def rlm_resnet56():
    return net_resnet(RLM_block, [9, 9, 9])

def rlm_resnet110():
    return net_resnet(RLM_block, [18, 18, 18])