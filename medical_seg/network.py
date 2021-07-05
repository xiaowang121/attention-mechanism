import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from SENet import SELayer
from RLV_net import RLV
from RLM_block import RLM
from CBAM_nett import CBAM

def init_weights(net, init_type = 'normal', gain = 0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain = gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a = 0, mode = 'fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain = gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size = 3, stride = 1, padding = 1, bias = True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace = True),
            nn.Conv2d(ch_out, ch_out, kernel_size = 3, stride = 1, padding = 1, bias = True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace = True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class SE_Block(nn.Module):  ####residualblock

    def __init__(self, ch_in, ch_out, stride = 1):
        super(SE_Block, self).__init__()
        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size = 3, stride = stride, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(ch_out)

        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size = 3, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(ch_out)

        self.se = SELayer(ch_out)

    def forward(self, x):
        out1 = F.relu(self.bn1(self.conv1(x)))

        out2 = self.bn2(self.conv2(out1))
        out = self.se(out2)

        out = F.relu(out)
        return out

class CBAM_Block(nn.Module):  ####residualblock

    def __init__(self, ch_in, ch_out, stride = 1):
        super(CBAM_Block, self).__init__()
        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size = 3, stride = stride, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(ch_out)

        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size = 3, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(ch_out)

        self.CBAM = CBAM(ch_out)

    def forward(self, x):
        out1 = F.relu(self.bn1(self.conv1(x)))

        out2 = self.bn2(self.conv2(out1))
        out = self.CBAM(out2)

        out = F.relu(out)
        return out


class RLV_Block(nn.Module):  ####residualblock

    def __init__(self, ch_in, ch_out, stride = 1):
        super(RLV_Block, self).__init__()
        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size = 3, stride = stride, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(ch_out)

        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size = 3, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(ch_out)

        self.RLV = RLV(ch_out)

    def forward(self, x):
        out1 = F.relu(self.bn1(self.conv1(x)))

        out2 = self.bn2(self.conv2(out1))
        out = self.RLV(out1, out2)

        out = F.relu(out)
        return out


class RLM_Block(nn.Module):  ####residualblock

    def __init__(self, ch_in, ch_out, stride = 1):
        super(RLM_Block, self).__init__()
        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size = 3, stride = stride, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(ch_out)

        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size = 3, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(ch_out)

        self.RLM = RLM(ch_out)

    def forward(self, x):
        out1 = F.relu(self.bn1(self.conv1(x)))

        out2 = self.bn2(self.conv2(out1))
        out = self.RLM(out1, out2)

        out = F.relu(out)
        return out

class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor = 2),
            nn.Conv2d(ch_in, ch_out, kernel_size = 3, stride = 1, padding = 1, bias = True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace = True)
        )

    def forward(self, x):
        x = self.up(x)
        return x

class single_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(single_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size = 3, stride = 1, padding = 1, bias = True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace = True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class U_Net(nn.Module):
    def __init__(self, img_ch = 3, output_ch = 1):
        super(U_Net, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.Conv1 = conv_block(ch_in = img_ch, ch_out = 32)
        self.Conv2 = conv_block(ch_in = 32, ch_out = 64)
        self.Conv3 = conv_block(ch_in = 64, ch_out = 128)
        self.Conv4 = conv_block(ch_in = 128, ch_out = 256)
        self.Conv5 = conv_block(ch_in=256,ch_out=512)

        self.Up5 = up_conv(ch_in=512,ch_out=256)
        self.Up_conv5 = conv_block(ch_in=512, ch_out=256)

        self.Up4 = up_conv(ch_in = 256, ch_out = 128)
        self.Up_conv4 = conv_block(ch_in = 256, ch_out = 128)

        self.Up3 = up_conv(ch_in = 128, ch_out = 64)
        self.Up_conv3 = conv_block(ch_in = 128, ch_out = 64)

        self.Up2 = up_conv(ch_in = 64, ch_out = 32)
        self.Up_conv2 = conv_block(ch_in = 64, ch_out = 32)

        self.Conv_1x1 = nn.Conv2d(32, output_ch, kernel_size = 1, stride = 1, padding = 0)


    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4,d5),dim=1)

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim = 1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim = 1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim = 1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1

class UNet_cbam(nn.Module):
    def __init__(self, img_ch = 3, output_ch = 1):
        super(UNet_cbam, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.Conv1 = CBAM_Block(ch_in = img_ch, ch_out = 32)
        self.Conv2 = CBAM_Block(ch_in = 32, ch_out = 64)
        self.Conv3 = CBAM_Block(ch_in = 64, ch_out = 128)
        self.Conv4 = CBAM_Block(ch_in = 128, ch_out = 256)
        self.Conv5 = CBAM_Block(ch_in=256,ch_out=512)

        self.Up5 = up_conv(ch_in=512,ch_out=256)
        self.Up_conv5 = CBAM_Block(ch_in=512, ch_out=256)

        self.Up4 = up_conv(ch_in = 256, ch_out = 128)
        self.Up_conv4 = CBAM_Block(ch_in = 256, ch_out = 128)

        self.Up3 = up_conv(ch_in = 128, ch_out = 64)
        self.Up_conv3 = CBAM_Block(ch_in = 128, ch_out = 64)

        self.Up2 = up_conv(ch_in = 64, ch_out = 32)
        self.Up_conv2 = CBAM_Block(ch_in = 64, ch_out = 32)

        self.Conv_1x1 = nn.Conv2d(32, output_ch, kernel_size = 1, stride = 1, padding = 0)


    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4,d5),dim=1)

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim = 1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim = 1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim = 1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1

class UNet_se(nn.Module):
    def __init__(self, img_ch = 3, output_ch = 1):
        super(UNet_se, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.Conv1 = SE_Block(ch_in = img_ch, ch_out = 32)
        self.Conv2 = SE_Block(ch_in = 32, ch_out = 64)
        self.Conv3 = SE_Block(ch_in = 64, ch_out = 128)
        self.Conv4 = SE_Block(ch_in = 128, ch_out = 256)
        self.Conv5 = SE_Block(ch_in=256,ch_out=512)

        self.Up5 = up_conv(ch_in=512,ch_out=256)
        self.Up_conv5 = SE_Block(ch_in=512, ch_out=256)

        self.Up4 = up_conv(ch_in = 256, ch_out = 128)
        self.Up_conv4 = SE_Block(ch_in = 256, ch_out = 128)

        self.Up3 = up_conv(ch_in = 128, ch_out = 64)
        self.Up_conv3 = SE_Block(ch_in = 128, ch_out = 64)

        self.Up2 = up_conv(ch_in = 64, ch_out = 32)
        self.Up_conv2 = SE_Block(ch_in = 64, ch_out = 32)

        self.Conv_1x1 = nn.Conv2d(32, output_ch, kernel_size = 1, stride = 1, padding = 0)

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4,d5),dim=1)

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim = 1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim = 1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim = 1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1


class UNet_RLV(nn.Module):
    def __init__(self, img_ch = 3, output_ch = 1):
        super(UNet_RLV, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.Conv1 = RLV_Block(ch_in = img_ch, ch_out = 32)
        self.Conv2 = RLV_Block(ch_in = 32, ch_out = 64)
        self.Conv3 = RLV_Block(ch_in = 64, ch_out = 128)
        self.Conv4 = RLV_Block(ch_in = 128, ch_out = 256)
        self.Conv5 = RLV_Block(ch_in=256,ch_out=512)

        self.Up5 = up_conv(ch_in=512,ch_out=256)
        self.Up_conv5 = RLV_Block(ch_in=512, ch_out=256)

        self.Up4 = up_conv(ch_in = 256, ch_out = 128)
        self.Up_conv4 = RLV_Block(ch_in = 256, ch_out = 128)

        self.Up3 = up_conv(ch_in = 128, ch_out = 64)
        self.Up_conv3 = RLV_Block(ch_in = 128, ch_out = 64)

        self.Up2 = up_conv(ch_in = 64, ch_out = 32)
        self.Up_conv2 = RLV_Block(ch_in = 64, ch_out = 32)

        self.Conv_1x1 = nn.Conv2d(32, output_ch, kernel_size = 1, stride = 1, padding = 0)

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4,d5),dim=1)

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim = 1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim = 1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim = 1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1


class UNet_RLM(nn.Module):
    def __init__(self, img_ch = 3, output_ch = 1):
        super(UNet_RLM, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.Conv1 = RLM_Block(ch_in = img_ch, ch_out = 32)
        self.Conv2 = RLM_Block(ch_in = 32, ch_out = 64)
        self.Conv3 = RLM_Block(ch_in = 64, ch_out = 128)
        self.Conv4 = RLM_Block(ch_in = 128, ch_out = 256)
        self.Conv5 = RLM_Block(ch_in=256,ch_out=512)

        self.Up5 = up_conv(ch_in=512,ch_out=256)
        self.Up_conv5 = RLM_Block(ch_in=512, ch_out=256)

        self.Up4 = up_conv(ch_in = 256, ch_out = 128)
        self.Up_conv4 = RLM_Block(ch_in = 256, ch_out = 128)

        self.Up3 = up_conv(ch_in = 128, ch_out = 64)
        self.Up_conv3 = RLM_Block(ch_in = 128, ch_out = 64)

        self.Up2 = up_conv(ch_in = 64, ch_out = 32)
        self.Up_conv2 = RLM_Block(ch_in = 64, ch_out = 32)

        self.Conv_1x1 = nn.Conv2d(32, output_ch, kernel_size = 1, stride = 1, padding = 0)

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4,d5),dim=1)

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim = 1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim = 1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim = 1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1
