import torchvision.models as models
import torch
from torch import nn
import torch.nn.functional as f
import torchvision
import numpy as np
from CBAM_net import CBAM
from RLM_net import RLM
from RLV_net import RLV
from SENet import SELayer

def bilinear_kernel(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                      dtype='float32')
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(np.array(weight))

model_root = r'./resnet34-333f7ec4.pth'   ### you need to download this model
pretrained_net = models.resnet34(pretrained=False)
pre = torch.load(model_root)
pretrained_net.load_state_dict(pre)

class fcn(nn.Module):
    def __init__(self, num_classes):
        super(fcn, self).__init__()

        self.stage1 = nn.Sequential(*list(pretrained_net.children())[:-4])
        self.stage2 = list(pretrained_net.children())[-4]
        self.stage3 = list(pretrained_net.children())[-3]
        self.scores1 = nn.Conv2d(512, num_classes, 1)
        self.scores2 = nn.Conv2d(256, num_classes, 1)
        self.scores3 = nn.Conv2d(128, num_classes, 1)

        self.upsample_8x = nn.ConvTranspose2d(num_classes, num_classes, kernel_size = 16, stride = 8, padding = 4,
                                              bias = False)
        self.upsample_8x.weight.data = bilinear_kernel(num_classes, num_classes, 16)
        self.upsample_4x = nn.ConvTranspose2d(num_classes, num_classes, kernel_size = 4, stride = 2, padding = 1,
                                              bias = False)
        self.upsample_4x.weight.data = bilinear_kernel(num_classes, num_classes, 4)
        self.upsample_2x = nn.ConvTranspose2d(num_classes, num_classes, kernel_size = 4, stride = 2, padding = 1,
                                              bias = False)
        self.upsample_2x.weight.data = bilinear_kernel(num_classes, num_classes, 4)

    def forward(self, x):
        x = self.stage1(x)
        s1 = x

        x = self.stage2(x)
        s2 = x

        x = self.stage3(x)
        s3 = x

        s3 = self.scores1(s3)
        s3 = self.upsample_2x(s3)
        s2 = self.scores2(s2)
        s2 = s2 + s3

        s1 = self.scores3(s1)
        s2 = self.upsample_4x(s2)
        s = s1 + s2

        s = self.upsample_8x(s)
        return s

class fcn_se(nn.Module):
    def __init__(self, num_classes):
        super(fcn_se, self).__init__()

        self.stage1 = nn.Sequential(*list(pretrained_net.children())[:-4])
        self.stage2 = list(pretrained_net.children())[-4]
        self.stage3 = list(pretrained_net.children())[-3]
        self.scores1 = nn.Conv2d(512, num_classes, 1)
        self.scores2 = nn.Conv2d(256, num_classes, 1)
        self.scores3 = nn.Conv2d(128, num_classes, 1)

        self.upsample_8x = nn.ConvTranspose2d(num_classes, num_classes, kernel_size = 16, stride = 8, padding = 4,
                                              bias = False)
        self.upsample_8x.weight.data = bilinear_kernel(num_classes, num_classes, 16)
        self.upsample_4x = nn.ConvTranspose2d(num_classes, num_classes, kernel_size = 4, stride = 2, padding = 1,
                                              bias = False)
        self.upsample_4x.weight.data = bilinear_kernel(num_classes, num_classes, 4)
        self.upsample_2x = nn.ConvTranspose2d(num_classes, num_classes, kernel_size = 4, stride = 2, padding = 1,
                                              bias = False)
        self.upsample_2x.weight.data = bilinear_kernel(num_classes, num_classes, 4)

        self.se = SELayer(num_classes)

    def forward(self, x):
        x = self.stage1(x)
        s1 = x

        x = self.stage2(x)
        s2 = x

        x = self.stage3(x)
        s3 = x

        s3 = self.scores1(s3)

        s3 = self.upsample_2x(s3)
        s3 = self.se(s3)

        s2 = self.scores2(s2)

        s2 = s2 + s3


        s1 = self.scores3(s1)
        s1 = self.se(s1)

        s2 = self.upsample_4x(s2)

        s = s1 + s2

        s = self.upsample_8x(s)


        return s


class fcn_cbam(nn.Module):
    def __init__(self, num_classes):
        super(fcn_cbam, self).__init__()

        self.stage1 = nn.Sequential(*list(pretrained_net.children())[:-4])
        self.stage2 = list(pretrained_net.children())[-4]
        self.stage3 = list(pretrained_net.children())[-3]
        self.scores1 = nn.Conv2d(512, num_classes, 1)
        self.scores2 = nn.Conv2d(256, num_classes, 1)
        self.scores3 = nn.Conv2d(128, num_classes, 1)

        self.upsample_8x = nn.ConvTranspose2d(num_classes, num_classes, kernel_size = 16, stride = 8, padding = 4,
                                              bias = False)
        self.upsample_8x.weight.data = bilinear_kernel(num_classes, num_classes, 16)
        self.upsample_4x = nn.ConvTranspose2d(num_classes, num_classes, kernel_size = 4, stride = 2, padding = 1,
                                              bias = False)
        self.upsample_4x.weight.data = bilinear_kernel(num_classes, num_classes, 4)
        self.upsample_2x = nn.ConvTranspose2d(num_classes, num_classes, kernel_size = 4, stride = 2, padding = 1,
                                              bias = False)
        self.upsample_2x.weight.data = bilinear_kernel(num_classes, num_classes, 4)

        self.cbam = CBAM(num_classes)

    def forward(self, x):
        x = self.stage1(x)
        s1 = x

        x = self.stage2(x)
        s2 = x

        x = self.stage3(x)
        s3 = x

        s3 = self.scores1(s3)

        s3 = self.upsample_2x(s3)
        s3 = self.cbam(s3)

        s2 = self.scores2(s2)

        s2 = s2 + s3

        s1 = self.scores3(s1)
        s1 = self.cbam(s1)
        s2 = self.upsample_4x(s2)
        s = s1 + s2

        s = self.upsample_8x(s)

        return s

class fcn_RLV(nn.Module):
    def __init__(self, num_classes):
        super(fcn_RLV, self).__init__()

        self.stage1 = nn.Sequential(*list(pretrained_net.children())[:-4])
        self.stage2 = list(pretrained_net.children())[-4]
        self.stage3 = list(pretrained_net.children())[-3]
        self.scores1 = nn.Conv2d(512, num_classes, 1)
        self.scores2 = nn.Conv2d(256, num_classes, 1)
        self.scores3 = nn.Conv2d(128, num_classes, 1)

        self.upsample_8x = nn.ConvTranspose2d(num_classes, num_classes, kernel_size = 16, stride = 8, padding = 4,
                                              bias = False)
        self.upsample_8x.weight.data = bilinear_kernel(num_classes, num_classes, 16)
        self.upsample_4x = nn.ConvTranspose2d(num_classes, num_classes, kernel_size = 4, stride = 2, padding = 1,
                                              bias = False)
        self.upsample_4x.weight.data = bilinear_kernel(num_classes, num_classes, 4)
        self.upsample_2x = nn.ConvTranspose2d(num_classes, num_classes, kernel_size = 4, stride = 2, padding = 1,
                                              bias = False)
        self.upsample_2x.weight.data = bilinear_kernel(num_classes, num_classes, 4)

        self.RLV = RLV(num_classes)

    def forward(self, x):
        x = self.stage1(x)
        s1 = x

        x = self.stage2(x)
        s2 = x

        x = self.stage3(x)
        s3 = x

        s3 = self.scores1(s3)
        s33 = self.upsample_2x(s3)
        s3 = self.RLV(s3, s33)

        s22 = self.scores2(s2)

        s2 = s22 + s3

        s1 = self.scores3(s1)


        s2 = self.upsample_4x(s2)

        s11 = self.RLV(s2, s1)

        s = s11 + s2

        s = self.upsample_8x(s)

        return s


class fcn_RLM(nn.Module):
    def __init__(self, num_classes):
        super(fcn_RLM, self).__init__()

        self.stage1 = nn.Sequential(*list(pretrained_net.children())[:-4])
        self.stage2 = list(pretrained_net.children())[-4]
        self.stage3 = list(pretrained_net.children())[-3]
        self.scores1 = nn.Conv2d(512, num_classes, 1)
        self.scores2 = nn.Conv2d(256, num_classes, 1)
        self.scores3 = nn.Conv2d(128, num_classes, 1)

        self.upsample_8x = nn.ConvTranspose2d(num_classes, num_classes, kernel_size = 16, stride = 8, padding = 4,
                                              bias = False)
        self.upsample_8x.weight.data = bilinear_kernel(num_classes, num_classes, 16)
        self.upsample_4x = nn.ConvTranspose2d(num_classes, num_classes, kernel_size = 4, stride = 2, padding = 1,
                                              bias = False)
        self.upsample_4x.weight.data = bilinear_kernel(num_classes, num_classes, 4)
        self.upsample_2x = nn.ConvTranspose2d(num_classes, num_classes, kernel_size = 4, stride = 2, padding = 1,
                                              bias = False)
        self.upsample_2x.weight.data = bilinear_kernel(num_classes, num_classes, 4)

        self.RLM = RLM(num_classes)

    def forward(self, x):
        x = self.stage1(x)
        s1 = x

        x = self.stage2(x)
        s2 = x

        x = self.stage3(x)
        s3 = x

        s3 = self.scores1(s3)
        s33 = self.upsample_2x(s3)

        s3 = self.RLM(s3, s33)

        s22 = self.scores2(s2)

        s2 = s22 + s3

        s1 = self.scores3(s1)

        s2 = self.upsample_4x(s2)
        s1 = self.RLM(s2, s1)

        s = s1 + s2

        s = self.upsample_8x(s)

        return s