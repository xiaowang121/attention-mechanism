import torch
import torch.nn as nn
import torchvision


class NL_local(nn.Module):
    def __init__(self, channel):
        super(NL_local, self).__init__()
        self.inter_channel = channel // 2
        self.conv_phi = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1,padding=0, bias=False)
        self.conv_theta = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_g = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.conv_mask = nn.Conv2d(in_channels=self.inter_channel, out_channels=channel, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):

        b, c, h, w = x.size()

        x_phi = self.conv_phi(x).view(b, c, -1)

        x_theta = self.conv_theta(x).view(b, c, -1).permute(0, 2, 1).contiguous()
        x_g = self.conv_g(x).view(b, c, -1).permute(0, 2, 1).contiguous()

        mul_theta_phi = torch.matmul(x_theta, x_phi)
        mul_theta_phi = self.softmax(mul_theta_phi)

        mul_theta_phi_g = torch.matmul(mul_theta_phi, x_g)

        mul_theta_phi_g = mul_theta_phi_g.permute(0,2,1).contiguous().view(b,self.inter_channel, h, w)

        mask = self.conv_mask(mul_theta_phi_g)
        out = mask + x
        return out
