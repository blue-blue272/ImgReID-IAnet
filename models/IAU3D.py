from __future__ import absolute_import

import torch
import math
from torch import nn
from torch.nn import functional as F
import numpy as np
from models.STIAU import STIAUModule


class ConvBlock(nn.Module):
    """Basic convolutional block"""
    def __init__(self, in_c, out_c, k, s=1, p=0):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv3d(in_c, out_c, k, stride=s, padding=p)
        self.bn = nn.BatchNorm3d(out_c)

    def forward(self, x):
        return self.bn(self.conv(x))


class SpatialAttn(nn.Module):
    """Spatial Attention """
    def __init__(self, in_channels, number):
        super(SpatialAttn, self).__init__()
        self.conv = ConvBlock(in_channels, number, 1)

    def forward(self, x):
        x = self.conv(x) #[bs, 4, t, h, w]
        a = torch.sigmoid(x)
        return a


class IAUBlock3D(nn.Module):
    def __init__(self, in_channels, seq_len):
        super(IAUBlock3D, self).__init__()
        self.in_channels = in_channels
        conv_nd = nn.Conv3d
        bn = nn.BatchNorm3d
        self.inter_channels = in_channels // 2

        self.SA = SpatialAttn(in_channels, number=4)
        self.g = nn.Conv2d(self.in_channels, self.inter_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.STIAU = STIAUModule(self.inter_channels, T=seq_len)

        self.W1 = nn.Sequential(
                conv_nd(self.in_channels, self.in_channels,
                    kernel_size=1, stride=1, padding=0, bias=True),
                bn(self.in_channels)
            )

        self.W2 = nn.Sequential(
                conv_nd(self.in_channels, self.in_channels,
                    kernel_size=1, stride=1, padding=0, bias=True),
                bn(self.in_channels)
            )
        
        # init
        for m in self.modules():
            if isinstance(m, conv_nd) or isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, bn):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        nn.init.constant_(self.W1[1].weight.data, 0.0)
        nn.init.constant_(self.W1[1].bias.data, 0.0)
        nn.init.constant_(self.W2[1].weight.data, 0.0)
        nn.init.constant_(self.W2[1].bias.data, 0.0)

    def apply_attention(self, x, a):
        b, c, t, h, w = x.size()
        a = a.view(a.size(0), a.size(1), h * w) 
        x = x.transpose(1, 2).contiguous().view(b * t, -1, h * w) 
        y = torch.matmul(a, x.transpose(1, 2)) 
        y = y.view(b, t, -1, c)
        return y

    def reduce_dimension(self, x, u):
        bs, t, n, c = x.size()

        x = x.view(bs * t * n, c)
        x = torch.cat((x, u), 0) 
        x = self.g(x.view(x.size(0), x.size(1), 1, 1))
        x = x.view(x.size(0), x.size(1))
        
        u = x[bs * t * n:, :] 
        x = x[: bs * t * n, :].view(bs, t, n, -1)
        return x, u

    def forward(self, x):
        # CIAU
        batch_size, t = x.size(0), x.size(2)

        g_x = x.view(batch_size, self.in_channels * t, -1) 

        theta_x = g_x 
        phi_x = g_x.permute(0, 2, 1) 
        f = torch.matmul(theta_x, phi_x) 
        f = F.softmax(f, dim=-1)

        y = torch.matmul(f, g_x) 
        y = y.view(batch_size, self.in_channels, *x.size()[2:])
        y = self.W1(y)
        z = y + x

        # STIAU
        x = z
        inputs = x

        b, c, t, h, w = x.size()
        u = x.view(b, c, -1).mean(2) 

        a = self.SA(x) 
        a = a.transpose(1, 2).contiguous()
        a = a.view(b * t, -1, h, w) 

        x = self.apply_attention(x, a)
        x, u = self.reduce_dimension(x, u)
        y = self.STIAU(x, u) 

        y = torch.mean(y, 2) 
        u = u.unsqueeze(1).expand_as(y)
        u = torch.cat((y, u), 2) 

        y = self.W2(u.transpose(1, 2).unsqueeze(-1).unsqueeze(-1))
        z = y + inputs
        return z, a
