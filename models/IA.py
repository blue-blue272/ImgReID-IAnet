from __future__ import absolute_import

import torch
import math
from torch import nn
from torch.nn import functional as F
import numpy as np


def generate_grid(h, w):
    x = np.linspace(0, w-1, w)
    y = np.linspace(0, h-1, h)
    xv, yv = np.meshgrid(x, y)
    xv = xv.flatten()
    yv = yv.flatten()
    return xv, yv

def generate_gaussian(height, width, alpha_x, alpha_y):
    Dis = np.zeros((height*width, height*width))
    xv, yv = generate_grid(height, width)
    for i in range(0, width):
        for j in range(0, height):
            d = (np.square(xv - i))/ (2 * alpha_x**2)  + (np.square(yv - j)) / (2 * alpha_y**2)
            Dis[i+j*width] = -1 *  d 
    Dis = torch.from_numpy(Dis).float()
    Dis = F.softmax(Dis, dim=-1)
    return Dis


class _IABlockND(nn.Module):
    def __init__(self, in_channels, height, width,
            alpha_x, alpha_y):
        super(_IABlockND, self).__init__()

        self.in_channels = in_channels
        conv_nd = nn.Conv2d
        max_pool = nn.MaxPool2d
        bn = nn.BatchNorm2d

        self.Dis = generate_gaussian(height=height, width=width, alpha_x=alpha_x, alpha_y=alpha_y)
        self.W1 = bn(self.in_channels)
        self.W2 = bn(self.in_channels)
        
        # init
        for m in self.modules():
            if isinstance(m, conv_nd):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, bn):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        nn.init.constant_(self.W1.weight.data, 0.0)
        nn.init.constant_(self.W1.bias.data, 0.0)
        nn.init.constant_(self.W2.weight.data, 0.0)
        nn.init.constant_(self.W2.bias.data, 0.0)


    def forward(self, x):
        '''
        :param x: (b, c, h, w)
        :return:
        '''

        batch_size = x.size(0)

        g_x = x.view(batch_size, self.in_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        f_cluster = []
        f_loc = torch.unsqueeze(self.Dis.cuda(), 0)
        f_loc = f_loc.expand(batch_size, -1, -1)
        f_cluster.append(torch.unsqueeze(f_loc, 1))

        theta_x = x.view(batch_size, self.in_channels, -1)
        theta_x = theta_x.permute(0, 2, 1) #[B, H*W, C]
        phi_x = x.view(batch_size, self.in_channels, -1)
        f = torch.matmul(theta_x, phi_x) #[B, H*W, H*W]
        f = f / np.sqrt(self.in_channels)
        f = F.softmax(f, dim=-1)
        f_cluster.append(torch.unsqueeze(f, 1))
        
        f_cluster = torch.cat(f_cluster, 1)
        f = torch.prod(f_cluster, dim=1)
        f = F.softmax(f, dim=-1)

        y = torch.matmul(f, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.in_channels, *x.size()[2:])
        y = self.W1(y)
        z = y + x

        x = z
        g_x = x.view(batch_size, self.in_channels, -1) #[B, c, h*w]
        theta_x = g_x #[B, c, h*w]
        phi_x = g_x.permute(0, 2, 1) #[B, h*w, c]
        f = torch.matmul(theta_x, phi_x) #[B, c, c]
        f = F.softmax(f, dim=-1)
        y = torch.matmul(f, g_x)
        y = y.view(batch_size, self.in_channels, *x.size()[2:])
        y = self.W2(y)
        z = y + x

        return z



class IABlock2D(_IABlockND):
    def __init__(self, in_channels, height, width, alpha_x, alpha_y, **kwargs):
        super(IABlock2D, self).__init__(in_channels, height=height, width=width,
                                alpha_x=alpha_x, alpha_y=alpha_y)

