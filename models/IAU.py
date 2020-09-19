import torch
from torch import nn
from torch.nn import functional as F
import math
from SIAU import SIAU


class ConvBlock(nn.Module):
    """Basic convolutional block"""
    def __init__(self, in_c, out_c, k, s=1, p=0):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, stride=s, padding=p)
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
        return self.bn(self.conv(x))


class SpatialAttn(nn.Module):
    """Spatial Attention """
    def __init__(self, in_channels, number):
        super(SpatialAttn, self).__init__()
        self.conv = ConvBlock(in_channels, number, 1)

    def forward(self, x):
        x = self.conv(x) 
        a = torch.sigmoid(x)
        return a


class IAU(nn.Module):
    def __init__(self, in_channels):
        super(IAU, self).__init__()

        inter_stride = 2
        self.in_channels = in_channels
        conv_nd = nn.Conv2d
        bn = nn.BatchNorm2d
        self.inter_channels = in_channels // inter_stride

        self.sa = SpatialAttn(in_channels, number=4)

        self.g = conv_nd(self.in_channels, self.inter_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.SIAU = SIAU(self.inter_channels)

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
            if isinstance(m, conv_nd):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, bn):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        nn.init.constant_(self.W1[1].weight.data, 0.0)
        nn.init.constant_(self.W1[1].bias.data, 0.0)
        nn.init.constant_(self.W2[1].weight.data, 0.0)
        nn.init.constant_(self.W2[1].bias.data, 0.0)


    def reduce_dimension(self, x, global_node):
        bs, c = global_node.size()

        x = x.transpose(1, 2).unsqueeze(3) 
        x = torch.cat((x, global_node.view(bs, c, 1, 1)), 2) 
        x = self.g(x).squeeze(3) 

        global_node = x[:,:,-1] 
        x = x[:,:,:-1].transpose(1, 2) 
        return x, global_node


    def forward(self, x):
        # CIAU
        batch_size = x.size(0)

        g_x = x.view(batch_size, self.in_channels, -1)
        theta_x = g_x 
        phi_x = g_x.permute(0, 2, 1) 
        f = torch.matmul(theta_x, phi_x) 
        f = F.softmax(f, dim=-1)
        y = torch.matmul(f, g_x)
        y = y.view(batch_size, self.in_channels, *x.size()[2:])
        y = self.W1(y)
        z = y + x

        # SIAU
        x = z
        inputs = x
        b, c, h, w = x.size()
        u = x.view(b, c, -1).mean(2)

        a = self.sa(x) 
        x = torch.bmm(a.view(b, -1, h * w), x.view(b, c, -1).transpose(1, 2)) 
        x, u = self.reduce_dimension(x, u)
        y = self.SIAU(x, u) 

        y = torch.mean(y, 1) #[b, c//2]
        u = torch.cat((y, u), 1) 

        y = self.W2(u.view(u.size(0), u.size(1), 1, 1))
        z = y + inputs
        return z, a
