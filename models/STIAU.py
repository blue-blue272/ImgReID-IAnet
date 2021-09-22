import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import math


class Gconv(nn.Module):
    def __init__(self, in_channels):
        super(Gconv, self).__init__()
        fsm_blocks = []
        fsm_blocks.append(nn.Conv2d(in_channels * 2, in_channels, 1))
        fsm_blocks.append(nn.BatchNorm2d(in_channels))
        fsm_blocks.append(nn.ReLU(inplace=True))
        self.fsm = nn.Sequential(*fsm_blocks)
        # init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, W, x):
        bs, n, c = x.size()

        x_neighbor = torch.bmm(W, x) 
        x = torch.cat([x, x_neighbor], 2) 
        x = x.view(-1, x.size(2), 1, 1) 
        x = self.fsm(x) 
        x = x.view(bs, n, c)
        return x 


class Wcompute(nn.Module):
    def __init__(self, in_channels):
        super(Wcompute, self).__init__()
        self.in_channels = in_channels

        edge_block = []
        edge_block.append(nn.Conv2d(in_channels * 2, 1, 1))
        edge_block.append(nn.BatchNorm2d(1))
        self.relation = nn.Sequential(*edge_block)

        #init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        

    def forward(self, x, W_id, y):
        bs, N, C = x.size()

        W1 = x.unsqueeze(2) 
        W2 = torch.transpose(W1, 1, 2) 
        W_new = torch.abs(W1 - W2) 
        W_new = torch.transpose(W_new, 1, 3) 
        y = y.view(bs, C, 1, 1).expand_as(W_new)
        W_new = torch.cat((W_new, y), 1) 

        W_new = self.relation(W_new) 
        W_new = torch.transpose(W_new, 1, 3)  
        W_new = W_new.squeeze(3) 

        W_new = W_new - W_id.expand_as(W_new) * 1e8
        W_new = F.softmax(W_new, dim=2)
        return W_new


class STIAUModule(nn.Module):
    def __init__(self, in_channels, T, N=4):
        super(STIAUModule, self).__init__()
        self.T = T
        self.in_channels = in_channels
        self.module_w = Wcompute(in_channels)
        self.module_l = Gconv(in_channels)

        W0 = torch.eye(N)
        W = (1 - W0).repeat(T, T)
        for i in range(T):
            W[i*N: (i+1)*N, i*N: (i+1)*N] = W0
        self.W_init = W

    def forward(self, x, y):
        bs, t, N, C = x.size()
        x = x.view(bs, -1, C)

        if t == self.T:
            W_init = self.W_init 
        else:
            W_init = self.W_init[:t*N, :t*N]

        W = self.module_w(x, W_init.unsqueeze(0).cuda(), y) 
        s = self.module_l(W, x) 
        s = s.view(bs, t, N, C)
        return s
