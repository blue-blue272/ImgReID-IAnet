from __future__ import absolute_import
from __future__ import division

import torch
from torch import nn
from torch.nn import functional as F
import torchvision
from .IAU import IAU
from .resnets1 import resnet50_s1


class IAUnet(nn.Module):
    def __init__(self, num_classes):
        super(IAUnet, self).__init__()
        resnet50 = resnet50_s1(pretrained=True)

        self.conv1 = resnet50.conv1
        self.bn1 = resnet50.bn1
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = resnet50.maxpool

        self.layer1 = resnet50.layer1
        self.layer2 = resnet50.layer2
        self.layer3 = resnet50.layer3 
        self.layer4 = resnet50.layer4 

        self.IAU2 = IAU(512) 
        self.IAU3 = IAU(1024)

        self.feat_dim = 2048
        self.bn = nn.BatchNorm1d(self.feat_dim)
        self.classifier = nn.Linear(self.feat_dim, num_classes)

    def forward(self, x):
        x = self.bn1(self.conv1(x))
        x = self.maxpool(self.relu(x))

        x1 = self.layer1(x) 

        x2 = self.layer2(x1) 
        x2, a2 = self.IAU2(x2) 

        x3 = self.layer3(x2) 
        x3, a3 = self.IAU3(x3) 

        x4 = self.layer4(x3) 

        f = F.avg_pool2d(x4, x4.size()[2:])
        f = f.view(f.size(0), -1)
        f = self.bn(f)
        y = self.classifier(f)

        a_head = [a2[:,0:1], a3[:,0:1]]
        a_upper = [a2[:,1:2], a3[:,1:2]]
        a_lower = [a2[:,2:3], a3[:,2:3]]
        a_shoes = [a2[:,3:4], a3[:,3:4]]

        return y, f, a_head, a_upper, a_lower, a_shoes
