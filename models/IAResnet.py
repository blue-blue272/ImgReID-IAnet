from __future__ import absolute_import
from __future__ import division

import torch
from torch import nn
from torch.nn import functional as F
import torchvision
from .resnets1 import resnet50_s1
from .IA import IABlock2D


class IAResNet50_location(nn.Module):

    def __init__(self, last_s1, num_classes,  **kwargs):

        super(IAResNet50_location, self).__init__()

        if not last_s1:
            resnet50 = torchvision.models.resnet50(pretrained=True)
        else:
            resnet50 = resnet50_s1(pretrained=True)
        
        self.conv1 = resnet50.conv1
        self.bn1 = resnet50.bn1
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = resnet50.maxpool

        self.layer1 = self._inflate_reslayer(resnet50.layer1)
        self.layer2 = self._inflate_reslayer(resnet50.layer2, IA_idx=[3], height=32,
                                    width=16, alpha_x=10, alpha_y=20, IA_channels=512)
        self.layer3 = self._inflate_reslayer(resnet50.layer3, IA_idx=[5], height=16,
                                    width=8, alpha_x=5, alpha_y=10, IA_channels=1024)
        self.layer4 = self._inflate_reslayer(resnet50.layer4)

        
        self.bn = nn.BatchNorm1d(2048)
        self.classifier = nn.Linear(2048, num_classes)


    def _inflate_reslayer(self, reslayer, height=0, width=0,
                    alpha_x=0, alpha_y=0, IA_idx=[], IA_channels=0):
        reslayers = []
        for i, layer2d in enumerate(reslayer):
            reslayers.append(layer2d)

            if i in IA_idx:
                IA_block = IABlock2D(in_channels=IA_channels, height=height,
                        width=width, alpha_x=alpha_x, alpha_y=alpha_y)
                reslayers.append(IA_block)

        return nn.Sequential(*reslayers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        f = F.avg_pool2d(x, x.size()[2:])
        f = f.view(f.size(0), -1)
        f = self.bn(f)
        if not self.training:
            return f
        y = self.classifier(f)

        return y, f






