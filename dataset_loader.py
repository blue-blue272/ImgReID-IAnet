from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
from PIL import Image
import numpy as np
import os.path as osp
import lmdb
import io

import torch
from torch.utils.data import Dataset


def read_image(img_path, mode='RGB'):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert(mode)
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


class ImageDataset(Dataset):
    """Image Person ReID Dataset"""
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid = self.dataset[index]
        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return img, pid, camid


class ImageDataset_seg(Dataset):
    """Image Person ReID Dataset"""
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        path, pid, camid = self.dataset[index]
        img_path, head_path, upper_body_path, lower_body_path, shoes_path, foreground_path = path
        
        img = read_image(img_path, mode='RGB')
        head = read_image(head_path, mode='L')
        upper_body = read_image(upper_body_path, mode='L')
        lower_body = read_image(lower_body_path, mode='L')
        shoes = read_image(shoes_path, mode='L')
        foreground = read_image(foreground_path, mode='L')

        sequence = [img, head, upper_body, lower_body, shoes, foreground]
        
        if self.transform is not None:
            self.transform.randomize_parameters()
            sequence = [self.transform(img) for img in sequence]

        sequence = torch.cat(sequence, 0) #[3+1+1+1, h, w]
        
        return sequence, pid, camid
