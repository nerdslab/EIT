"""transformation: operations for images in pairs and not in pairs"""

import torch
from torchvision.transforms import RandomCrop, RandomHorizontalFlip, Resize
import torchvision.transforms.functional as F


class Two_RandomCrop(RandomCrop):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_two(self, imgA, imgB):
        if self.padding is not None:
            imgA = F.pad(imgA, self.padding, self.fill, self.padding_mode)
            imgB = F.pad(imgB, self.padding, self.fill, self.padding_mode)

        width, height = F.get_image_size(imgA)
        # pad the width if needed
        if self.pad_if_needed and width < self.size[1]:
            padding = [self.size[1] - width, 0]
            imgA = F.pad(imgA, padding, self.fill, self.padding_mode)
            imgB = F.pad(imgB, padding, self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and height < self.size[0]:
            padding = [0, self.size[0] - height]
            imgA = F.pad(imgA, padding, self.fill, self.padding_mode)
            imgB = F.pad(imgB, padding, self.fill, self.padding_mode)

        i, j, h, w = self.get_params(imgA, self.size)

        return F.crop(imgA, i, j, h, w), F.crop(imgB, i, j, h, w)

class Two_RandomHorizontalFlip(RandomHorizontalFlip):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_two(self, imgA, imgB):
        if torch.rand(1) < self.p:
            return F.hflip(imgA), F.hflip(imgB)
        return imgA, imgB

class Two_Resize(Resize):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_two(self, imgA, imgB):
        return F.resize(imgA, self.size, self.interpolation, self.max_size, self.antialias),\
               F.resize(imgB, self.size, self.interpolation, self.max_size, self.antialias)

class Two_aug_sequential(object):
    def __init__(self, func):
        """func as a list"""
        self.func = func

    def augment(self, imgA, imgB):
        for f in self.func:
            imgA, imgB = f.get_two(imgA, imgB)
        return imgA, imgB

class One_RandomCrop(RandomCrop):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_one(self, img):
        if self.padding is not None:
            img = F.pad(img, self.padding, self.fill, self.padding_mode)

        width, height = F.get_image_size(img)
        # pad the width if needed
        if self.pad_if_needed and width < self.size[1]:
            padding = [self.size[1] - width, 0]
            img = F.pad(img, padding, self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and height < self.size[0]:
            padding = [0, self.size[0] - height]
            img = F.pad(img, padding, self.fill, self.padding_mode)

        i, j, h, w = self.get_params(img, self.size)

        return F.crop(img, i, j, h, w)

class One_RandomHorizontalFlip(RandomHorizontalFlip):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_one(self, imgA):
        if torch.rand(1) < self.p:
            return F.hflip(imgA)
        return imgA

class One_Resize(Resize):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_one(self, imgA):
        return F.resize(imgA, self.size, self.interpolation, self.max_size, self.antialias)

class One_aug_sequential(object):
    def __init__(self, func):
        """func as a list"""
        self.func = func

    def augment(self, imgA):
        for f in self.func:
            imgA = f.get_one(imgA)
        return imgA