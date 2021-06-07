import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

class Compose():
    """Composes several transforms together.
    Args:
        transforms: list, list of transforms to use.
    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bbox_coord=None, bbox_class=None, val_class=None):
        for transform in self.transforms:
            img, bbox_coord, bbox_class, val_class = transform(img, bbox_coord, bbox_class, val_class)
        return img, bbox_coord, bbox_class, val_class


class ToTensor():
    def __call__(self, img, bbox_coord=None, bbox_class=None, val_class=None):
        img = F.to_tensor(img)
        bbox_coord = torch.from_numpy(bbox_coord)
        bbox_class = torch.from_numpy(bbox_class)
        val_class = torch.Tensor([val_class])
        return img, bbox_coord, bbox_class, val_class

class Resize():
    """Resize images"""
    def __init__(self, width=512, height=512):
        self.size = (width, height)
        self.resize = transforms.Resize(size = self.size)

    def __call__(self, img, bbox_coord=None, bbox_class=None, val_class=None):
        return self.resize(img), bbox_coord, bbox_class, val_class

class PILToTensor():
    """Converts a PIL Image (H x W x C) to a Tensor of shape (C x H x W)"""
    def __call__(self, img, bbox_coord=None, bbox_class=None, val_class=None):
        t = transforms.PILToTensor()
        return t(img), bbox_coord, bbox_class, val_class

class Normalize():
    """Normalizes images"""
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.t = transforms.Normalize(mean = mean, std = std)

    def __call__(self, img, bbox_coord=None, bbox_class=None, val_class=None):
        return self.t(img), bbox_coord, bbox_class, val_class

class Permute():
    """Permutes images"""
    def __call__(self, img, bbox_coord=None, bbox_class=None, val_class=None):
        return img.permute(2, 0, 1), bbox_coord, bbox_class, val_class   