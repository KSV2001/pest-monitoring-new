import cv2
import numpy as np
import torch

class Compose():
    """Composes several augmentations together.
    Args:
        transforms: list, list of transforms to use.
    Example:
        >>> augmentations.Compose([
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
        img = torch.from_numpy(img)
        bbox_coord = torch.from_numpy(bbox_coord)
        bbox_class = torch.from_numpy(bbox_class)
        val_class = torch.Tensor([val_class])
        return img, bbox_coord, bbox_class, val_class

    
class Resize():
    """Resize images"""
    def __init__(self, size=(512,512)):
        self.size = size

    def __call__(self, img, bbox_coord=None, bbox_class=None, val_class=None):
        img = cv2.resize(img, self.size)
        return img, bbox_coord, bbox_class, val_class


class Permute():
    """Permutes images"""
    def __call__(self, img, bbox_coord=None, bbox_class=None, val_class=None):
        img = np.transpose(img, (2, 0, 1))
        return img, bbox_coord, bbox_class, val_class   