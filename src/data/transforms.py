import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2


class Compose:
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
            img, bbox_coord, bbox_class, val_class = transform(
                img, bbox_coord, bbox_class, val_class
            )
        return img, bbox_coord, bbox_class, val_class

class baseAugmentation():
    def __init__(self):
        self.transform = ...
    
    def __call__(self, img, bbox_coord=None, bbox_class=None, val_class=None):
        transformed_img = self.transform(image=img)['image']
        return transformed_img, bbox_coord, bbox_class, val_class

class ToTensor:
    def __call__(self, img, bbox_coord=None, bbox_class=None, val_class=None):
        transform = ToTensorV2()
        transformed_img = transform(image=img)['image']
        bbox_coord = None if bbox_coord is None else torch.from_numpy(bbox_coord)
        bbox_class = None if bbox_class is None else torch.from_numpy(bbox_class)
        val_class = None if val_class is None else torch.Tensor([val_class])
        return transformed_img, bbox_coord, bbox_class, val_class

class Resize(baseAugmentation):
    """Resize images"""
    def __init__(self, height=512, width=512):
        self.transform = A.Resize(height, width)

class Normalize(baseAugmentation):
    """Normalizes images"""

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.transform = A.Normalize(mean=mean, std=std)

class GaussNoise(baseAugmentation):
    """Normalizes images"""
    def __init__(self, var_limit=(10.0, 50.0), mean=0, p=0.1):
        self.transform = A.GaussNoise(var_limit=var_limit, mean=mean, p=p)

class RandomBlur(baseAugmentation):
    """stregth should be an odd integer"""
    def __init__(self, strength=1, p=0.1):
        self.transform = A.OneOf([
            A.Blur(blur_limit=7*strength, p=1),
            A.GaussianBlur(blur_limit=(3*strength, 7*strength), p=1),
            A.MotionBlur(blur_limit=(3*strength,7*strength), p=1)
        ], p=p)

class HorizontalFlip():
    def __init__(self, p=0.25):
        self.transform = A.HorizontalFlip(p=p)
    
    def __call__(self, img, bbox_coord=None, bbox_class=None, val_class=None):
        transformed = self.transform(image=img) if bbox_coord is None else self.transform(image=img, bboxes=bbox_coord)
        transformed_img = transformed['image']
        transformed_bbox_coord = None if bbox_coord is None else transformed['bboxes'] 
        return transformed_img, transformed_bbox_coord, bbox_class, val_class

class VerticalFlip():
    def __init__(self, p=0.25):
        self.transform = A.VerticalFlip(p=p)
    
    def __call__(self, img, bbox_coord=None, bbox_class=None, val_class=None):
        transformed = self.transform(image=img) if bbox_coord is None else self.transform(image=img, bboxes=bbox_coord)
        transformed_img = transformed['image']
        transformed_bbox_coord = None if bbox_coord is None else transformed['bboxes'] 
        return transformed_img, transformed_bbox_coord, bbox_class, val_class
    

# class PILToTensor():
#     """Converts a PIL Image (H x W x C) to a Tensor of shape (C x H x W)"""
#     def __call__(self, img, bbox_coord=None, bbox_class=None, val_class=None):
#         t = transforms.PILToTensor()
#         return t(img), bbox_coord, bbox_class, val_class

# class Permute():
#     """Permutes images"""
#     def __call__(self, img, bbox_coord=None, bbox_class=None, val_class=None):
#         return img.permute(2, 0, 1), bbox_coord, bbox_class, val_class

