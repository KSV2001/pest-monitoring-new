import logging
import os
import hydra
import random
import time
import urllib.request
from typing import Optional, Sequence, Tuple, Union
from PIL import Image
import numpy as np
import cv2
import pandas as pd

from torch.utils.data.dataloader import default_collate
import torch
from omegaconf import DictConfig
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms

from src.data.pest_default_dataset import PestDefaultDataset

class DetectionDataset(PestDefaultDataset):
    def __init__(
        self,
        dataset_config: DictConfig,
        IMG_DIR: str,
        SPLIT_DIR: str,
        mode: str,
        BOX_DIR: Union[str, None] = None,
        VAL_FILE: Union[str, None] = None,
        transforms = None,
        **kwargs,
    ):
        """//  todo -1 (general) +0: TODO: add type annotations"""
        super().__init__(
            dataset_config,
            IMG_DIR,
            SPLIT_DIR,
            mode,
            BOX_DIR,
            VAL_FILE,
            transforms,
            **kwargs)
    
    def __getitem__(self, idx: int):
        img_id, img, bbox_coord, bbox_class, val_class = self.pull_item(idx)
        
        if len(bbox_class) == 0:
            return None, None, None

        bbox_coord = torch.tensor(bbox_coord)
        bbox_class = torch.tensor(bbox_class)

        if self.transforms is not None:
            img, bbox_coord, bbox_class = self.transforms(img, bbox_coord, bbox_class.long())

        return img, bbox_coord, bbox_class

    def read_img(self, path, *args, **kwargs):
        """//  todo -1 (general) +0: TODO: This should also have exception. Ideally read image with cv2 -> current docker does not support cv2."""
        im = Image.open(path).convert('RGB')
        return im    

    def collate_fn(self, batch):
        items = list(zip(*batch))
        items[0] = default_collate([i for i in items[0] if torch.is_tensor(i)])
        items[1] = default_collate([i for i in items[1] if torch.is_tensor(i)])
        items[2] = default_collate([i for i in items[2] if torch.is_tensor(i)])    
        
        return items
