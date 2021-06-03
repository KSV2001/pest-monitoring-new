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

import torch
from omegaconf import DictConfig
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms

class BaseDataset(Dataset):
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
        super().__init__()
        self.IMG_DIR = IMG_DIR
        self.mode = mode
        self.SPLIT_DIR = SPLIT_DIR
        self.SPLIT_FILE = os.path.join(self.SPLIT_DIR, self.mode+'.txt')
        self.BOX_DIR = BOX_DIR
        self.VAL_FILE = VAL_FILE
        self.dataset_config = dataset_config

        self.prepare_data()

        self.transforms = transforms
            
    def __len__(self) -> int:
        return len(self.img_ids)
    
    def __getitem__(self, idx: int):
        img_id, img, bbox_coord, bbox_class, val_class = self.pull_item(idx)
        if self.transforms is not None:
            img, bbox_coord, bbox_class, val_class = self.transforms(img, bbox_coord, bbox_class, val_class)
        record = {
                    "img_id": img_id, 
                    "img": img.contiguous(), 
                    "bbox_coord": bbox_coord, 
                    "bbox_class": bbox_class, 
                    "val_class": val_class,
                }
        return record

    def pull_item(self, idx: int):
        img_id = self.img_ids[idx]
        img = self.read_img(self.img_paths[idx])
        bbox_coord, bbox_class = self.read_box(self.box_paths[idx])
        val_class = self.val_classes[idx]
        
        return img_id, img, bbox_coord, bbox_class, val_class
        
    
    def prepare_data(self, *args, **kwargs):
        """//  todo -4 (general) +0: TODO: Mention required dtypes, format of self.img_ids...."""
        self.img_ids = []
        self.img_paths = []
        self.box_paths = []
        self.val_classes = []

        assert self.BOX_DIR is not None or self.VAL_FILE is not None, "Pass either box annotations or validation labels"

        with open(self.SPLIT_FILE) as f:
            self.img_ids = f.read().splitlines()

        self.img_paths += [os.path.join(self.IMG_DIR, img_id + ".jpg") for img_id in self.img_ids]
        
        if self.BOX_DIR is not None:
            self.box_paths += [os.path.join(self.BOX_DIR, img_id + ".txt") for img_id in self.img_ids]
        else:
            self.box_paths = [None]*len(self.img_ids)

        if self.VAL_FILE is not None:
            self.val_classes = self.read_val_class()
        else:
            self.val_classes = [None]*len(self.img_ids)

    
    def read_img(self, path, *args, **kwargs):
        """//  todo -1 (general) +0: TODO: This should also have exception. Ideally read image with cv2 -> current docker does not support cv2."""
        im = Image.open(path)
        im = np.asarray(im).copy()
        return im

    
    def read_box(self, path, *args, **kwargs):
        if path is None or not os.stat(path).st_size:
            annot = np.zeros((0, 5), dtype=np.float32)
        else:
            annot = np.loadtxt(path, dtype=np.float32).reshape(-1, 5)
        bbox_coord = annot[:, 1:]
        bbox_class = annot[:, 0]
        return bbox_coord, bbox_class

    def read_val_class(self, *args, **kwargs):
        labels = pd.read_csv(self.VAL_FILE, index_col=0, header=None).loc[self.img_ids, 1].tolist()
        return [self.dataset_config.label_map[x] for x in labels]