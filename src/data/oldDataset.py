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

from .baseDataset import baseDataset


class oldDataset(baseDataset):
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
        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
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
