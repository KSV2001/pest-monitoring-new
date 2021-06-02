import torch
from torch.utils.data import Dataset
import json
import os
from os.path import join
from PIL import Image
from pycocotools.coco import COCO
from torchvision.datasets import CocoDetection
import numpy as np
import torchvision
from torch.utils.data.dataloader import default_collate
from .utils import Encoder
from PIL import Image
from typing import Any, Callable, Optional, Tuple, List

class COCODataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, root, split = 'train', version = '2014', transform=None):
        """
        :param image_dir: folder where data files are stored
        :param annotations_dir: folder where annotation files corresponding to splits are stored
        :param split: 
        :param split: split, one of 'train' or 'val' or 'test'
        """
        assert split in {'train', 'val', 'test'}
        self.root = os.path.join(root, split)
        self.version = version
        self.split = split
        
        # Load coco object using pycoco
        if split == 'test':
            self.coco = COCO(join(root, 'annotations', f'image_info_test{version}.json'))
        else:
            self.coco = COCO(join(root, 'annotations', f'instances_{split}{version}.json'))
        self.ids = list(sorted(self.coco.imgs.keys()))
        self._load_categories()
        self.transform = transform

    def _load_image(self, id: int) -> Image.Image:
        path = self.coco.loadImgs(id)[0]["file_name"]
        return Image.open(os.path.join(self.root, path)).convert("RGB")

    def _load_target(self, id) -> List[Any]:
        return self.coco.loadAnns(self.coco.getAnnIds(id))

    def _load_categories(self):
        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x["id"])

        self.label_map = {}
        self.label_info = {}
        counter = 1
        self.label_info[0] = "background"
        for c in categories:
            self.label_map[c["id"]] = counter
            self.label_info[counter] = c["name"]
            counter += 1

    def __getitem__(self, index):
        # Image ID
        img_id = self.ids[index]
        image, target = self._load_image(img_id), self._load_target(img_id)
        width, height = image.size

        if len(target) == 0:
            return None, None, None, None, None

        # Bounding boxes for objects
        # In coco format, bbox = [xmin, ymin, width, height]
        # In pytorch, the input should be [xmin, ymin, xmax, ymax] normalized
        boxes = []
        labels = []
        for annotation in target:
            bbox = annotation.get("bbox")
            boxes.append([bbox[0] / width, bbox[1] / height, (bbox[0] + bbox[2]) / width, (bbox[1] + bbox[3]) / height])
            labels.append(self.label_map[annotation.get("category_id")])

        boxes = torch.tensor(boxes)
        labels = torch.tensor(labels)
        
        if self.transform is not None:
            image, boxes, labels = self.transform(image, boxes, labels)

        return image, boxes, labels

    def __len__(self):
        return len(self.ids)

    def collate_fn(self, batch):
        items = list(zip(*batch))
        items[0] = default_collate([i for i in items[0] if torch.is_tensor(i)])
        items[1] = default_collate([i for i in items[1] if torch.is_tensor(i)])
        items[2] = default_collate([i for i in items[2] if torch.is_tensor(i)])    
        
        return items
