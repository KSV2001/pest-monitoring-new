import os
from os.path import join

import torch
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset


class COCODataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, root, split="train", version="2014", transforms=None):
        """
        :param image_dir: folder where data files are stored
        :param annotations_dir: folder where annotation files corresponding to splits are stored
        :param split:
        :param split: split, one of 'train' or 'val' or 'test'
        """
        assert split in {"train", "val", "test"}
        self.root = root
        self.verion = version
        self.split = split

        # Load coco object using pycoco
        # Load coco object using pycoco
        if split == "test":
            self.coco = COCO(join(root, "annotations", f"image_info_test{version}.json"))
        else:
            self.coco = COCO(join(root, "annotations", f"instances_{split}{version}.json"))
        self.image_dir = join(root, f"{split}")
        self.image_ids = list(sorted(self.coco.imgs.keys()))
        self.transforms = transforms

    def __getitem__(self, index):
        # Own coco file
        coco = self.coco

        # Image ID
        img_id = self.image_ids[index]
        # List: get annotation id from coco
        ann_ids = coco.getAnnIds(imgIds=img_id)
        # Dictionary: target coco_annotation file for an image
        coco_annotation = coco.loadAnns(ann_ids)
        # path for input image
        path = coco.loadImgs(img_id)[0]["file_name"]
        # open the input image
        img = Image.open(os.path.join(self.image_dir, path))
        # number of objects in the image
        num_objs = len(coco_annotation)

        # Bounding boxes for objects
        # In coco format, bbox = [xmin, ymin, width, height]
        # In pytorch, the input should be [xmin, ymin, xmax, ymax]
        boxes = []
        for i in range(num_objs):
            xmin = coco_annotation[i]["bbox"][0]
            ymin = coco_annotation[i]["bbox"][1]
            xmax = xmin + coco_annotation[i]["bbox"][2]
            ymax = ymin + coco_annotation[i]["bbox"][3]
            boxes.append([xmin, ymin, xmax, ymax])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # Labels (In my case, I only one class: target class or background)
        labels = torch.ones((num_objs,), dtype=torch.int64)
        # Tensorise img_id
        img_id = torch.tensor([img_id])
        # Size of bbox (Rectangular)
        areas = []
        for i in range(num_objs):
            areas.append(coco_annotation[i]["area"])
        areas = torch.as_tensor(areas, dtype=torch.float32)
        # Iscrowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # Annotation is in dictionary format
        my_annotation = {}
        my_annotation["boxes"] = boxes
        my_annotation["labels"] = labels
        my_annotation["image_id"] = img_id
        my_annotation["area"] = areas
        my_annotation["iscrowd"] = iscrowd

        if self.transforms is not None:
            data = {
                "image": img.numpy(),
                "bboxes": my_annotation["boxes"].numpy(),
                "labels": my_annotation["labels"].numpy(),
            }
            augmented = self.transforms(**data)
            img = torch.tensor(augmented["image"])
            bboxes = torch.tensor(augmented["bboxes"])
            labels = torch.tensor(augmented["labels"])

        return img, bboxes, labels

    def __len__(self):
        return len(self.image_ids)

    def collate_fn(self, batch):
        """
        Since each image may have a different number of objects, we need a
        collate function (to be passed to the DataLoader).
        This describes how to combine these tensors of different sizes. We use lists.
        Note: this need not be defined in this Class, can be standalone.
        :param batch: an iterable of N sets from __getitem__()
        :return: a tensor of images, lists of varying-size tensors of bounding
        boxes, labels, and difficulties
        """

        images = list()
        boxes = list()
        labels = list()

        for b in batch:
            images.append(b[0])
            boxes.append(b[1])
            labels.append(b[2])

        images = torch.stack(images, dim=0)

        return images, boxes, labels  # tensor (N, 3, 300, 300), 3 lists of N tensors each
