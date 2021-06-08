import os

import torch
from PIL import Image
from torch.utils.data import Dataset


class CatsDogsDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(
        self,
        root: str = "/scratchh/home/apoorv/data/catsdogs",
        split: str = "train",
        transforms=None,
    ):
        """
        :param image_dir: folder where data files are stored
        :param split: split, one of 'train' or 'val' or 'test'
        """
        assert split in {"train", "test"}
        self.root = root
        self.split = split
        self.image_dir = os.path.join(self.root, self.split)
        self.image_names = os.listdir(self.image_dir)
        self.transforms = transforms

    def __getitem__(self, index):
        # Image Name
        image_name = self.image_names[index]
        image = Image.open(os.path.join(self.image_dir, image_name))

        if self.transforms is not None:
            data = {
                "image": image.numpy(),
            }
            augmented = self.transforms(**data)
            image = torch.tensor(augmented["image"])

        return image

    def __len__(self):
        return len(self.image_names)

    # def collate_fn(self, batch):
    #     """
    #     Since each image may have a different number of objects, we need a
    #     collate function (to be passed to the DataLoader).
    #     This describes how to combine these tensors of different sizes. We use lists.
    #     Note: this need not be defined in this Class, can be standalone.
    #     :param batch: an iterable of N sets from __getitem__()
    #     :return: a tensor of images, lists of varying-size tensors of bounding
    #     boxes, labels, and difficulties
    #     """

    #     images = list()

    #     for b in batch:
    #         images.append(b[0])

    #     images = torch.stack(images, dim=0)

    #     # tensor (N, 300, 300), 3 lists of N tensors each
    #     return images
