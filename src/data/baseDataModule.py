import platform
import os
from typing import Optional
from warnings import warn
import hydra
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import LightningDataModule
from torchvision import transforms
from .baseCollate import default_collate
from .augmentations import *

class baseDataModule(LightningDataModule):
    """
    Standard MNIST, train, val, test splits and transforms

    """

    name = "base"

    def __init__(
        self,
        data_config: DictConfig,
        dataset: Dataset,
        num_workers: int = 16,
        batch_size: int = 32,
        shuffle: bool = True,
        drop_last: bool = True,
        pin_memory: bool = True,
        *args,
        **kwargs,
    ):
        """
        Args:
            data_dir: where to save/load the data
            val_split: how many of the training images to use for the validation split
            num_workers: how many workers to use for loading data
            normalize: If true applies image normalize
            seed: starting seed for RNG.
            batch_size: desired batch size.
        """
        super().__init__(*args, **kwargs)
        
        self.data_config = data_config
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.dataset_train = ...
        self.dataset_val = ...
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.pin_memory = pin_memory

        self.sampler = None if "sampler" not in kwargs else kwargs["sampler"]  # Take sampler as a input in config file
        self.collate_fn = default_collate if "collate_fn" not in self.data_config else self.data_config.collate_fn

        
        self.train_transforms = self.default_transforms if self.data_config.dataset.transforms.train is None \
                                else transforms.Compose([hydra.utils.instantiate(transform) for transform in self.data_config.dataset.transforms.train])
        self.val_transforms = self.default_transforms if self.data_config.dataset.transforms.train is None \
                                else transforms.Compose([hydra.utils.instantiate(transform) for transform in self.data_config.dataset.transforms.val])
        self.test_transforms = self.default_transforms if self.data_config.dataset.transforms.train is None \
                                else transforms.Compose([hydra.utils.instantiate(transform) for transform in self.data_config.dataset.transforms.test])

    # def prepare_data(self):
    #     """Saves MNIST files to `data_dir`"""
    #     MNIST(self.data_dir, train=True, download=True)
    #     MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None):
        """Split the train and valid dataset"""
        self.dataset_train: Dataset = hydra.utils.instantiate(self.data_config.dataset,
                                                             dataset_config = self.data_config.dataset,
                                                             mode = 'train',
                                                             transforms = self.train_transforms,
                                                              _recursive_=False)
        self.dataset_val: Dataset = hydra.utils.instantiate(self.data_config.dataset,
                                                            dataset_config = self.data_config.dataset,
                                                            mode = 'val',
                                                            transforms = self.val_transforms,
                                                             _recursive_=False)
        
    def train_dataloader(self):
        """MNIST train set removes a subset to use for validation"""
        loader = DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
            sampler=self.sampler,
            collate_fn=self.collate_fn,
        )
        return loader

    def val_dataloader(self):
        """MNIST val set uses a subset of the training set for validation"""
        loader = DataLoader(
            self.dataset_val,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
            sampler=self.sampler,
            collate_fn=self.collate_fn,
        )
        return loader

    def test_dataloader(self):
        """MNIST test set uses the test split"""
        self.dataset_test: Dataset = hydra.utils.instantiate(self.data_config.dataset,
                                                            dataset_config = self.data_config.dataset,
                                                            mode = 'test',
                                                            transforms = self.test_transforms,
                                                             _recursive_=False)
        loader = DataLoader(
            self.dataset_test,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
            sampler=self.sampler,
            collate_fn=self.collate_fn,
        )
        return loader

    @property
    def default_transforms(self):
        """TODO: Discuss"""
        return Compose([
            Resize(),
            ToTensor(),
        ])