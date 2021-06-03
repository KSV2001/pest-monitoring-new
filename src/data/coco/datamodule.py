import platform
import os
from typing import Optional
from warnings import warn
import hydra
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import LightningDataModule
from torchvision import transforms
from .transforms import SSDTransformer
from src.data.utils import generate_dboxes

class COCODetectionDataModule(LightningDataModule):
    """
    COCODetectionDataModule Lightning Datamodule for Object Detection
    """
    def __init__(
        self,
        data_config: DictConfig,
        dataset: Dataset,
        type: str,
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
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.pin_memory = pin_memory

        self.dboxes = generate_dboxes(self.data_config.type)
        self.sampler = None if "sampler" not in kwargs else kwargs["sampler"]

    def setup(self, stage: Optional[str] = None):
        """Split the train and valid dataset"""
        self.dataset_train: Dataset = hydra.utils.instantiate(self.data_config.dataset,
                                                            root = self.data_config.dataset.root,
                                                            transform = SSDTransformer(dboxes = self.dboxes, test=False),
                                                            split = 'train')
        self.dataset_val: Dataset = hydra.utils.instantiate(self.data_config.dataset,
                                                            root = self.data_config.dataset.root,
                                                            transform = SSDTransformer(dboxes = self.dboxes, test=False),
                                                            split = 'val')
        
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
            collate_fn=self.dataset_train.collate_fn,
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
            collate_fn=self.dataset_val.collate_fn,
        )
        return loader

    def test_dataloader(self):
        """MNIST test set uses the test split"""
        self.dataset_test: Dataset = hydra.utils.instantiate(self.data_config.dataset,
                                                            root = self.data_config.dataset.root,
                                                            transform = SSDTransformer(dboxes = self.dboxes, val=True),
                                                            split = 'test')
        loader = DataLoader(
            self.dataset_test,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
            sampler=self.sampler,
            collate_fn=self.dataset_test.collate_fn,
        )
        return loader