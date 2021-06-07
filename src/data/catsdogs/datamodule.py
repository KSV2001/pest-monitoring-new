from typing import Optional

import hydra
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms


class CatsDogsDataModule(LightningDataModule):
    """Creates a data module from the kaggle dataset for cats and dogs.
    Demo Datamodule for Image Classification
    """

    def __init__(
        self,
        dataset: Dataset,
        dataset_test: Dataset,
        num_workers: int = 16,
        batch_size: int = 32,
        shuffle: bool = True,
        drop_last: bool = True,
        pin_memory: bool = True,
        train_val_split: float = 0.8,
        *args,
        **kwargs,
    ):
        """
        Args:
            data_config: Cofiguration of Dataset to be used
            num_workers: how many workers to use for loading data
            normalize: If true applies image normalize
            seed: starting seed for RNG.
            batch_size: desired batch size.
        """
        super().__init__(*args, **kwargs)

        self.dataset = dataset
        self.dataset_test = dataset_test
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.pin_memory = pin_memory
        self.train_val_split = train_val_split

        self.sampler = None if "sampler" not in kwargs else kwargs["sampler"]

    def setup(self, stage: Optional[str] = None):
        """Split the train and valid dataset"""

        train_legth = len(self.dataset) * self.train_val_split
        print(self.dataset)
        self.dataset_train, self.dataset_val = random_split(
            self.dataset, [train_legth, len(self.dataset) - train_legth]
        )
        del self.dataset

    def train_dataloader(self):
        """CatsDogsDataset train set removes a subset to use for validation"""
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
        """CatsDogsDataset val set uses a subset of the training set for validation"""
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
        self.dataset_test: Dataset = hydra.utils.instantiate(
            self.data_config.dataset,
            root=self.data_config.dataset.root,
            split="test",
        )
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

    @property
    def default_transforms(self):
        """TODO: Discuss"""
        return transforms.Compose([transforms.ToTensor(), transforms.Resize(self.img_size)])
