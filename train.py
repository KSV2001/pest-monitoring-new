import warnings
import logging
import argparse
import os
from os.path import join, dirname
import multiprocessing as mp
import wandb
from ace.config import Config
from src.models.base import DemoModel
from ace.utils.logger import set_logger
from training.utils import seed_everything

warnings.simplefilter('ignore')


def main():
    # data
    dataset = MNIST('', train=True, download=True, transform=transforms.ToTensor())
    mnist_train, mnist_val = random_split(dataset, [55000, 5000])
    train_loader = DataLoader(mnist_train, batch_size=32)
    val_loader = DataLoader(mnist_val, batch_size=32)
    # model
    model = Model()
    # training
    trainer = pl.Trainer(gpus=4, precision=16, limit_train_batches=0.5)
    trainer.fit(model, train_loader, val_loader)

if __name__ == '__main__':

    main()
