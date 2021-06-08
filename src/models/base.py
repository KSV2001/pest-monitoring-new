from typing import Any

import hydra
import pytorch_lightning as pl
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.nn import Module
from torch.optim.optimizer import Optimizer
from torchmetrics import MetricCollection


class Model(pl.LightningModule):
    """Base class for any machine learning model using Pytorch Lightning

    :param config: Config object
    :type config: Config
    """

    def __init__(self, model_config: DictConfig, **kwargs):
        super().__init__()
        self.model_config = model_config

        self.network: Module = hydra.utils.instantiate(self.model_config.network)
        self.criterion: Module = hydra.utils.instantiate(self.model_config.loss)

        metrics = MetricCollection(
            [hydra.utils.instantiate(metric) for metric in self.model_config.metrics.metric_list]
        )
        self.train_metrics = metrics.clone(prefix="train/")
        self.valid_metrics = metrics.clone(prefix="val/")

    def forward(self, x):
        return self.network(x)

    def configure_optimizers(self):
        optimizer: Optimizer = hydra.utils.instantiate(
            self.model_config.optimizer, params=self.parameters()
        )
        return optimizer

    def step(self, batch: Any):
        x, y = batch
        out = self.forward(x)
        loss = self.criterion(out, y)
        return loss, out, y

    def training_step(self, batch, batch_idx):
        loss, out, y = self.step(batch)

        output = self.train_metrics(F.softmax(out, dim=1), y)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log_dict(output, on_step=False, on_epoch=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, out, y = self.step(batch)

        output = self.valid_metrics(F.softmax(out, dim=1), y)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log_dict(output, on_step=False, on_epoch=True)
