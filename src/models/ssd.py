from typing import Any

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from torch.nn import Module
from torch.optim.optimizer import Optimizer

class Model(pl.LightningModule):
    """Base class for any machine learning model using Pytorch Lightning

    :param config: Config object
    :type config: Config
    """

    def __init__(self, model_config: DictConfig, **kwargs):
        super().__init__()
        self.model_config = model_config

        self.network: Module = hydra.utils.instantiate(self.model_config.network)
        self.criterion: Module = hydra.utils.instantiate(
            self.model_config.loss, priors_cxcy=self.network.priors_cxcy
            )

    def forward(self, x):
        return self.network(x)

    def configure_optimizers(self):
        optimizer: Optimizer = hydra.utils.instantiate(
            self.model_config.optimizer, params=self.parameters()
        )
        return optimizer

    def step(self, batch: Any):
        images, glocs, glabels = (
            batch["imgs"],
            batch["bbox_coords"],
            batch["bbox_classes"]
        )
        plocs, plabels = self.forward(images)
        loss = self.criterion(plocs, plabels, glocs, glabels)
        return loss, glocs, glabels, plocs, plabels

    def training_step(self, batch, batch_idx):
        loss, glocs, glabels, plocs, plabels = self.step(batch)

        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, glocs, glabels, plocs, plabels = self.step(batch)

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
