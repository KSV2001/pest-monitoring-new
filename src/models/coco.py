from typing import Any

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from torch.nn import Module
from torch.optim.optimizer import Optimizer

from src.data.utils import generate_dboxes


class Model(pl.LightningModule):
    """Base class for any machine learning model using Pytorch Lightning

    :param config: Config object
    :type config: Config
    """

    def __init__(self, model_config: DictConfig, **kwargs):
        super().__init__()
        self.model_config = model_config
        self.dboxes = generate_dboxes(model=self.model_config.type)

        self.network: Module = hydra.utils.instantiate(self.model_config.network)
        self.criterion: Module = hydra.utils.instantiate(self.model_config.loss, dboxes=self.dboxes)

    def forward(self, x):
        return self.network(x)

    def configure_optimizers(self):
        optimizer: Optimizer = hydra.utils.instantiate(
            self.model_config.optimizer, params=self.parameters()
        )
        return optimizer

    def step(self, batch: Any):
        image, gloc, glabel = batch
        ploc, plabel = self.forward(image)
        ploc, plabel = ploc.float(), plabel.float()
        gloc = gloc.transpose(1, 2).contiguous()

        loss = self.criterion(ploc, plabel, gloc, glabel)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.step(batch)

        # output = self.train_metrics(out, y)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        # self.log_dict(output, on_step=False, on_epoch=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch)

        # output = self.valid_metrics(out, y)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        # self.log_dict(output, on_step=False, on_epoch=True)
