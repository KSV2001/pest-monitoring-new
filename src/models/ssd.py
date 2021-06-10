from typing import Any

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
import torch
from torch.nn import Module
from torch.optim.optimizer import Optimizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        self.metric = hydra.utils.instantiate(self.model_config.metric)

    def forward(self, x):
        return self.network(x)

    def configure_optimizers(self):
        optimizer: Optimizer = hydra.utils.instantiate(
            self.model_config.optimizer, params=self.parameters()
        )
        return optimizer

    def step(self, batch: Any):
        images, glocs, glabels, img_ids = (
            batch["imgs"],
            batch["bbox_coords"],
            batch["bbox_classes"],
            batch["img_ids"]
        )
        glabels = [glabel + 1 for glabel in glabels]
        plocs, plabels = self.forward(images)
        loss, conf_loss, loc_loss = self.criterion(plocs, plabels, glocs, glabels)
        return {'loss' : loss, 
                'conf_loss' : conf_loss, 
                'loc_loss' : loc_loss, 
                'plocs' : plocs, 
                'plabels' : plabels, 
                'glocs' : glocs, 
                'glabels' : glabels,
                'img_ids' : img_ids
                }

    def training_step(self, batch, batch_idx):
        step_output = self.step(batch)
                                         
        self.log("train/loss", step_output['loss'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("train/conf_loss", step_output['conf_loss'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("train/loc_loss", step_output['loc_loss'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return step_output['loss']

    def validation_step(self, batch, batch_idx):
        step_output = self.step(batch)

        self.log("val/loss", step_output['loss'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val/conf_loss", step_output['conf_loss'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val/loc_loss", step_output['loc_loss'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
