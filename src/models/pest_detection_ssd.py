from typing import Any

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
import torch
from torch.nn import Module
from torch.optim.optimizer import Optimizer

from src.data.utils import Encoder
from src.utils.prior_boxes import generate_dboxes
from src.metrics.pascal_voc_evaluator import get_metrics


class Model(pl.LightningModule):
    """Base class for any machine learning model using Pytorch Lightning

    :param config: Config object
    :type config: Config
    """

    def __init__(self, model_config: DictConfig, **kwargs):
        super().__init__()
        self.model_config = model_config
        self.dboxes = generate_dboxes(size=self.model_config.img_size)
        self.box_encoder = Encoder(dboxes=self.dboxes)
        self.img_shape = (self.model_config.img_size, self.model_config.img_size)

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
        images, glocs, glabels, img_ids = (
            batch["imgs"],
            batch["bbox_coords"],
            batch["bbox_classes"],
            batch["img_ids"],
        )
        gloc_anchored, glabel_anchored, glocs, glabels = self.pre_forward_step(glocs, glabels)
        plocs, plabels = self.forward(images)
        # plocs, plabels = plocs.transpose(1, 2), plabels.transpose(1, 2)
        loss, loc_loss, conf_loss = self.criterion(plocs, plabels, gloc_anchored, glabel_anchored)
        return {'loss' : loss, 
                'conf_loss' : conf_loss, 
                'loc_loss' : loc_loss, 
                'plocs' : plocs, 
                'plabels' : plabels, 
                'glocs' : glocs, 
                'glabels' : glabels,
                'img_ids' : img_ids
                }

    def pre_forward_step(self, glocs, glabels):
        glabels = [glabel + 1 for glabel in glabels]
        gloc_anchored, glabel_anchored = self.box_encoder.encode_batch(glocs, glabels)
        gloc_anchored = gloc_anchored.transpose(1, 2).contiguous()
        glabel_anchored = glabel_anchored.long()
        return gloc_anchored, glabel_anchored, glocs, glabels

    def training_step(self, batch, batch_idx):
        step_output = self.step(batch)

        self.log("train/loss", step_output['loss'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("train/conf_loss", step_output['conf_loss'], on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("train/loc_loss", step_output['loc_loss'], on_step=False, on_epoch=True, prog_bar=False, logger=True)
        return step_output['loss']

    def validation_step(self, batch, batch_idx):
        step_output = self.step(batch)

        self.log("val/loss", step_output['loss'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val/conf_loss", step_output['conf_loss'], on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("val/loc_loss", step_output['loc_loss'], on_step=False, on_epoch=True, prog_bar=False, logger=True)
