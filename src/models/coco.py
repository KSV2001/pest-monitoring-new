import hydra

from omegaconf import DictConfig
from typing import List, Optional, Any
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Module
from torch.optim.optimizer import Optimizer
import pytorch_lightning as pl
from torchmetrics import MetricCollection
from src.data.COCO.utils import generate_dboxes

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
        self.criterion: Module = hydra.utils.instantiate(self.model_config.loss, dboxes = self.dboxes)
    
    def forward(self, x):
        return self.network(x) 

    def configure_optimizers(self):
        optimizer: Optimizer = hydra.utils.instantiate(self.model_config.optimizer, params = self.parameters())
        return optimizer      
    
    def step(self, batch: Any):
        image, gloc, glabel = batch
        ploc, plabel = self.forward(image)
        ploc, plabel = ploc.float(), plabel.float()
        gloc = gloc.transpose(1, 2).contiguous()
        
        try:
            loss = self.criterion(ploc, plabel, gloc, glabel)
        except:
            import ipdb; ipdb.set_trace()
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.step(batch)

        # output = self.train_metrics(out, y)
        self.log('train/loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        # self.log_dict(output, on_step=False, on_epoch=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch)

        # output = self.valid_metrics(out, y)
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        # self.log_dict(output, on_step=False, on_epoch=True)