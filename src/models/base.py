import hydra

from omegaconf import DictConfig
import numpy as np
import torch
from torch import nn
from torch.nn import Module
from torch.optim.optimizer import Optimizer
import pytorch_lightning as pl
from torchmetrics import MetricCollection

class Model(pl.LightningModule):
    """Base class for any machine learning model using Pytorch Lightning

    :param config: Config object
    :type config: Config
    """
    def __init__(self, model_config: DictConfig, **kwargs):
        super().__init__() 
        Module: self.network = hydra.utils.instantiate(model_config.network)
        Module: self.criterion = hydra.utils.instantiate(model_config.loss)
        metrics = MetricCollection([
            hydra.utils.instantiate(metric) for metric in model_config.metrics
        ])
        self.train_metrics = metrics.clone(prefix='train_')
        self.valid_metrics = metrics.clone(prefix='val_')
    
    def forward(self, x):
        return self.network(x) 

    def configure_optimizers(self):
        Optimizer: optimizer = hydra.utils.instantiate(model_config.optimizer, params = self.parameters())
        return optimizer      
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self.network(x)
        output = self.train_metrics(preds, y)
        self.log_dict(output)

    def training_step_end(self, batch_parts):
        return None

    def training_epoch_end(self, training_step_outputs):
        return None 

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self.network(x)
        output = self.train_metrics(preds, y)
        self.log_dict(output)

    def validation_step_end(self, batch_parts):
        return None

    def validation_epoch_end(self, validation_step_outputs):
        return None