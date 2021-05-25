import hydra
from torch.optim.optimizer import Optimizer

from omegaconf import DictConfig
import numpy as np
import torch
from torch import nn
import pytorch_lightning as pl

class Model(pl.LightningModule):
    """Base class for any machine learning model using Pytorch Lightning

    :param config: Config object
    :type config: Config
    """
    def __init__(self, model_config: DictConfig, network: nn.Module, optimizer: Optimizer):
        super().__init__() 
        nn.Module: self.network = hydra.utils.instantiate(model_config.network)
    
    def forward(self, x):
        return self.network(x) 

    def configure_optimizers(self):
        Optimizer: optimizer = hydra.utils.instantiate(model_config.optimizer, params = self.parameters())
        return optimizer      
        
    def training_step(self, batch, batch_idx):
        return None

    def training_step_end(self, batch_parts):
        return None

    def training_epoch_end(self, training_step_outputs):
        return None 

    def validation_step(self, batch, batch_idx):
        return None

    def validation_step_end(self, batch_parts):
        return None

    def validation_epoch_end(self, validation_step_outputs):
        return None