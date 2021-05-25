import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl

class Model(pl.LightningModule):
    """Base class for any machine learning model using Pytorch Lightning

    :param config: Config object
    :type config: Config
    """
    def __init__(self, config):
        super().__init__()
        self.setup_configs(config)
        self.setup_network()
    
    def _setup_configs(self, config):
        self.network_config = config.network
        self.model_config = config.model

    def _setup_network(self):
        # TODO : Insert Network 

    def _setup_optimizers(self, model_config):
        return None

    def forward(self):
        raise NotImplementedError

    def configure_optimizers(self):
        return _setup_optimizers(self.model_config)        
        
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