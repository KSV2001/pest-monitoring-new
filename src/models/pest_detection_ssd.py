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
from src.data.utils import generate_dboxes, Encoder
from src.metrics.pascal_voc_evaluator import get_metrics

class Model(pl.LightningModule):
    """Base class for any machine learning model using Pytorch Lightning

    :param config: Config object
    :type config: Config
    """
    def __init__(self, model_config: DictConfig, **kwargs):
        super().__init__() 
        self.model_config = model_config
        self.dboxes = generate_dboxes(model=self.model_config.type)
        self.box_encoder = Encoder(dboxes = self.dboxes)
        
        self.network: Module = hydra.utils.instantiate(self.model_config.network)
        self.criterion: Module = hydra.utils.instantiate(self.model_config.loss, dboxes = self.dboxes)
    
    def forward(self, x):
        return self.network(x) 

    def configure_optimizers(self):
        optimizer: Optimizer = hydra.utils.instantiate(self.model_config.optimizer, params = self.parameters())
        return optimizer      
    
    def step(self, batch: Any):
        images, glocs, glabels, img_ids = batch['imgs'], batch['bbox_coords'], batch['bbox_classes'], batch['img_ids']
        gloc_anchored, glabel_anchored, glocs, glabels = self.pre_forward_step(glocs, glabels)
        plocs, plabels = self.forward(images)
        loss = self.criterion(plocs, plabels, gloc_anchored, glabel_anchored)
        return loss, glocs, glabels, plocs, plabels, img_ids

    def pre_forward_step(self, glocs, glabels):
        for glabel in glabels:
            if glabel.numel() > 0:
                glabel += 1 # Adding 1 to all classes as 0. is background class, specific to SSD
        gloc_anchored, glabel_anchored = self.box_encoder.encode_batch(glocs, glabels)
        gloc_anchored = gloc_anchored.transpose(1, 2).contiguous()
        glabel_anchored = glabel_anchored.long()
        return gloc_anchored, glabel_anchored, glocs, glabels

    def training_step(self, batch, batch_idx):
        loss, glocs, glabels, plocs, plabels, img_ids = self.step(batch)

        output = get_metrics(img_ids = img_ids,
                            ploc = plocs.detach().clone(), plabel = plabels.detach().clone(), gloc = glocs, glabel = glabels,
                            img_shape = (300, 300),
                            nms_threshold = 0.5,
                            max_num = 200,
                            iou_threshold = 0.2,
                            encoder = self.box_encoder)
        self.log('train/mAP', output['mAP'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('train/loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        # self.log_dict(output, on_step=False, on_epoch=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, glocs, glabels, plocs, plabels, img_ids = self.step(batch)

        output = get_metrics(img_ids = img_ids,
                            ploc = plocs.detach().clone(), plabel = plabels.detach().clone(), gloc = glocs, glabel = glabels,
                            img_shape = (300, 300),
                            nms_threshold = 0.5,
                            max_num = 200,
                            iou_threshold = 0.2,
                            encoder = self.box_encoder)
        self.log('val/mAP', output['mAP'], on_step=False, on_epoch=True, prog_bar=True, logger=True)        
        # output = self.valid_metrics(out, y)
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        # self.log_dict(output, on_step=False, on_epoch=True)