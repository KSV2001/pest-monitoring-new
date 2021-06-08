from typing import Any

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from torch.nn import Module
from torch.optim.optimizer import Optimizer

from src.data.utils import Encoder, generate_dboxes
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
        self.box_encoder = Encoder(dboxes=self.dboxes)

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
        image, gloc_, glabel_, img_ids = (
            batch["imgs"],
            batch["bbox_coords"],
            batch["bbox_classes"],
            batch["img_ids"],
        )
        gloc, glabel = self.box_encoder.encode_batch(gloc_, glabel_)
        ploc, plabel = self.forward(image)
        loss = self.criterion(ploc, plabel, gloc, glabel)
        return loss, gloc_, glabel_, img_ids, ploc, plabel

    def training_step(self, batch, batch_idx):
        loss, gloc_, glabel_, img_ids, ploc, plabel = self.step(batch)

        output = get_metrics(
            img_ids=img_ids,
            ploc=ploc.detach().clone(),
            plabel=plabel.detach().clone(),
            gloc=gloc_,
            glabel=glabel_,
            img_shape=(300, 300),
            nms_threshold=0.5,
            max_num=200,
            iou_threshold=0.5,
            encoder=self.box_encoder,
        )
        self.log(
            "train/mAP", output["mAP"], on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        # self.log_dict(output, on_step=False, on_epoch=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, gloc_, glabel_, img_ids, ploc, plabel = self.step(batch)

        output = get_metrics(
            img_ids=img_ids,
            ploc=ploc.detach().clone(),
            plabel=plabel.detach().clone(),
            gloc=gloc_,
            glabel=glabel_,
            img_shape=(300, 300),
            nms_threshold=0.5,
            max_num=200,
            iou_threshold=0.5,
            encoder=self.box_encoder,
        )
        self.log("val/mAP", output["mAP"], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        # output = self.valid_metrics(out, y)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        # self.log_dict(output, on_step=False, on_epoch=True)
