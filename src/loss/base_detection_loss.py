import torch
import torch.nn as nn

from .loss import DetectionHardMinedCELoss, DetectionSmoothL1Loss


class BaseDetectionLoss(nn.Module):
    """
    Implements the loss as the sum of the followings:
    1. Confidence Loss: All labels, with hard negative mining
    2. Localization Loss: Only on positive labels
    Suppose input dboxes has the shape 8732x4
    """

    def __init__(self, dboxes):
        super(BaseDetectionLoss, self).__init__()
        self.scale_xy = 1.0 / dboxes.scale_xy
        self.scale_wh = 1.0 / dboxes.scale_wh

        self.loc_criterion = DetectionSmoothL1Loss()
        self.loc_weight = 1.0
        self.conf_criterion = DetectionHardMinedCELoss()
        self.conf_weight = 1.0

        self.dboxes = nn.Parameter(
            dboxes(order="xywh").transpose(0, 1).unsqueeze(dim=0), requires_grad=False
        )
        # Two factor are from following links
        # http://jany.st/post/2017-11-05-single-shot-detector-ssd-from-scratch-in-tensorflow.html

    def _loc_vec(self, loc):
        """
        Generate Location Vectors
        """
        gxy = (
            self.scale_xy
            * (loc[:, :2, :] - self.dboxes[:, :2, :])
            / self.dboxes[
                :,
                2:,
            ]
        )
        gwh = self.scale_wh * (loc[:, 2:, :] / self.dboxes[:, 2:, :]).log()
        return torch.cat((gxy, gwh), dim=1).contiguous()

    def reduce(self, loss, mask):
        pos_num = mask.sum(dim=1)
        num_mask = (pos_num > 0).float()
        pos_num = pos_num.float().clamp(min=1e-6)
        red_loss = (loss * num_mask / pos_num).mean(dim=0)
        return red_loss

    def forward(self, pred_loc, pred_bclass, true_loc, true_bclass):
        """
        pred_loc, pred_bclass: Nx4x8732, Nxlabel_numx8732
            predicted location and labels
        true_loc, true_bclass: Nx4x8732, Nx8732
            ground truth location and labels
        """
        mask = true_bclass > 0

        true_loc_vec = self._loc_vec(true_loc)

        # sum on four coordinates, and mask
        loc_loss = self.loc_criterion(pred_loc, pred_bclass, true_loc_vec, true_bclass)

        # hard negative mining on conf loss
        conf_loss = self.conf_criterion(pred_loc, pred_bclass, true_loc_vec, true_bclass)

        # avoid no object detected
        total_loss = self.loc_weight * loc_loss + self.conf_weight * conf_loss
        red_loss = self.reduce(total_loss, mask)
        red_loc_loss = self.reduce(loc_loss, mask)
        red_conf_loss = self.reduce(conf_loss, mask)

        return red_loss, red_loc_loss, red_conf_loss
