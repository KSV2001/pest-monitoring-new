import torch
import torch.nn as nn
from .base_detection_loss import *
from .loss import *

class BaseLoss(nn.Module):
    """
        Implements the loss as the sum of the followings:
        1. Confidence Loss: All labels, with hard negative mining
        2. Localization Loss: Only on positive labels
        Suppose input dboxes has the shape 8732x4
    """
    def __init__(
        self,
        detection_criterion, 
        validation_criterion,
        detection_weight = 1.,
        validation_weight = 1.,
        dboxes=None,
        ):
        super(BaseLoss, self).__init__()

        assert dboxes is not None if detection_criterion is not None else True, "Detection Loss requires default boxes"
        self.dboxes = dboxes

        self.detection_criterion = detection_criterion(self.dboxes)
        self.detection_weight = detection_weight
        self.validation_criterion = validation_criterion()
        self.validation_weight = validation_weight

        # Two factor are from following links
        # http://jany.st/post/2017-11-05-single-shot-detector-ssd-from-scratch-in-tensorflow.html


    def forward(self, pred_loc, pred_bclass, pred_vclass, true_loc, true_bclass, true_vclass):
        """
            pred_loc, pred_bclass: Nx4x8732, Nxlabel_numx8732
                predicted location and labels
            true_loc, true_bclass: Nx4x8732, Nx8732
                ground truth location and labels
        """
        detection_loss = self.detection_criterion(pred_loc, pred_bclass, true_loc, true_bclass)
        validation_loss = self.validation_criterion(pred_vclass, true_vclass)
        print(f'Detection:{detection_loss}, Validation: {validation_loss}')
        loss = self.detection_weight*detection_loss + self.validation_weight*validation_loss        

        return loss