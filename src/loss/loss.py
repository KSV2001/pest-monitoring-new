import torch
import torch.nn as nn

class DetectionSmoothL1Loss(nn.Module):
    def __init__(self, *args, **kwargs):
        super(DetectionSmoothL1Loss, self).__init__()
        self.criterion = nn.SmoothL1Loss(reduce=False, *args, **kwargs)

    def forward(self, pred_loc, pred_bclass, true_loc_vec, true_bclass):
        mask = true_bclass > 0

        # sum on four coordinates, and mask
        loss = self.criterion(pred_loc, true_loc_vec).sum(dim=1)
        masked_loss = (mask.float()*loss).sum(dim=1)

        return masked_loss

class DetectionHardMinedCELoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super(DetectionHardMinedCELoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss(reduce=False, *args, **kwargs)

    def forward(self, pred_loc, pred_bclass, true_loc_vec, true_bclass):
        mask = true_bclass > 0
        pos_num = mask.sum(dim=1)

        # hard negative mining
        loss = self.criterion(pred_bclass, true_bclass)

        # postive mask will never selected
        con_neg = loss.clone()
        con_neg[mask] = 0
        _, con_idx = con_neg.sort(dim=1, descending=True)
        _, con_rank = con_idx.sort(dim=1)

        # number of negative three times positive
        neg_num = torch.clamp(3*pos_num, max=mask.size(1)).unsqueeze(-1)
        neg_mask = con_rank < neg_num

        #print(con.shape, mask.shape, neg_mask.shape)

        # TODO: Confirm that positive class is counted only once
        # hard_mined_loss = (loss*(mask.float() + neg_mask.float())).sum(dim=1)
        hard_mined_loss = (loss*(mask + neg_mask).float()).sum(dim=1)


        return hard_mined_loss

class baseValidationLoss(nn.Module):
    """
        Implements the loss as the sum of the followings:
        1. Confidence Loss: All labels, with hard negative mining
        2. Localization Loss: Only on positive labels
        Suppose input dboxes has the shape 8732x4
    """
    def __init__(self, *args, **kwargs):
        super(baseValidationLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss(*args, **kwargs)

    def forward(self, pred_vclass, true_vclass):

        loss = self.criterion(pred_vclass, true_vclass)

        return loss