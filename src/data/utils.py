import itertools
from math import sqrt

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.ops.boxes import box_convert, box_iou


class Encoder(object):
    """
    Inspired by https://github.com/kuangliu/pytorch-src
    Transform between (bboxes, labels) <-> SSD output
    dboxes: default boxes in size 8732 x 4,
        encoder: input ltrb format, output xywh format
        decoder: input xywh format, output ltrb format
    encode:
        input  : bboxes_in (Tensor nboxes x 4), labels_in (Tensor nboxes)
        output : bboxes_out (Tensor 8732 x 4), labels_out (Tensor 8732)
        criteria : IoU threshold of bboexes
    decode:
        input  : bboxes_in (Tensor 8732 x 4), scores_in (Tensor 8732 x nitems)
        output : bboxes_out (Tensor nboxes x 4), labels_out (Tensor nboxes)
        criteria : IoU threshold of bboexes
        max_output : maximum number of output bboxes
    """

    def __init__(self, dboxes):
        self.dboxes = dboxes(order="ltrb")
        self.dboxes_xywh = dboxes(order="xywh").unsqueeze(dim=0)
        self.nboxes = self.dboxes.size(0)
        self.scale_xy = dboxes.scale_xy
        self.scale_wh = dboxes.scale_wh

    def encode(self, bboxes_in, labels_in, criteria=0.5):
        ious = box_iou(bboxes_in, self.dboxes)
        best_dbox_ious, best_dbox_idx = ious.max(dim=0)
        best_bbox_ious, best_bbox_idx = ious.max(dim=1)

        # set best ious 2.0
        best_dbox_ious.index_fill_(0, best_bbox_idx, 2.0)

        idx = torch.arange(0, best_bbox_idx.size(0), dtype=torch.int64)
        best_dbox_idx[best_bbox_idx[idx]] = idx

        # filter IoU > 0.5
        masks = best_dbox_ious > criteria
        labels_out = torch.zeros(self.nboxes, dtype=torch.float)
        labels_out[masks] = labels_in[best_dbox_idx[masks]]
        bboxes_out = self.dboxes.clone()
        bboxes_out[masks, :] = bboxes_in[best_dbox_idx[masks], :]
        bboxes_out = box_convert(bboxes_out, in_fmt="xyxy", out_fmt="cxcywh")
        return bboxes_out, labels_out

    def encode_batch(self, bboxes_in_batch, labels_in_batch, criteria=0.5):
        bboxes_out_batch, labels_out_batch = [], []
        for bboxes_in, labels_in in zip(bboxes_in_batch, labels_in_batch):
            if labels_in.numel() != 0:
                bboxes_out, labels_out = self.encode(bboxes_in, labels_in)
                bboxes_out_batch.append(bboxes_out)
                labels_out_batch.append(labels_out)
            else:
                bboxes_out = self.dboxes.clone()
                bboxes_out = box_convert(bboxes_out, in_fmt="xyxy", out_fmt="cxcywh")
                bboxes_out_batch.append(bboxes_out)
                labels_out_batch.append(torch.zeros(self.nboxes, dtype=torch.long))

        return (
            torch.stack(bboxes_out_batch, 0).transpose(1, 2).contiguous(),
            torch.stack(labels_out_batch, 0).long(),
        )

    def scale_back_batch(self, bboxes_in, scores_in):
        """
        Do scale and transform from xywh to ltrb
        suppose input Nx4xnum_bbox Nxlabel_numxnum_bbox
        """
        self.dboxes = self.dboxes
        self.dboxes_xywh = self.dboxes_xywh

        bboxes_in = bboxes_in.permute(0, 2, 1)
        scores_in = scores_in.permute(0, 2, 1)

        bboxes_in[:, :, :2] = self.scale_xy * bboxes_in[:, :, :2]
        bboxes_in[:, :, 2:] = self.scale_wh * bboxes_in[:, :, 2:]

        bboxes_in[:, :, :2] = (
            bboxes_in[:, :, :2] * self.dboxes_xywh[:, :, 2:] + self.dboxes_xywh[:, :, :2]
        )
        bboxes_in[:, :, 2:] = bboxes_in[:, :, 2:].exp() * self.dboxes_xywh[:, :, 2:]
        bboxes_in = box_convert(bboxes_in, in_fmt="cxcywh", out_fmt="xyxy")

        return bboxes_in, F.softmax(scores_in, dim=-1)

    def decode_batch(self, bboxes_in, scores_in, nms_threshold=0.45, max_output=200):
        bboxes, probs = self.scale_back_batch(bboxes_in, scores_in)
        output = []
        for bbox, prob in zip(bboxes.split(1, 0), probs.split(1, 0)):
            bbox = bbox.squeeze(0)
            prob = prob.squeeze(0)
            output.append(self.decode_single(bbox, prob, nms_threshold, max_output))
        return output

    def decode_single(self, bboxes_in, scores_in, nms_threshold, max_output, max_num=200):
        bboxes_out = []
        scores_out = []
        labels_out = []

        for i, score in enumerate(scores_in.split(1, 1)):
            if i == 0:
                continue

            score = score.squeeze(1)
            mask = score > 0.05

            bboxes, score = bboxes_in[mask, :], score[mask]
            if score.size(0) == 0:
                continue

            score_sorted, score_idx_sorted = score.sort(dim=0)

            # select max_output indices
            score_idx_sorted = score_idx_sorted[-max_num:]
            candidates = []

            while score_idx_sorted.numel() > 0:
                idx = score_idx_sorted[-1].item()
                bboxes_sorted = bboxes[score_idx_sorted, :]
                bboxes_idx = bboxes[idx, :].unsqueeze(dim=0)
                iou_sorted = box_iou(bboxes_sorted, bboxes_idx).squeeze()
                # we only need iou < nms_threshold
                score_idx_sorted = score_idx_sorted[iou_sorted < nms_threshold]
                candidates.append(idx)

            bboxes_out.append(bboxes[candidates, :])
            scores_out.append(score[candidates])
            labels_out.extend([i] * len(candidates))

        if not bboxes_out:
            return [torch.tensor([]) for _ in range(3)]

        bboxes_out, labels_out, scores_out = (
            torch.cat(bboxes_out, dim=0),
            torch.tensor(labels_out, dtype=torch.long),
            torch.cat(scores_out, dim=0),
        )

        _, max_ids = scores_out.sort(dim=0)
        max_ids = max_ids[-max_output:]
        return bboxes_out[max_ids, :], labels_out[max_ids], scores_out[max_ids]


class DefaultBoxes(object):
    def __init__(
        self, fig_size, feat_size, steps, scales, aspect_ratios, scale_xy=0.1, scale_wh=0.2
    ):

        self.feat_size = feat_size
        self.fig_size = fig_size

        self.scale_xy = scale_xy
        self.scale_wh = scale_wh

        self.steps = steps
        self.scales = scales

        fk = fig_size / np.array(steps)
        self.aspect_ratios = aspect_ratios

        self.default_boxes = []
        for idx, sfeat in enumerate(self.feat_size):

            sk1 = scales[idx] / fig_size
            sk2 = scales[idx + 1] / fig_size
            sk3 = sqrt(sk1 * sk2)
            all_sizes = [(sk1, sk1), (sk3, sk3)]

            for alpha in aspect_ratios[idx]:
                w, h = sk1 * sqrt(alpha), sk1 / sqrt(alpha)
                all_sizes.append((w, h))
                all_sizes.append((h, w))
            for w, h in all_sizes:
                for i, j in itertools.product(range(sfeat), repeat=2):
                    cx, cy = (j + 0.5) / fk[idx], (i + 0.5) / fk[idx]
                    self.default_boxes.append((cx, cy, w, h))

        self.dboxes = torch.tensor(self.default_boxes, dtype=torch.float)
        self.dboxes.clamp_(min=0, max=1)
        self.dboxes_ltrb = box_convert(self.dboxes, in_fmt="cxcywh", out_fmt="xyxy")

    def __call__(self, order="ltrb"):
        if order == "ltrb":
            return self.dboxes_ltrb
        else:  # order == "xywh"
            return self.dboxes


def generate_dboxes(model="ssd"):
    if model == "ssd":
        figsize = 300
        feat_size = [38, 19, 10, 5, 3, 1]
        steps = [8, 16, 32, 64, 100, 300]
        scales = [21, 45, 99, 153, 207, 261, 315]
        aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
        dboxes = DefaultBoxes(figsize, feat_size, steps, scales, aspect_ratios)
    else:  # "ssdlite"
        figsize = 300
        feat_size = [19, 10, 5, 3, 2, 1]
        steps = [16, 32, 64, 100, 150, 300]
        scales = [60, 105, 150, 195, 240, 285, 330]
        aspect_ratios = [[2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3]]
        dboxes = DefaultBoxes(figsize, feat_size, steps, scales, aspect_ratios)
    return dboxes
