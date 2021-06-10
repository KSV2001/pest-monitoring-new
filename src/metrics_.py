import torch
from torchmetrics import Metric
from typing import List
import numpy as np

from src.metrics.bounding_box import BoundingBox
from src.metrics.enumerators import BBType, CoordinatesType, BBFormat, MethodAveragePrecision
from src.metrics.pascal_voc_evaluator import get_pascalvoc_metrics

# example metric
class ObjectDetectionMetrics(Metric):
    '''ObjectDetectionMetrics
    PytorchLightning support metrics to compute PascalVocMetrics, as implemented in 
    (https://github.com/rafaelpadilla/review_object_detection_metrics)
    Parameters
        ----------
            iou_threshold : float
                IOU threshold indicating which detections will be considered TP or FP
            img_size : int
                Integer value representing the size of an Image, assuming image shape to be (img_size, img_size)
            type_coordinates : str
                str representing if the bounding box coordinates (x,y,w,h) are ABSOLUTE or RELATIVE
                to size of the image. Default:'ABSOLUTE'.
            bb_format : str (optional)
                BBFormat.XYWH: <left> <top> <width> <height>
                BBFomat.XYX2Y2: <left> <top> <right> <bottom>.
            method : str (optional)
                Two Methods in PascalVocMetrics to evaluate Object Detection Average Precision
    '''
    def __init__(self,
                dist_sync_on_step: bool = False,
                iou_threshold: float = 0.5,
                img_size: int = 300, 
                type_coordinates: str = 'RELATIVE',
                bb_format: str = 'XYX2Y2',
                method: str = 'EVERY_POINT_INTERPOLATION'):
        # call `self.add_state`for every internal state that is needed for the metrics
        # computations dist_reduce_fx indicates the function that should be used to reduce
        # state from multiple processes
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        assert type_coordinates in ['RELATIVE', 'ABSOLUTE'], 'type_coordinates should be either RELATIVE or ABSOLUTE'
        assert bb_format in ['XYX2Y2', 'XYWH'], 'BBFormat should be in XYX2Y2 or XYWH'
        assert method in ['EVERY_POINT_INTERPOLATION', 'ELEVEN_POINT_INTERPOLATION'], 'method should be either \
        EVERY_POINT_INTERPOLATION or ELEVEN_POINT_INTERPOLATION'

        self.iou_threshold = iou_threshold
        self.img_shape = (img_size, img_size)
        self.type_coordinates = eval(f'CoordinatesType.{type_coordinates}')
        self.bb_format = eval(f'BBFormat.{bb_format}')
        self.method = eval(f'MethodAveragePrecision.{method}')

        self.add_state("plocs", default=[], dist_reduce_fx=None)
        self.add_state("plabels", default=[], dist_reduce_fx=None)
        self.add_state("pscores", default=[], dist_reduce_fx=None)
        self.add_state("glocs", default=[], dist_reduce_fx=None)
        self.add_state("glabels", default=[], dist_reduce_fx=None)
        self.add_state("img_ids", default=[], dist_reduce_fx=None)

    def update(self, plocs: List[torch.Tensor], plabels: List[torch.Tensor], pscores: List[torch.Tensor],
                glocs: List[torch.Tensor], glabels: List[torch.Tensor], img_ids: List[str]):

        self.plocs.extend(plocs)
        self.plabels.extend(plabels)
        self.pscores.extend(pscores)
        self.glocs.extend(glocs)
        self.glabels.extend(glabels)
        self.img_ids.extend(img_ids)
        self.n = len(self.img_ids)

    def get_boxes(self):
        gt_boxes = []
        pd_boxes = []
        for idx in range(self.n):
            gloc_i = self.glocs[idx]
            glabel_i = self.glabels[idx]

            if glabel_i.numel() != 0:            
                gloc_i, glabel_i = gloc_i.detach().cpu().numpy(), glabel_i.detach().cpu().numpy().astype(np.int)
                for loc_, label_ in zip(gloc_i, glabel_i):
                    gt_boxes.append(BoundingBox(image_name = self.img_ids[idx],
                            class_id=str(label_),
                            coordinates=(loc_[0], loc_[1], loc_[2], loc_[3]),
                            type_coordinates=self.type_coordinates,
                            img_size=self.img_shape,
                            bb_type=BBType.GROUND_TRUTH,
                            format=self.bb_format)) 

            loc_, label_, prob_ = self.plocs[idx].detach().cpu().numpy(), \
                                self.plabels[idx].detach().cpu().numpy().astype(np.int),\
                                self.pscores[idx].cpu().detach().numpy()
            for i in range(loc_.shape[0]):
                bbox = BoundingBox(image_name = self.img_ids[idx], 
                                class_id = str(label_[i]),
                        coordinates=(loc_[i][0], loc_[i][1], loc_[i][2], loc_[i][3]),
                        type_coordinates=self.type_coordinates,
                        img_size=self.img_shape,
                        bb_type=BBType.DETECTED,
                        confidence=prob_[i],
                        format=self.bb_format)
                pd_boxes.append(bbox)      
        return gt_boxes, pd_boxes

    def compute(self):
        gt_boxes, pd_boxes = self.get_boxes()
        output = get_pascalvoc_metrics(
                        gt_boxes,
                        pd_boxes,
                        iou_threshold=self.iou_threshold,
                        method=self.method,
                        generate_table=False,
                )
        return {'mAP' : output['mAP']}
