import torchmetrics

from typing import List, Dict
from torch import Tensor

from torchmetrics.detection import MeanAveragePrecision
from src.models.od.utils.yolo_decode import decode_yolo_nms
from src.models.od.utils.yolo_nms import non_max_suppression


class MeanAveragePrecisionYolo(MeanAveragePrecision):
    def __init__(
        self,
        box_format="xyxy",
        iou_type="bbox",
        iou_thresholds=None,
        rec_thresholds=None,
        max_detection_thresholds=None,
        class_metrics=False,
        extended_summary=False,
        average="macro",
        backend="pycocotools",
        confidence_threshold=0.001,
        **kwargs
    ):
        super().__init__(
            box_format,
            iou_type,
            iou_thresholds,
            rec_thresholds,
            max_detection_thresholds,
            class_metrics,
            extended_summary,
            average,
            backend,
            **kwargs
        )
        self.confidence_threshold = confidence_threshold

    def update(self, preds, target):
        if not isinstance(preds[0], Dict):
            preds = decode_yolo_nms(
                non_max_suppression(
                    preds, confidence_threshold=self.confidence_threshold
                )
            )
        return super().update(preds, target)
