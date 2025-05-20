from .voc_yolo import YOLOVOCDataModule2012 as VOC2012
from .voc_yolo import YOLOVOCDataModule2007 as VOC2007

__all__ = [
    "VOC2012",
    "VOC2007"
]