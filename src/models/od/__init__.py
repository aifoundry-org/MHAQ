from .yolo_v11 import (
    yolo_v11_n,
    yolo_v11_t,
    yolo_v11_s,
    yolo_v11_m,
    yolo_v11_l,
    yolo_v11_x,
)


__all__ = [
    "yolo_v11_n",
    "yolo_v11_t",
    "yolo_v11_s",
    "yolo_v11_m",
    "yolo_v11_l",
    "yolo_v11_x",
]

YOLO_FAMILY = ["YOLOV11", *__all__]
