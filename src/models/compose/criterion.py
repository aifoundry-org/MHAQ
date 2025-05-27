from src.models.od.loss.yolo_loss import ComputeYoloLoss as yolo_loss
from torch.nn import CrossEntropyLoss
from torch import nn

def get_criterion(criterion_name:str, **kwargs):
    if criterion_name == "yolo_loss":
        params = {"box": 7.5, "cls": 0.5, "dfl": 1.5}
        return yolo_loss(kwargs["model"], params)
    else:
        return getattr(nn, criterion_name)()
    