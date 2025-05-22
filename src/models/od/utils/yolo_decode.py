from torch import Tensor
from typing import List, Dict

def decode_yolo_nms(output: List[Tensor]) -> List[Dict[str, Tensor]]:
    return [
        {
            "boxes":  boxes,               # [Ni,4]
            "scores": scores.squeeze(1),   # [Ni]
            "labels": labels.squeeze(1).long()  # [Ni]
        }
        for out in output
        for boxes, scores, labels in (out.split([4,1,1], dim=1),)
    ]