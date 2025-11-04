import torch

from torchvision.transforms.v2 import functional as F
from typing import Tuple
from torch import Tensor


class ResizeWithPadding(torch.nn.Module):
    def __init__(self, size: Tuple, *args, **kwargs):
        self.size = size
        super().__init__(*args, **kwargs)

    def forward(self, img: Tensor, label):
        shape = img.shape[1:]

        r = min(self.size[0] / shape[0], self.size[1] / shape[1])
        r = min(r, 1.0)

        pad = int(round(shape[1] * r)), int(round(shape[0] * r))
        w = (self.size[0] - pad[0]) / 2
        h = (self.size[1] - pad[1]) / 2

        if shape[::-1] != pad:
            img = F.resize_image(
                image=img, size=pad, interpolation=F.InterpolationMode.BILINEAR
            )
        top, bottom = int(round(h - 0.1)), int(round(h + 0.1))
        left, right = int(round(w - 0.1)), int(round(w + 0.1))
        img = F.pad(img, [left, top, right, bottom], 0)

        if "boxes" in label:
            boxes = label["boxes"]
            label["boxes"].data, label["boxes"].canvas_size = F.pad_bounding_boxes(boxes.data, boxes.format, boxes.canvas_size, [left, top, right, bottom])

        return img, label
