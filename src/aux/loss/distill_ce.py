from torch import Tensor
import torch.nn as nn

class CrossEntropyLoss(nn.CrossEntropyLoss):
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return super().forward(input, target.softmax(-1))