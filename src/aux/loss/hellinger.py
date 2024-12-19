import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules.loss import _Loss


class HellingerLoss(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super().__init__(size_average, reduce, reduction)            
    
    def forward(self, input: Tensor, target: Tensor) -> Tensor:             
        return F.mse_loss(input.softmax(-1).sqrt(), target.softmax(-1).sqrt())