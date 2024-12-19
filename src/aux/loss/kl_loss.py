import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules.loss import _Loss, _Reduction

class KL(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super().__init__(size_average, reduce, reduction)
            
    
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        x = F.log_softmax(input, dim=1)
        y = F.log_softmax(target, dim=1)
        return F.kl_div(x, y, log_target=True)