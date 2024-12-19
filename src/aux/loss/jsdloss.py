import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules.loss import _Loss, _Reduction

class JSDLoss(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super().__init__(size_average, reduce, reduction)
            
    
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        p = input.log_softmax(-1)
        q = target.log_softmax(-1)
        m = 0.5 * (p + q)
        return F.kl_div(m, p, log_target=True) + F.kl_div(m, q, log_target=True)