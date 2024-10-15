import torch
from torch import Tensor
from torch.nn.modules.loss import _Loss, _Reduction

class HellingerLoss(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super().__init__(size_average, reduce, reduction)
    
    def _distance(self, p, q):
        # p = torch.clamp(p, min=1e-10)
        # q = torch.clamp(q, min=1e-10)
        
        # shift_value = min(p.min(), q.min()).abs() + 1e-10
        # p = p + shift_value
        # q = q + shift_value
        return torch.sqrt(torch.sum((torch.sqrt(q.softmax(-1)) - torch.sqrt(p.softmax(-1))) ** 2, dim=-1)) / torch.sqrt(torch.tensor(2.0))
        
    
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return torch.mean(self._distance(input, target))