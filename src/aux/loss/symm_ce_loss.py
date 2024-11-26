import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules.loss import _Loss, _Reduction


class SymmetricalCrossEntropyLoss(_Loss):
    def __init__(self):
        super(SymmetricalCrossEntropyLoss, self).__init__()

    def forward(self, logits, targets):
        loss = torch.sum(F.softmax(targets, dim=1) * F.log_softmax(logits, dim=1), dim=1).mean() + \
            torch.sum(F.softmax(logits, dim=1) * F.log_softmax(targets, dim=1), dim=1).mean()
        return -loss