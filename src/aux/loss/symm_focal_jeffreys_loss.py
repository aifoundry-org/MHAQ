import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules.loss import _Loss

class SymmetricFocalJeffreys(_Loss):
    def __init__(self, T: float = 1.0, beta = 0, gamma: float = 10, eps: float = 1e-8):
        super().__init__()
        self.T, self.beta, self.gamma, self.eps = T, beta, gamma, eps

    def forward(self, student_logits: Tensor, teacher_logits: Tensor, mask: Tensor | None = None) -> Tensor:
        log_pT = F.log_softmax(teacher_logits / self.T, dim=-1)
        log_q  = F.log_softmax(student_logits, dim=-1)
        pT = log_pT.exp() #.clamp_min(self.eps)
        q  = log_q.exp() #.clamp_min(self.eps)

        # Jeffreys per-sample
        j = (pT * (log_pT - log_q) + q * (log_q - log_pT)).sum(dim=-1)  # [B]

        # symmetric focal weight
        with torch.no_grad():
            conf_t = pT.max(dim=-1).values  # teacher confidence
            conf_s = q.max(dim=-1).values   # student confidence
            w = ((1.0 - conf_t).clamp_min(0.0).pow(self.gamma) + (1.0 - conf_s).clamp_min(0.0).pow(self.gamma)).pow(1/self.gamma)
            if mask is not None:
                w = w * mask.to(w.dtype)

        denom = w.sum().clamp_min(self.eps)
        return (1-self.beta) * j.mean() + self.beta * (w * j).sum() / denom
