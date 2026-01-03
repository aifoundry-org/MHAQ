import torch
from torch import nn, inf
import torch.nn.functional as F

from src.aux.types import QScheme
from src.quantization.rniq.rniq import Quantizer
from src.quantization.rniq.utils.enums import QMode

def softclamp(x, l, h, s_l, s_h):
    # s must be > 0. If you want to train it, pass s = softplus(s_raw) + eps
    s_l = F.softplus(s_l)
    s_h = F.softplus(s_h)

    return l + F.softplus(s_l*(x - l)) / s_l - F.softplus(s_h*(x - h)) / s_h


class NoisyAct(nn.Module):
    def __init__(self, init_s=-10, init_q=10, signed=True, noise_ratio=1, disable=False) -> None:
        super().__init__()
        self.disable = disable
        self.signed = signed
        self._act_b = torch.tensor([0]).float()
        self._log_act_s = torch.tensor([init_s]).float()
        self._log_act_q = torch.tensor([init_q]).float()
        self._noise_ratio = torch.tensor(noise_ratio)
        #self._act_smooth_l = torch.tensor([50.0]).float()
        #self._act_smooth_h = torch.tensor([50.0]).float()

        self.log_act_q = torch.nn.Parameter(self._log_act_q, requires_grad=True)
        if signed:
            self.act_b = torch.nn.Parameter(self._act_b, requires_grad=True)
        else:
            self.act_b = torch.nn.Parameter(self._act_b, requires_grad=False)

        self.log_act_s = torch.nn.Parameter(self._log_act_s, requires_grad=True)
        #self.act_smooth_l = torch.nn.Parameter(self._act_smooth_l, requires_grad=True)
        #self.act_smooth_h = torch.nn.Parameter(self._act_smooth_h, requires_grad=True)
        self.Q = Quantizer(self, torch.exp2(self._log_act_s), 0, -inf, inf)
        self.bw = torch.tensor(0.0)

    def forward(self, x):
        if self.disable:
            return x
        s = torch.exp2(self.log_act_s)
        q = torch.exp2(self.log_act_q)
        
        self.Q.zero_point = self.act_b
        self.Q.min_val = self.act_b
        self.Q.max_val = self.act_b + q - s
        self.Q.scale = s

        #x = softclamp(x, self.Q.min_val, self.Q.max_val, self.act_smooth_l, self.act_smooth_h)

        q = self.Q.quantize(x)
        if not self.training: 
            # assume q is int
            minmax = q.aminmax()
            self.bw = torch.log2(minmax.max - minmax.min + 1)
        return self.Q.dequantize(q)
