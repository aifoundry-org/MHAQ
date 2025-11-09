import torch
import torch.nn.functional as F

from typing import Tuple
from torch import nn, inf

from src.aux.types import QScheme
from src.quantization.gdnsq.gdnsq import Quantizer
from src.quantization.gdnsq.gdnsq_utils import QNMethod
from src.aux.qutils import is_biased


class NoisyLinear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
        qscheme: QScheme = QScheme.PER_TENSOR,
        log_s_init: float = -12,
        rand_noise: bool = False,
        qnmethod: QNMethod = QNMethod.AEWGS,
    ) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        self.qscheme = qscheme

        if self.qscheme == QScheme.PER_TENSOR:
            self.log_wght_s = nn.Parameter(
                torch.Tensor([log_s_init]), requires_grad=True
            )
        elif self.qscheme == QScheme.PER_CHANNEL:
            self.log_wght_s = nn.Parameter(
                torch.empty((out_features, 1)).fill_(log_s_init), requires_grad=True
            )

        self._noise_ratio = nn.Parameter(
            torch.Tensor(
                [
                    1,
                ]
            ),
            requires_grad=False,
        )
        self.Q = Quantizer(
            self, torch.exp2(self.log_wght_s), 0, -inf, inf, qnmethod=qnmethod
        )

        if self.qscheme == QScheme.PER_TENSOR:
            self.log_wght_s = nn.Parameter(
                torch.Tensor([log_s_init]), requires_grad=True
            )
        elif self.qscheme == QScheme.PER_CHANNEL:
            self.log_wght_s = nn.Parameter(
                torch.empty((out_features, 1, 1, 1)).fill_(log_s_init),
                requires_grad=True,
            )
        self.rand_noise = rand_noise

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        s = torch.exp2(self.log_wght_s)
        self.Q.scale = s
        self.Q.rnoise_ratio.data = (
            self._noise_ratio
            if self.rand_noise
            else torch.zeros_like(self._noise_ratio)
        )

        if self.qscheme == QScheme.PER_CHANNEL:
            min = self.weight.amin((1, 2, 3), keepdim=True)
        elif self.qscheme == QScheme.PER_TENSOR:
            min = self.weight.amin()
        self.Q.zero_point = min

        weight = self.Q.dequantize(self.Q.quantize(self.weight))

        return F.linear(input, weight, self.bias)

    def extra_repr(self) -> str:
        bias = is_biased(self)
        # log_wght_s = self.log_wght_s.item()
        # noise_ratio = self._noise_ratio.item()

        log_wght_s = self.log_wght_s
        noise_ratio = (
            self._noise_ratio
            if self.rand_noise
            else torch.zeros_like(self._noise_ratio)
        )

        return (
            f"in_features={self.in_features}, out_features={self.out_features}, bias={bias},\n"
            f"log_wght_s={log_wght_s}, noise_ratio={noise_ratio}"
        )
