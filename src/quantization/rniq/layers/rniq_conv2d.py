import torch

from typing import Tuple
from torch import nn, inf

from src.aux.types import QScheme
from src.quantization.rniq.rniq import Quantizer

from src.aux.qutils import attrsetter, is_biased


class NoisyConv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Tuple[int, int],
        stride: int | Tuple[int, int] = 1,
        padding: str | int | Tuple[int, int] = 0,
        dilation: int | Tuple[int, int] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
        qscheme: QScheme = QScheme.PER_TENSOR,
        log_s_init: float = -12,
        rand_noise: bool = False
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            device,
            dtype,
        )
        self.qscheme = qscheme

        if self.qscheme == QScheme.PER_TENSOR:
            self.log_wght_s = nn.Parameter(
                torch.Tensor([log_s_init]), requires_grad=True)
        elif self.qscheme == QScheme.PER_CHANNEL:
            self.log_wght_s = nn.Parameter(
                torch.empty((out_channels, 1, 1, 1)).fill_(log_s_init),
                requires_grad=True,
            )
        self._noise_ratio = torch.nn.Parameter(torch.Tensor([1]), requires_grad=False)
        self.Q = Quantizer(self, torch.exp2(self.log_wght_s), 0, -inf, inf)
        self.rand_noise = rand_noise

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        s = torch.exp2(self.log_wght_s)
        self.Q.scale = s
        self.Q.rnoise_ratio.data = self._noise_ratio if self.rand_noise else torch.zeros_like(self._noise_ratio)

        if self.qscheme == QScheme.PER_CHANNEL:
            min = self.weight.amin()
        elif self.qscheme == QScheme.PER_TENSOR:
            min = self.weight.amin((1,2,3))
        self.Q.zero_point = min

        weight = self.Q.dequantize(self.Q.quantize(self.weight))

        return self._conv_forward(input, weight, self.bias)

    def extra_repr(self) -> str:
        bias = is_biased(self)
        # log_wght_s = self.log_wght_s.item()
        # noise_ratio = self._noise_ratio.item()

        log_wght_s = self.log_wght_s
        noise_ratio = self._noise_ratio if self.rand_noise else torch.zeros_like(self._noise_ratio)
        
        return (
            f"in_channels={self.in_channels}, out_channels={self.out_channels}, kernel_size={self.kernel_size},\n"
            f"stride={self.stride}, padding={self.padding}, dilation={self.dilation},\n"
            f"groups={self.groups}, bias={bias}, log_wght_s_mean={log_wght_s.mean()},\n"
            f"noise_ratio={noise_ratio}"
        )
