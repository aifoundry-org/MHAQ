import torch

from typing import Tuple
from torch import nn

from src.aux.types import QScheme
from src.quantization.gdnsq.gdnsq_utils import QNMethod
from src.quantization.gdnsq.layers.gdnsq_act_lin import NoisyActLin

from src.aux.qutils import is_biased


class NoisyConv2d(NoisyActLin, nn.Conv2d):
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
        rand_noise: bool = False,
        quant_bias: bool = False,
        disable: bool = False,
        act_init_s: float = -10,
        act_init_q: float = 10,
        qnmethod: QNMethod = QNMethod.AEWGS,
        weight_guard_bit: int = 0,
        act_guard_bit: int = 0,
    ) -> None:
        nn.Conv2d.__init__(
            self,
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
        self.quant_bias = bool(quant_bias)
        # 'signed' is kept for backward compatibility but currently ignored.
        self._init_noisy_actlin(
            qscheme=qscheme,
            log_s_init=log_s_init,
            rand_noise=rand_noise,
            disable=disable,
            act_init_s=act_init_s,
            act_init_q=act_init_q,
            qnmethod=qnmethod,
            per_channel_shape=(out_channels, 1, 1, 1),
            weight_guard_bit=weight_guard_bit,
            act_guard_bit=act_guard_bit,
        )

    def _weight_quantization_dims(self) -> tuple[int, ...]:
        return (1, 2, 3)

    def _apply_affine(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self._conv_forward(input, weight, bias)

    def _get_affine_bias(self) -> torch.Tensor | None:
        return self.bias

    def extra_repr(self) -> str:
        bias = is_biased(self)
        # log_wght_s = self.log_wght_s.item()

        log_wght_s = self.log_wght_s

        return (
            f"in_channels={self.in_channels}, out_channels={self.out_channels}, kernel_size={self.kernel_size},\n"
            f"stride={self.stride}, padding={self.padding}, dilation={self.dilation},\n"
            f"groups={self.groups}, bias={bias}, log_wght_s_mean={log_wght_s.mean()},\n"
            f"quantized_bias={self.quant_bias}"
        )
