import torch
import torch.nn.functional as F

from torch import nn

from src.aux.types import QScheme
from src.quantization.gdnsq.gdnsq_utils import QNMethod
from src.quantization.gdnsq.layers.gdnsq_act_lin import NoisyActLin
from src.aux.qutils import is_biased


class NoisyLinear(NoisyActLin, nn.Linear):
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
        disable: bool = False,
        act_init_s: float = -10,
        act_init_q: float = 10,
        qnmethod: QNMethod = QNMethod.STE,
        weight_guard_bit: int = 0,
        act_guard_bit: int = 0,
    ) -> None:
        nn.Linear.__init__(self, in_features, out_features, bias, device, dtype)
        # 'signed' is kept for backward compatibility but currently ignored.
        self._init_noisy_actlin(
            qscheme=qscheme,
            log_s_init=log_s_init,
            rand_noise=rand_noise,
            disable=disable,
            act_init_s=act_init_s,
            act_init_q=act_init_q,
            qnmethod=qnmethod,
            per_channel_shape=(out_features, 1),
            weight_guard_bit=weight_guard_bit,
            act_guard_bit=act_guard_bit,
        )

    def _weight_quantization_dims(self) -> tuple[int, ...]:
        return (1,)

    def _apply_affine(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return F.linear(input, weight, bias)

    def _get_affine_bias(self) -> torch.Tensor | None:
        return self.bias

    def extra_repr(self) -> str:
        bias = is_biased(self)

        log_wght_s = self.log_wght_s

        return (
            f"in_features={self.in_features}, out_features={self.out_features}, bias={bias},\n"
            f"log_wght_s={log_wght_s}"
        )
