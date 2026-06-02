import torch

from torch import nn, inf

from src.aux.types import QScheme
from src.quantization.gdnsq.gdnsq import Quantizer
from src.quantization.gdnsq.gdnsq_utils import QNMethod


class NoisyActLin(nn.Module):

    def _init_noisy_actlin(
        self,
        *,
        qscheme: QScheme,
        log_s_init: float,
        rand_noise: bool,
        disable: bool = False,
        act_init_s: float = -10,
        act_init_q: float = 10,
        qnmethod: QNMethod = QNMethod.STE,
        per_channel_shape: tuple[int, ...],
        weight_guard_bit: int = 0,
        act_guard_bit: int = 0,
    ) -> None:
        """
        Common initialization for activation and weight quantization used by
        NoisyLinear/NoisyConv2d subclasses.
        """
        # Store guard bits for later use in quantization math or hardware export.
        self.weight_guard_bit = int(weight_guard_bit)
        self.act_guard_bit = int(act_guard_bit)
        # Inline activation quantization init
        self.disable = disable
        zero_point = -torch.exp2(torch.tensor(act_init_q - 1).float())
        self._act_b = torch.tensor([zero_point]).float()
        self._log_act_s = torch.tensor([act_init_s]).float()
        self._log_act_q = torch.tensor([act_init_q]).float()
        self._noise_ratio = nn.Parameter(torch.tensor([1.0]).float(), requires_grad=False)

        self.log_act_q = nn.Parameter(self._log_act_q, requires_grad=True)
        self.act_b = nn.Parameter(self._act_b, requires_grad=True)
        self.log_act_s = nn.Parameter(self._log_act_s, requires_grad=True)
        self.Q_input = Quantizer(
            self, torch.exp2(self._log_act_s), 0, -inf, inf, qnmethod=qnmethod
        )
        self.bw = torch.tensor(0.0)

        # Inline weight quantization init
        self.qscheme = qscheme
        if self.qscheme == QScheme.PER_TENSOR:
            self.log_wght_s = nn.Parameter(torch.tensor([log_s_init]).float(), requires_grad=True)
        elif self.qscheme == QScheme.PER_CHANNEL:
            self.log_wght_s = nn.Parameter(
                torch.empty(per_channel_shape).fill_(log_s_init), requires_grad=True
            )
        else:
            raise NotImplementedError(f"Unsupported qscheme {self.qscheme}")

        self.Q = Quantizer(
            self, torch.exp2(self.log_wght_s), 0, -inf, inf, qnmethod=qnmethod
        )
        self.rand_noise = rand_noise

    def _weight_quantization_dims(self) -> tuple[int, ...]:
        raise NotImplementedError

    def _apply_affine(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError

    def _get_affine_bias(self) -> torch.Tensor | None:
        raise NotImplementedError

    def get_weight_minmax(self, keepdim: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        if self.qscheme == QScheme.PER_CHANNEL:
            dims = self._weight_quantization_dims()
            return (
                self.weight.amin(dims, keepdim=keepdim),
                self.weight.amax(dims, keepdim=keepdim),
            )
        if self.qscheme == QScheme.PER_TENSOR:
            return self.weight.amin(), self.weight.amax()
        raise NotImplementedError(f"Unsupported qscheme {self.qscheme}")

    def _configure_activation_quantizer(self) -> tuple[torch.Tensor, torch.Tensor]:
        s = torch.exp2(self.log_act_s)
        q = torch.exp2(self.log_act_q)
        guard_ratio = 2 ** self.act_guard_bit
        qzp = torch.round(self.act_b / s * guard_ratio) 
        zp = qzp / guard_ratio * s
        self.Q_input.zero_point = zp
        self.Q_input.min_val = zp
        self.Q_input.max_val = zp + q - s
        self.Q_input.scale = s
        return s, qzp / guard_ratio

    def quantize_input(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Quantize the input tensor and return:
          - quantized codes
          - activation scale
          - activation quantization zero-point
        """
        if self.disable:
            # No activation quantization: behave as identity with unit scale.
            scale = torch.tensor(1.0, device=x.device, dtype=x.dtype)
            zero_point = torch.tensor(0.0, device=x.device, dtype=x.dtype)
            return x, scale, zero_point

        scale, q_zero_point = self._configure_activation_quantizer()
        qx = self.Q_input.quantize(x)
        if not self.training:
            minmax = qx.aminmax()
            self.bw = torch.log2(minmax.max - minmax.min + 1)

        return qx, scale, q_zero_point

    def _configure_weight_quantizer(self) -> tuple[torch.Tensor, torch.Tensor]:
        scale = torch.exp2(self.log_wght_s)
        wmin, wmax = self.get_weight_minmax(keepdim=self.qscheme == QScheme.PER_CHANNEL)
        guard_ratio = 2 ** self.weight_guard_bit
        qwmin = torch.round(wmin / scale * guard_ratio) 
        qwmax = torch.round(wmax / scale * guard_ratio) 
        self.Q.scale = scale
        self.Q.min_val = qwmin / guard_ratio * scale
        self.Q.max_val = qwmax / guard_ratio * scale
        self.Q.zero_point = self.Q.min_val
        return scale, qwmin / guard_ratio

    def quantize_weight(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        scale, q_zero_point = self._configure_weight_quantizer()
        return self.Q.quantize(self.weight), scale, q_zero_point


    def forward(self, input: torch.Tensor) -> torch.Tensor:
        bias = self._get_affine_bias()
        q_input, act_scale, act_quant_zero_point = self.quantize_input(input)
        q_weight, weight_scale, weight_quant_zero_point = self.quantize_weight()
        scale = weight_scale * act_scale

        # Bias can be None (no affine offset).
        norm_bias = None
        if bias is not None:
            norm_bias = bias.view(-1) / scale.view(-1)

        # Work in fixpoint domain except bias.
        fix_point_out = self._apply_affine(
            q_input + act_quant_zero_point,
            q_weight + weight_quant_zero_point,
            norm_bias,
        )

        # Broadcast weight_scale over the output channel dimension only.
        scale = scale.view(1, -1, *([1] * (self.weight.dim() - 2)))

        return fix_point_out * scale