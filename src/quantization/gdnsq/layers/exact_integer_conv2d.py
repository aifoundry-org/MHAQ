"""
Exact integer conv (post-training fusion) and optional binarized tail (w1a1).

`ExactIntegerConv2d` replaces `NoisyConv2d` after BN fold; `BinarizedExactIntegerConv2d`
drops fp_scale/bias and emits two-level codes via integer threshold on the accumulator.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.fx import Proxy

from src.quantization.gdnsq.layers.gdnsq_conv2d import NoisyConv2d


def _buffers_like(x: torch.Tensor, *bufs: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """
    Align buffer tensors to x's device/dtype for real execution.

    Under ``torch.fx.symbolic_trace``, ``x`` is a ``Proxy`` and ``x.device`` /
    ``x.dtype`` are not valid for ``.to()``; return buffers unchanged so tracing
    can record the graph.
    """
    if isinstance(x, Proxy):
        return bufs
    return tuple(b.to(device=x.device, dtype=x.dtype) for b in bufs)


def activation_quant_codes(
    x: torch.Tensor,
    act_s: torch.Tensor,
    act_q: torch.Tensor,
    azp: torch.Tensor,
    guard_a: torch.Tensor,
) -> torch.Tensor:
    """Same clamp/round as `ExactIntegerConv2d.forward` input quant (float x -> code-domain float)."""
    zp = azp * act_s / guard_a
    x = torch.clamp(x, min=zp, max=zp + act_q - act_s)
    return guard_a * torch.round((x - zp) / act_s)


class ExactIntegerConv2d(nn.Module):
    """
    Exact integer-input replacement for NoisyConv2d with inlined activation quant.

    The activation integer zero-point offset is folded using a **position-independent**
    correction: for constant code-domain input ``azp``, each output channel of
    ``conv(azp * 1, guard_w * w + wzp)`` equals ``azp * sum(guard_w * w + wzp)`` over
    that channel's kernel (ignoring padding / border effects). That yields a 1xCx1x1
    bias instead of a full-resolution map (which would track image size).
    """

    def __init__(
        self,
        module: NoisyConv2d,
        in_scale: torch.Tensor,
        post_scale: torch.Tensor | None = None,
        post_shift: torch.Tensor | None = None,
    ):
        super().__init__()
        self.stride = module.stride
        self.padding = module.padding
        self.dilation = module.dilation
        self.groups = module.groups
        self.padding_mode = module.padding_mode
        self.kernel_size = module.kernel_size
        weight_scale, weight_zero_point = module._configure_weight_quantizer()
        w = module.Q.quantize(module.weight)
        p_scale = (
            post_scale.reshape(-1)
            if post_scale is not None
            else torch.ones(module.out_channels, device=w.device, dtype=w.dtype)
        )
        p_shift = (
            post_shift.reshape(-1)
            if post_shift is not None
            else torch.zeros(module.out_channels, device=w.device, dtype=w.dtype)
        )
        if module.bias is None:
            b = torch.zeros(module.out_channels, device=w.device, dtype=w.dtype)
        else:
            b = module.bias

        self.act_guard_bit = module.act_guard_bit
        self.weight_guard_bit = module.weight_guard_bit

        guard_a = float(2 ** int(self.act_guard_bit))
        guard_w = float(2 ** int(self.weight_guard_bit))

        azp = torch.round(module.act_b / torch.exp2(module.log_act_s) * guard_a).reshape([])
        wzp = torch.round(guard_w * weight_zero_point).reshape(-1)

        fp_scale = (
            in_scale
            * weight_scale.reshape(1, -1, 1, 1)
            * p_scale.view(1, -1, 1, 1)
            / (guard_a * guard_w)
        )
        bias_base = b.view(1, -1, 1, 1) / p_scale.view(1, -1, 1, 1) + p_shift.view(1, -1, 1, 1)

        wzp_kernel = wzp.view(-1, 1, 1, 1).expand_as(w)
        wk = guard_w * w + wzp_kernel
        d_per_out_ch = azp * wk.sum(dim=(1, 2, 3))
        bias_spatial = d_per_out_ch.view(1, -1, 1, 1) * fp_scale
        bias_full = bias_base + bias_spatial

        self.register_buffer("weight", w.detach().clone().to(device=w.device, dtype=w.dtype), persistent=True)
        self.register_buffer(
            "act_s",
            torch.exp2(module.log_act_s).reshape([]).detach().clone().to(device=w.device, dtype=w.dtype),
            persistent=True,
        )
        self.register_buffer(
            "act_q",
            torch.exp2(module.log_act_q).reshape([]).detach().clone().to(device=w.device, dtype=w.dtype),
            persistent=True,
        )
        self.register_buffer("azp", azp.detach().clone().to(device=w.device, dtype=w.dtype), persistent=True)
        self.register_buffer("wzp", wzp.detach().clone().to(device=w.device, dtype=w.dtype), persistent=True)
        self.register_buffer("fp_scale", fp_scale.detach().clone().to(device=w.device, dtype=w.dtype), persistent=True)
        self.register_buffer("bias", bias_full.detach().clone().to(device=w.device, dtype=w.dtype), persistent=True)
        self.register_buffer(
            "guard_a",
            torch.tensor(guard_a, device=w.device, dtype=w.dtype),
            persistent=True,
        )
        self.register_buffer(
            "guard_w",
            torch.tensor(guard_w, device=w.device, dtype=w.dtype),
            persistent=True,
        )

    def _conv(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        if self.padding_mode != "zeros":
            pad_h, pad_w = self.padding
            x = F.pad(x, (pad_w, pad_w, pad_h, pad_h), mode=self.padding_mode)
            padding = (0, 0)
        else:
            padding = self.padding
        return F.conv2d(x, weight, None, self.stride, padding, self.dilation, self.groups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        zp = self.azp * self.act_s / self.guard_a
        x = torch.clamp(x, min=zp, max=zp + self.act_q - self.act_s)
        x = self.guard_a * torch.round((x - zp) / self.act_s)

        c1 = self._conv(x, self.weight)
        c2 = self._conv(x, torch.ones_like(self.weight))
        out = self.guard_w * c1 + self.wzp.view(1, -1, 1, 1) * c2

        (bias,) = _buffers_like(x, self.bias)
        # Broadcast bias 1xCx1x1 over batch (avoid expand(batch) - breaks under FX Proxy shapes).
        return out * self.fp_scale + bias


class BinarizedExactIntegerConv2d(nn.Module):
    """
    Same core as `ExactIntegerConv2d` but replaces ``out * fp_scale + bias`` with
    ``where(out >= th, code_hi, code_lo)`` (integer `th` per channel, broadcast 1xCx1x1).
    """

    def __init__(self, base: ExactIntegerConv2d):
        super().__init__()
        self.stride = base.stride
        self.padding = base.padding
        self.dilation = base.dilation
        self.groups = base.groups
        self.padding_mode = base.padding_mode
        self.kernel_size = base.kernel_size
        self.act_guard_bit = base.act_guard_bit
        self.weight_guard_bit = base.weight_guard_bit

        for name in (
            "weight",
            "act_s",
            "act_q",
            "azp",
            "wzp",
            "guard_a",
            "guard_w",
        ):
            self.register_buffer(name, getattr(base, name).detach().clone(), persistent=True)

        c_out = base.weight.shape[0]
        self.register_buffer(
            "th",
            torch.zeros(c_out, device=base.weight.device, dtype=torch.int64).view(1, -1, 1, 1),
            persistent=True,
        )
        self.register_buffer(
            "code_lo",
            torch.zeros(1, c_out, 1, 1, device=base.weight.device, dtype=base.weight.dtype),
            persistent=True,
        )
        self.register_buffer(
            "code_hi",
            torch.zeros(1, c_out, 1, 1, device=base.weight.device, dtype=base.weight.dtype),
            persistent=True,
        )

    def _conv(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        if self.padding_mode != "zeros":
            pad_h, pad_w = self.padding
            x = F.pad(x, (pad_w, pad_w, pad_h, pad_h), mode=self.padding_mode)
            padding = (0, 0)
        else:
            padding = self.padding
        return F.conv2d(x, weight, None, self.stride, padding, self.dilation, self.groups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        zp = self.azp * self.act_s / self.guard_a
        x = torch.clamp(x, min=zp, max=zp + self.act_q - self.act_s)
        x = self.guard_a * torch.round((x - zp) / self.act_s)

        c1 = self._conv(x, self.weight)
        c2 = self._conv(x, torch.ones_like(self.weight))
        out = self.guard_w * c1 + self.wzp.view(1, -1, 1, 1) * c2

        # Integer-safe threshold: `out` is float-valued but should be integral; direct
        # `out >= th` can mis-classify at boundaries (e.g. 4.999999 vs th=5).
        lo0, hi0 = _buffers_like(out, self.code_lo, self.code_hi)
        th_i = self.th if isinstance(out, Proxy) else self.th.to(device=out.device)
        k_int = torch.round(out).to(torch.int64)
        mask = k_int >= th_i
        return torch.where(mask, hi0, lo0)


def apply_unary_chain(z: torch.Tensor, modules: list[nn.Module]) -> torch.Tensor:
    for m in modules:
        if isinstance(m, nn.LeakyReLU):
            z = F.leaky_relu(z, m.negative_slope, inplace=False)
        elif isinstance(m, nn.ReLU):
            z = F.relu(z, inplace=False)
        elif isinstance(m, nn.Identity):
            pass
        elif isinstance(m, nn.Dropout):
            pass
        else:
            z = m(z)
    return z


def derive_channel_thresholds_and_codes(
    fp_scale: torch.Tensor,
    bias: torch.Tensor,
    q: ExactIntegerConv2d,
    unary_modules: list[nn.Module],
    k_scan_radius: int = 500_000,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    For each output channel c of producer P, find integer th[c] and code_lo/hi such that
    activation_quant_codes(apply_unary(s*k+b), Q) matches where(k>=th, code_hi, code_lo).

    Assumes w1a1-style two-level quantization after the unary chain.
    """
    device = fp_scale.device
    dtype = fp_scale.dtype
    c_out = fp_scale.shape[1]
    guard_a = q.guard_a.to(device=device, dtype=dtype)

    th = torch.zeros(c_out, dtype=torch.int64, device=device)
    code_lo = torch.zeros(1, c_out, 1, 1, device=device, dtype=dtype)
    code_hi = torch.zeros(1, c_out, 1, 1, device=device, dtype=dtype)

    k_axis = torch.arange(-k_scan_radius, k_scan_radius + 1, device=device, dtype=dtype)

    for c in range(c_out):
        s = fp_scale[0, c, 0, 0]
        b = bias[0, c, 0, 0]
        if s <= 0:
            raise ValueError(f"derive_channel_thresholds: non-positive fp_scale at channel {c}")

        z = s * k_axis + b
        z = apply_unary_chain(z.unsqueeze(0).unsqueeze(-1).unsqueeze(-1), unary_modules).view(-1)
        codes = activation_quant_codes(z, q.act_s, q.act_q, q.azp, guard_a)
        uniq = torch.unique(codes)
        if uniq.numel() < 2:
            raise ValueError(
                f"derive_channel_thresholds: need >=2 quant levels along k scan for channel {c}, got {uniq.numel()}"
            )

        v_small_k = codes[0]
        v_large_k = codes[-1]
        if torch.isclose(v_small_k, v_large_k):
            raise ValueError(f"derive_channel_thresholds: channel {c} single level at k scan endpoints")

        code_lo[0, c, 0, 0] = v_small_k
        code_hi[0, c, 0, 0] = v_large_k
        target_at_large_k = v_large_k

        neq = codes != target_at_large_k
        if not bool(neq.any()):
            stable_from = 0
        else:
            last_problem = int(torch.where(neq)[0].max().item())
            stable_from = last_problem + 1
        if stable_from >= codes.numel():
            raise ValueError(f"derive_channel_thresholds: non-plateau tail for channel {c}")
        th[c] = int(k_axis[stable_from].item())

    return th.view(1, -1, 1, 1), code_lo, code_hi
