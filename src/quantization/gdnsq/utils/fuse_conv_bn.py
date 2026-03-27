import torch

from torch import nn
from operator import attrgetter
from src.aux.qutils import attrsetter
from src.quantization.gdnsq.gdnsq import BinaryQuantizer, Quantizer


def _channel_param(param: torch.Tensor, channel_idx: int, out_channels: int):
    tensor = torch.as_tensor(param)
    if tensor.ndim > 0 and tensor.shape[0] == out_channels:
        return tensor[channel_idx]
    return tensor


def _fold_bn_param(param: torch.Tensor, bn_scale: torch.Tensor, out_channels: int):
    tensor = torch.as_tensor(param)
    if tensor.ndim > 0 and tensor.shape[0] == out_channels:
        view_shape = (out_channels,) + (1,) * (tensor.ndim - 1)
        return tensor * bn_scale.view(view_shape).to(tensor.device)
    return tensor * bn_scale.mean().to(tensor.device)


def _binary_dequantize(weight: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor):
    quantized = torch.sign(weight.to(zero_point.device) - zero_point).clamp_min(0)
    return quantized * scale + (zero_point - scale / 2)


def _channel_margin(
    weight_channel: torch.Tensor,
    zero_point_channel: torch.Tensor,
    min_margin: float,
):
    finfo = torch.finfo(weight_channel.dtype)
    ref = torch.stack(
        [
            weight_channel.detach().abs().amax(),
            torch.as_tensor(zero_point_channel, device=weight_channel.device).detach().abs().amax(),
            weight_channel.new_tensor(1.0),
        ]
    ).amax()
    return torch.maximum(weight_channel.new_tensor(min_margin), ref * finfo.eps * 8)


def _minimal_stable_delta(
    weight: torch.Tensor,
    zero_point: torch.Tensor,
    margin: torch.Tensor,
    choose_upper: torch.Tensor,
):
    upper_delta = (zero_point + margin - weight).clamp_min(0)
    lower_delta = (zero_point - margin - weight).clamp_max(0)
    return torch.where(choose_upper, upper_delta, lower_delta)


def _optimize_binary_weights_sparse(
    weight: torch.Tensor,
    reference_weight: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    delta_limit: float | None,
    min_margin: float,
):  
    weight = weight.to(zero_point.device)
    margin = _channel_margin(weight, zero_point, min_margin)
    upper_level = zero_point + scale / 2
    lower_level = zero_point - scale / 2

    upper_error = (reference_weight.to(upper_level.device) - upper_level).abs()
    lower_error = (reference_weight.to(lower_level.device) - lower_level).abs()
    current_upper = weight > zero_point
    prefer_upper = upper_error < lower_error
    prefer_upper = torch.where(upper_error == lower_error, current_upper, prefer_upper)

    proposed_delta = _minimal_stable_delta(
        weight=weight,
        zero_point=zero_point,
        margin=margin,
        choose_upper=prefer_upper,
    )

    if delta_limit is None:
        can_apply = torch.ones_like(weight, dtype=torch.bool)
    else:
        can_apply = proposed_delta.abs() <= weight.new_tensor(delta_limit)

    base_delta = _minimal_stable_delta(
        weight=weight,
        zero_point=zero_point,
        margin=margin,
        choose_upper=current_upper,
    )
    apply_mask = can_apply & (prefer_upper != current_upper)
    final_delta = torch.where(apply_mask, proposed_delta, base_delta)
    optimized_weight = weight + final_delta
    optimized_binary = _binary_dequantize(optimized_weight, scale, zero_point)

    return {
        "optimized_weight": optimized_weight,
        "optimized_binary": optimized_binary,
        "weight_deltas": final_delta,
        "margin": margin,
        "changed_mask": apply_mask,
        "changed_count": int(apply_mask.sum().item()),
    }


def _compute_bn_fold_diagnostics(
    weight: torch.Tensor,
    reference_weight: torch.Tensor,
    new_scale: torch.Tensor,
    new_zero_point: torch.Tensor,
    bn_scale: torch.Tensor,
    fused_weight: torch.Tensor,
    fused_scale: torch.Tensor,
    fused_zero_point: torch.Tensor,
    min_margin: float,
):
    shape = [-1] + [1] * (weight.dim() - 1)
    bn_scale_view = bn_scale.view(shape)
    reference_weight = reference_weight.to(fused_weight.device)

    unfused_binary = _binary_dequantize(
        weight,
        new_scale,
        new_zero_point,
    ).to(fused_weight.device)
    folded_reference = reference_weight * bn_scale_view
    folded_from_unfused_binary = unfused_binary * bn_scale_view
    folded_binary = _binary_dequantize(
        fused_weight,
        fused_scale,
        fused_zero_point,
    )

    unfused_mask = (weight.to(new_zero_point.device) > new_zero_point).to(fused_weight.device)
    folded_mask = (fused_weight.to(fused_zero_point.device) > fused_zero_point).to(fused_weight.device)
    folded_margin = (fused_weight.to(fused_zero_point.device) - fused_zero_point).abs()

    return {
        "bn_scale_min": bn_scale.amin().item(),
        "bn_scale_max": bn_scale.amax().item(),
        "bn_scale_mean": bn_scale.mean().item(),
        "negative_bn_scale_channels": int((bn_scale < 0).sum().item()),
        "negative_bn_scale_channel_indices": (
            torch.nonzero(bn_scale < 0, as_tuple=False).reshape(-1).detach().cpu().tolist()
        ),
        "negative_bn_scale_fraction": (bn_scale < 0).float().mean().item(),
        "unfused_binary_mean_abs_diff": (
            reference_weight - unfused_binary
        ).abs().mean().item(),
        "folded_binary_mean_abs_diff": (
            folded_reference.to(folded_binary.device) - folded_binary
        ).abs().mean().item(),
        "folded_from_unfused_binary_mean_abs_diff": (
            folded_reference.to(folded_from_unfused_binary.device) - folded_from_unfused_binary
        ).abs().mean().item(),
        "folding_binary_drift_mean_abs": (
            folded_binary.to(folded_from_unfused_binary.device) - folded_from_unfused_binary
        ).abs().mean().item(),
        "binary_assignment_flip_fraction": (
            unfused_mask != folded_mask
        ).float().mean().item(),
        "near_threshold_fraction": (
            folded_margin <= _channel_margin(fused_weight, fused_zero_point, min_margin)
        ).float().mean().item(),
    }

def fuse_conv_bn(model: nn.Module, conv_name: str, bn_name: str):
    conv = attrgetter(conv_name)(model)

    W = conv.weight.clone()
    if conv.bias is not None:
        b = conv.bias.clone()
    else:
        b = torch.zeros(conv.out_channels, device=W.device)

    bn = attrgetter(bn_name)(model)
    mu = bn.running_mean
    var = bn.running_var
    eps = bn.eps
    gamma = bn.weight
    beta = bn.bias

    std = torch.sqrt(var + eps)
    scale = gamma / std
    shape = [-1] + [1] * (W.dim() - 1)

    conv.weight.data = W * scale.view(shape)
    conv.bias = nn.Parameter(beta + (b - mu) * scale)

    attrsetter(bn_name)(model, nn.Identity())  # Replacing bn module with Identity


# I could optimize the shit out of this function but since it is invocated only once, I don't care.
# So let's keep it KISS.
# TODO due to the questionable manipulations whom are mathematically correct (at first glance at least)
# There is slight metric loss. Because of the lack if fp32 precision I suppose.
# FIX IT!
def fuse_conv_bn_q(model: nn.Module, conv_name: str, bn_name: str):
    conv = attrgetter(conv_name)(model)
    prev_q = conv.Q
    new_zero_point = prev_q.zero_point + prev_q.scale.mul(0.5)
    new_scale = prev_q.scale

    W = conv.weight.detach().clone()
    if conv.bias is not None:
        b = conv.bias.detach().clone()
    else:
        b = torch.zeros(conv.out_channels, device=W.device)

    bn = attrgetter(bn_name)(model)
    mu = bn.running_mean
    var = bn.running_var
    eps = bn.eps
    gamma = bn.weight
    beta = bn.bias

    std = torch.sqrt(var + eps)
    bn_scale = gamma / std
    bn_scale_abs = bn_scale.abs()

    conv.scale = new_scale
    conv.zero_point = new_zero_point
    
    # hack with "shaking" weights in order to get proper binarization
    # (try to disable it and see what happens, or come up with a better solution)
    # conv.weight.data = conv.weight.data.double() + prev_q.scale.mul(0.5).to(conv.weight.device).double() - prev_q.scale.mul(
    # 0.5).to(conv.weight.device).double()
    # conv.weight.data = conv.weight.data.float()
    conv.weight.data = conv.weight.data + \
        prev_q.scale.mul(0.5).to(conv.weight.device) - \
        prev_q.scale.mul(0.5).to(conv.weight.device)
    # conv.weight.data *= bn_scale.view([-1] + [1] * (W.dim() - 1))
    # conv.bias = nn.Parameter(beta + (b - mu) * bn_scale)
    # new_zero_point *= bn_scale.view(
        # new_zero_point.shape).to(new_zero_point.device)
    # new_scale *= bn_scale_abs.view(new_scale.shape).to(new_zero_point.device)

    rnoise_ratio = float(torch.as_tensor(
        prev_q.rnoise_ratio).reshape(-1)[0])
    reference_q = Quantizer(
        module=conv,
        scale=prev_q.scale,
        zero_point=prev_q.zero_point,
        min_val=prev_q.min_val,
        max_val=prev_q.max_val,
        rnoise_ratio=rnoise_ratio,
        qnmethod=prev_q.qnmethod,
    )
    binary_q = BinaryQuantizer(
        module=conv,
        scale=new_scale,
        zero_point=new_zero_point,
        min_val=prev_q.min_val,
        max_val=prev_q.max_val,
        rnoise_ratio=rnoise_ratio,
        qnmethod=prev_q.qnmethod,
    )
    with torch.no_grad():
        reference_weight = reference_q.dequantize(
            reference_q.quantize(conv.weight.detach().to(new_scale.device))
        )
        binary_weight = binary_q.dequantize(
            binary_q.quantize(conv.weight.detach().to(new_scale.device))
        )
        mean_diff = (reference_weight - binary_weight).mean().item()
        mean_abs_diff = (reference_weight -
                            binary_weight).abs().mean().item()

    conv.Q = binary_q
    conv.binary_quantizer_weight_mean_diff = mean_diff
    conv.binary_quantizer_weight_mean_abs_diff = mean_abs_diff

    # attrsetter(bn_name)(model, nn.Identity())
    return {
        "mean_diff": mean_diff,
        "mean_abs_diff": mean_abs_diff,
    }


def fuse_conv_bn_q_optimized(
    model: nn.Module,
    conv_name: str,
    bn_name: str,
    delta_limit: float | None = 1e-6,
    min_margin: float = 1e-7,
    max_candidates: int | None = 1024,
    collect_diagnostics: bool = True,
):
    conv = attrgetter(conv_name)(model)
    prev_q = conv.Q
    out_channels = conv.out_channels
    shape = [-1] + [1] * (conv.weight.dim() - 1)

    W = conv.weight.detach().clone()
    if conv.bias is not None:
        b = conv.bias.detach().clone()
    else:
        b = torch.zeros(out_channels, device=W.device)

    bn = attrgetter(bn_name)(model)
    mu = bn.running_mean
    var = bn.running_var
    eps = bn.eps
    gamma = bn.weight
    beta = bn.bias

    std = torch.sqrt(var + eps)
    bn_scale = gamma / std
    bn_scale_abs = bn_scale.abs()
    bn_scale_view = bn_scale.view(shape)

    new_zero_point = prev_q.zero_point + prev_q.scale.mul(0.5)
    new_scale = prev_q.scale

    # fused_weight = W
    # fused_bias = b
    # fused_zero_point = new_zero_point
    # fused_scale = new_scale

    fused_weight = W * bn_scale_view
    fused_bias = beta + (b - mu) * bn_scale
    fused_zero_point = _fold_bn_param(new_zero_point, bn_scale, out_channels)
    fused_scale = _fold_bn_param(new_scale, bn_scale_abs, out_channels)

    rnoise_ratio = float(torch.as_tensor(prev_q.rnoise_ratio).reshape(-1)[0])
    reference_q = Quantizer(
        module=conv,
        scale=prev_q.scale,
        zero_point=prev_q.zero_point,
        min_val=prev_q.min_val,
        max_val=prev_q.max_val,
        rnoise_ratio=rnoise_ratio,
        qnmethod=prev_q.qnmethod,
    )

    with torch.no_grad():
        reference_weight = reference_q.dequantize(
            reference_q.quantize(W.to(prev_q.scale.device))
        ).to(fused_weight.device)
        reference_weight = reference_weight * bn_scale_view
        # reference_weight = reference_weight

        baseline_binary = _binary_dequantize(
            fused_weight,
            fused_scale,
            fused_zero_point,
        )
        baseline_error = reference_weight.to(baseline_binary.device) - baseline_binary
        baseline_mean_abs_diff = baseline_error.abs().mean().item()

        optimized = _optimize_binary_weights_sparse(
            weight=fused_weight,
            reference_weight=reference_weight.to(fused_weight.device),
            scale=fused_scale,
            zero_point=fused_zero_point,
            delta_limit=delta_limit,
            min_margin=min_margin,
        )
        optimized_weight = optimized["optimized_weight"]
        optimized_binary = optimized["optimized_binary"]
        weight_deltas = optimized["weight_deltas"]
        channel_mean_abs_diffs = [
            (reference_weight[idx].to(optimized_binary.device) - optimized_binary[idx]).abs().mean().item()
            for idx in range(out_channels)
        ]
        channel_margins = [
            _channel_margin(
                fused_weight[idx],
                _channel_param(fused_zero_point, idx, out_channels),
                min_margin,
            ).item()
            for idx in range(out_channels)
        ]
        channel_deltas = weight_deltas.reshape(out_channels, -1).mean(dim=1)
        changed_count = optimized["changed_count"]

        error = reference_weight.to(optimized_binary.device) - optimized_binary
        mean_diff = error.mean().item()
        mean_abs_diff = error.abs().mean().item()
        diagnostics = {}
        if collect_diagnostics:
            diagnostics = _compute_bn_fold_diagnostics(
                weight=W.to(new_zero_point.device),
                reference_weight=reference_q.dequantize(
                    reference_q.quantize(W.to(prev_q.scale.device))
                ),
                new_scale=new_scale,
                new_zero_point=new_zero_point,
                bn_scale=bn_scale,
                fused_weight=fused_weight,
                fused_scale=fused_scale,
                fused_zero_point=fused_zero_point,
                min_margin=min_margin,
            )

    conv.scale = fused_scale
    conv.zero_point = fused_zero_point
    conv.weight.data = optimized_weight
    conv.bias = nn.Parameter(fused_bias)
    conv.Q = BinaryQuantizer(
        module=conv,
        scale=fused_scale,
        zero_point=fused_zero_point,
        min_val=prev_q.min_val,
        max_val=prev_q.max_val,
        rnoise_ratio=rnoise_ratio,
        qnmethod=prev_q.qnmethod,
    )
    conv.binary_quantizer_channel_deltas = channel_deltas.detach().cpu()
    conv.binary_quantizer_weight_deltas = weight_deltas.detach().cpu()
    conv.binary_quantizer_weight_mean_diff = mean_diff
    conv.binary_quantizer_weight_mean_abs_diff = mean_abs_diff
    conv.binary_quantizer_bn_fold_diagnostics = diagnostics

    attrsetter(bn_name)(model, nn.Identity())
    return {
        "mean_diff": mean_diff,
        "mean_abs_diff": mean_abs_diff,
        "baseline_mean_abs_diff": baseline_mean_abs_diff,
        "improvement": baseline_mean_abs_diff - mean_abs_diff,
        "channel_deltas": channel_deltas.detach().cpu().tolist(),
        "channel_mean_abs_diffs": channel_mean_abs_diffs,
        "channel_margins": channel_margins,
        "changed_weights": changed_count,
        "bn_diagnostics": diagnostics,
    }
