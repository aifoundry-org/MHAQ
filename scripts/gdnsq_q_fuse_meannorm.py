"""
Fuse MeanNorm into previous NoisyConv2d bias, normalize NoisyAct scales,
run validation, print weight/bias value-set stats, and save fused checkpoint.
"""
import os
import sys
import resource

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))
import torch
import argparse
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from src.config.config_loader import load_and_validate_config
from src.data.compose.composer import DatasetComposer
from src.models.compose.composer import ModelComposer
from src.quantization.quantizer import Quantizer
from src.quantization.gdnsq.layers.gdnsq_act import NoisyAct
from src.quantization.gdnsq.layers.gdnsq_conv2d import NoisyConv2d
from src.quantization.gdnsq.layers.gdnsq_linear import NoisyLinear
from src.quantization.gdnsq.layers.gdnsq_mean_norm import MeanNorm1d, MeanNorm2d, MeanNorm3d
from src.training.trainer import Validator
from src.loggers.default_logger import logger

torch.set_float32_matmul_precision("high")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fuse MeanNorm, normalize NoisyAct scales, validate, print stats, save fused checkpoint."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the configuration file (YAML).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the trained quantized checkpoint (.ckpt).",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save the fused checkpoint.",
    )
    parser.add_argument(
        "--debug-divergence",
        action="store_true",
        help="Run fused and normalized models side by side and print the first divergent layer.",
    )
    return parser.parse_args()


class _PassThrough(torch.nn.Module):
    """Passthrough so a slot in the graph can be removed semantically (conv output goes to next layer)."""

    def forward(self, x):
        return x


class _RoundClampAct(torch.nn.Module):
    """Inference-only integer activation: clamp to range, then round to integer codes."""

    def __init__(
        self,
        log_act_s: torch.Tensor,
        log_act_q: torch.Tensor,
        act_b: torch.Tensor,
        disable: bool,
        bw: torch.Tensor | None = None,
    ):
        super().__init__()
        self.disable = disable
        self.register_buffer("log_act_s", log_act_s.detach().clone(), persistent=True)
        self.register_buffer("log_act_q", log_act_q.detach().clone(), persistent=True)
        self.register_buffer("act_b", act_b.detach().clone(), persistent=True)
        self.register_buffer(
            "bw",
            bw.detach().clone() if bw is not None else torch.tensor(0.0, device=log_act_s.device, dtype=log_act_s.dtype),
            persistent=True,
        )

    @classmethod
    def from_noisy_act(cls, module: NoisyAct) -> "_RoundClampAct":
        return cls(
            log_act_s=module.log_act_s,
            log_act_q=module.log_act_q,
            act_b=module.act_b,
            disable=module.disable,
            bw=module.bw if torch.is_tensor(module.bw) else None,
        ).to(device=module.log_act_s.device, dtype=module.log_act_s.dtype)

    def forward(self, x):
        if self.disable:
            return x

        s = torch.exp2(self.log_act_s)
        q = torch.exp2(self.log_act_q)
        x = torch.clamp(x, min=self.act_b, max=self.act_b + q - s)
        out = torch.round((x - self.act_b) / s)
        if not self.training:
            minmax = out.aminmax()
            self.bw = torch.log2(minmax.max - minmax.min + 1)
        return out


class _ExactIntegerConv2d(torch.nn.Module):
    """Exact integer-input replacement for NoisyConv2d."""

    def __init__(
        self,
        module: NoisyConv2d,
        weight: torch.Tensor,
        bias: torch.Tensor,
        input_scale: float,
        input_zero_point: float,
        weight_scale: torch.Tensor,
        weight_zero_point: torch.Tensor,
    ):
        super().__init__()
        self.stride = module.stride
        self.padding = module.padding
        self.dilation = module.dilation
        self.groups = module.groups
        self.padding_mode = module.padding_mode
        self.register_buffer("weight", weight.detach().clone(), persistent=True)
        self.register_buffer("bias", bias.detach().clone(), persistent=True)
        self.register_buffer(
            "input_scale",
            torch.tensor(float(input_scale), device=weight.device, dtype=weight.dtype),
            persistent=True,
        )
        self.register_buffer(
            "input_zero_point",
            torch.tensor(float(input_zero_point), device=weight.device, dtype=weight.dtype),
            persistent=True,
        )
        self.register_buffer("weight_scale", weight_scale.detach().clone(), persistent=True)
        self.register_buffer(
            "weight_zero_point",
            weight_zero_point.detach().clone().to(device=weight.device, dtype=weight.dtype),
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

    def forward(self, x):
        dequantized_input = x * self.input_scale
        if self.input_zero_point.item() != 0.0:
            dequantized_input = dequantized_input + self.input_zero_point
        dequantized_weight = self.weight * self.weight_scale + self.weight_zero_point
        out = self._conv(dequantized_input, dequantized_weight)
        return out + self.bias.view(1, -1, 1, 1)


class _ExactIntegerLinear(torch.nn.Module):
    """Exact integer-input replacement for NoisyLinear."""

    def __init__(
        self,
        module: NoisyLinear,
        weight: torch.Tensor,
        bias: torch.Tensor,
        input_scale: float,
        input_zero_point: float,
        weight_scale: torch.Tensor,
        weight_zero_point: torch.Tensor,
    ):
        super().__init__()
        del module
        self.register_buffer("weight", weight.detach().clone(), persistent=True)
        self.register_buffer("bias", bias.detach().clone(), persistent=True)
        self.register_buffer(
            "input_scale",
            torch.tensor(float(input_scale), device=weight.device, dtype=weight.dtype),
            persistent=True,
        )
        self.register_buffer(
            "input_zero_point",
            torch.tensor(float(input_zero_point), device=weight.device, dtype=weight.dtype),
            persistent=True,
        )
        self.register_buffer("weight_scale", weight_scale.detach().clone(), persistent=True)
        self.register_buffer(
            "weight_zero_point",
            weight_zero_point.detach().clone().to(device=weight.device, dtype=weight.dtype),
            persistent=True,
        )

    def forward(self, x):
        dequantized_input = x * self.input_scale
        if self.input_zero_point.item() != 0.0:
            dequantized_input = dequantized_input + self.input_zero_point
        dequantized_weight = self.weight * self.weight_scale + self.weight_zero_point
        return F.linear(dequantized_input, dequantized_weight, self.bias)


def _get_noisy_conv(module):
    """Return NoisyConv2d from module or from Sequential(..., '0'=NoisyConv2d)."""
    if isinstance(module, NoisyConv2d):
        return module
    if isinstance(module, torch.nn.Sequential) and "0" in module._modules:
        sub = module._modules["0"]
        if isinstance(sub, NoisyConv2d):
            return sub
    return None


def _get_quantized_weight_params(
    module: NoisyConv2d | NoisyLinear,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return integer-valued weight codes together with their quantization parameters."""
    weight = module.weight.detach()
    scale = torch.exp2(module.log_wght_s.detach())
    module.Q.scale = scale

    if module.qscheme.name == "PER_CHANNEL":
        reduce_dims = tuple(range(1, weight.dim()))
        zero_point = weight.amin(dim=reduce_dims, keepdim=True)
    else:
        zero_point = weight.amin()

    module.Q.zero_point = zero_point
    return module.Q.quantize(weight), scale.detach().clone(), zero_point.detach().clone()


def _get_effective_bias(module: NoisyConv2d | NoisyLinear) -> torch.Tensor:
    """Return the effective bias used by the quantized layer."""
    out_features = module.out_channels if isinstance(module, NoisyConv2d) else module.out_features
    if module.bias is None:
        return torch.zeros(out_features, device=module.weight.device, dtype=module.weight.dtype)

    if isinstance(module, NoisyConv2d) and getattr(module, "quant_bias", False):
        s = torch.exp2(module.log_wght_s.detach())
        if module.qscheme.name == "PER_CHANNEL":
            zero_point = module.weight.detach().amin((1, 2, 3), keepdim=True)
        else:
            zero_point = module.weight.detach().amin()
        module.Q_b.scale = s.ravel()
        module.Q_b.zero_point = zero_point.ravel()
        module.Q_b.rnoise_ratio.data = torch.zeros_like(module._noise_ratio)
        return module.Q_b.dequantize(module.Q_b.quantize(module.bias.detach()))

    return module.bias.detach().clone()


def _make_exact_integer_affine_from_quantized(
    module: NoisyConv2d | NoisyLinear,
    weight: torch.Tensor,
    bias: torch.Tensor,
    input_scale: float,
    input_zero_point: float,
    weight_scale: torch.Tensor,
    weight_zero_point: torch.Tensor,
) -> torch.nn.Module:
    if isinstance(module, NoisyConv2d):
        return _ExactIntegerConv2d(
            module, weight, bias, input_scale, input_zero_point, weight_scale, weight_zero_point
        )
    return _ExactIntegerLinear(
        module, weight, bias, input_scale, input_zero_point, weight_scale, weight_zero_point
    )


def _iter_quantized_blocks(root: torch.nn.Module):
    """Yield `Sequential(activations_quantizer=NoisyAct, 0=Noisy[Conv|Linear])` blocks in module order."""
    for name, module in root.named_modules():
        if not isinstance(module, torch.nn.Sequential):
            continue
        act = module._modules.get("activations_quantizer")
        qmodule = module._modules.get("0")
        if isinstance(act, NoisyAct) and isinstance(qmodule, (NoisyConv2d, NoisyLinear)):
            yield name, module, act, qmodule


def normalize_noisyact_scales(model: torch.nn.Module) -> int:
    """
    Assumes a feed-forward stack of affine layers and ReLUs.
    Replaces each NoisyAct with an integer-output activation layer and
    rewrites the following quantized affine to absorb the removed dequantization:
        conv(s * q + z, W) + b == s * conv(q, W) + z-correction + b
    """
    root = _get_model_root(model)
    normalized_count = 0

    for name, container, act, qmodule in _iter_quantized_blocks(root):
        current_scale = float(torch.exp2(act.log_act_s.detach()).cpu().item())
        current_zero_point = float(act.act_b.detach().cpu().item())
        quantized_weight, weight_scale, weight_zero_point = _get_quantized_weight_params(qmodule)
        effective_bias = _get_effective_bias(qmodule)

        with torch.no_grad():
            container._modules["activations_quantizer"] = _RoundClampAct.from_noisy_act(act)
            container._modules["0"] = _make_exact_integer_affine_from_quantized(
                qmodule,
                quantized_weight,
                effective_bias,
                current_scale,
                current_zero_point,
                weight_scale,
                weight_zero_point,
            )

        normalized_count += 1
        logger.info(
            "Replaced NoisyAct in %s with inline round-clamp act and exact integer affine: act_scale=%s act_zero_point=%s.",
            name,
            current_scale,
            current_zero_point,
        )

    return normalized_count


def _iter_tensors(value):
    if torch.is_tensor(value):
        yield value
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_tensors(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _iter_tensors(item)


def _first_tensor(value):
    for tensor in _iter_tensors(value):
        return tensor
    return None


def _get_model_root(model: torch.nn.Module) -> torch.nn.Module:
    return model.model if hasattr(model, "model") else model


def _get_model_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def _get_batch_inputs(batch):
    if isinstance(batch, (list, tuple)):
        return batch[0]
    return batch


def _get_first_validation_batch(datamodule):
    if hasattr(datamodule, "setup"):
        for stage in ("test", "validate", "fit", None):
            try:
                datamodule.setup(stage=stage)
                break
            except TypeError:
                datamodule.setup(stage)
                break
            except Exception:
                continue

    val_loader = datamodule.val_dataloader()
    if isinstance(val_loader, dict):
        first_key = next(iter(val_loader))
        val_loader = val_loader[first_key]
    elif isinstance(val_loader, (list, tuple)):
        val_loader = val_loader[0]

    return next(iter(val_loader))


def _move_to_device(value, device: torch.device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, (list, tuple)):
        return type(value)(_move_to_device(item, device) for item in value)
    if isinstance(value, dict):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    return value


def _prepare_debug_inputs(model: torch.nn.Module, datamodule):
    batch = _get_first_validation_batch(datamodule)
    return _move_to_device(_get_batch_inputs(batch), _get_model_device(model))


def _remove_hook_handles(handles):
    for handle in handles:
        handle.remove()


def _iter_leaf_modules(root: torch.nn.Module):
    for name, module in root.named_modules():
        if name and not any(module.children()):
            yield name, module


def _canonicalize_debug_tensor(module: torch.nn.Module, tensor: torch.Tensor) -> torch.Tensor:
    if isinstance(module, _RoundClampAct) and not module.disable:
        scale = torch.exp2(module.log_act_s.detach()).to(tensor.device, tensor.dtype)
        zero_point = module.act_b.detach().to(tensor.device, tensor.dtype)
        return tensor * scale + zero_point
    return tensor


def _register_debug_hooks(model: torch.nn.Module, records: list[tuple[str, torch.nn.Module, object]]):
    handles = []
    root = _get_model_root(model)

    def make_hook(name):
        def hook(module, _inputs, output):
            records.append((name, module, output))

        return hook

    for name, module in _iter_leaf_modules(root):
        handles.append(module.register_forward_hook(make_hook(name)))
    return handles


def _register_activation_debug_hooks(
    model: torch.nn.Module,
    records: list[tuple[str, torch.nn.Module, object, object]],
):
    handles = []
    root = _get_model_root(model)

    def make_hook(name):
        def hook(module, inputs, output):
            records.append((name, module, inputs[0] if inputs else None, output))

        return hook

    for name, module in root.named_modules():
        if isinstance(module, (NoisyAct, _RoundClampAct)):
            handles.append(module.register_forward_hook(make_hook(name)))
    return handles


def _quantize_with_activation(module: NoisyAct, x: torch.Tensor) -> torch.Tensor:
    if module.disable:
        return x

    scale = torch.exp2(module.log_act_s.detach()).to(x.device, x.dtype)
    q = torch.exp2(module.log_act_q.detach()).to(x.device, x.dtype)
    zero_point = module.act_b.detach().to(x.device, x.dtype)

    module.Q.zero_point = zero_point
    module.Q.min_val = zero_point
    module.Q.max_val = zero_point + q - scale
    module.Q.scale = scale
    return module.Q.quantize(x)


def _is_material_activation_mismatch(
    diff: torch.Tensor,
    q_abs_diff: torch.Tensor | None,
    input_diff: torch.Tensor | None,
) -> tuple[bool, str]:
    """
    Heuristic filter to ignore tiny near-threshold flips and keep searching.
    """
    max_abs_diff = diff.max().item()
    mean_abs_diff = diff.mean().item()

    if q_abs_diff is None:
        return max_abs_diff > 1e-3 or mean_abs_diff > 1e-5, ""

    q_flip_count = int((q_abs_diff > 0).sum().item())
    q_flip_ratio = q_flip_count / max(q_abs_diff.numel(), 1)
    input_max_abs_diff = input_diff.max().item() if input_diff is not None else 0.0
    input_mean_abs_diff = input_diff.mean().item() if input_diff is not None else 0.0

    is_material = (
        q_flip_count >= 32
        or q_flip_ratio >= 1e-4
        or input_max_abs_diff >= 1e-4
        or input_mean_abs_diff >= 1e-5
        or mean_abs_diff >= 1e-4
    )
    summary = (
        f" q_flip_count={q_flip_count}"
        f" q_flip_ratio={q_flip_ratio}"
        f" input_max_abs_diff={input_max_abs_diff}"
        f" input_mean_abs_diff={input_mean_abs_diff}"
    )
    return is_material, summary


def debug_first_integer_noisyact_mismatch(
    reference_model: torch.nn.Module,
    transformed_model: torch.nn.Module,
    datamodule,
    atol: float = 1e-5,
    rtol: float = 1e-4,
):
    inputs = _prepare_debug_inputs(reference_model, datamodule)

    ref_records = []
    tr_records = []
    ref_handles = _register_activation_debug_hooks(reference_model, ref_records)
    tr_handles = _register_activation_debug_hooks(transformed_model, tr_records)

    try:
        with torch.no_grad():
            reference_model(inputs)
            transformed_model(inputs)
    finally:
        _remove_hook_handles(ref_handles + tr_handles)

    for idx, ((ref_name, ref_module, ref_input, ref_value), (tr_name, tr_module, tr_input, tr_value)) in enumerate(
        zip(ref_records, tr_records)
    ):
        if ref_name != tr_name:
            logger.warning(
                "Activation debug trace mismatch at step %d: reference=%s transformed=%s.",
                idx,
                ref_name,
                tr_name,
            )
            return

        ref_tensor = _first_tensor(ref_value)
        tr_tensor = _first_tensor(tr_value)
        if ref_tensor is None or tr_tensor is None:
            continue

        ref_tensor = ref_tensor.detach().float()
        tr_tensor = tr_tensor.detach().float()
        ref_input_tensor = _first_tensor(ref_input)
        tr_input_tensor = _first_tensor(tr_input)
        if ref_input_tensor is not None:
            ref_input_tensor = ref_input_tensor.detach().float()
        if tr_input_tensor is not None:
            tr_input_tensor = tr_input_tensor.detach().float()

        if getattr(ref_module, "disable", False) or getattr(tr_module, "disable", False):
            comparable_ref = ref_tensor
            comparable_tr = tr_tensor
            integer_error = 0.0
            q_ref = None
            q_tr = None
        else:
            scale = torch.exp2(ref_module.log_act_s.detach()).to(ref_tensor.device, ref_tensor.dtype)
            zero_point = ref_module.act_b.detach().to(ref_tensor.device, ref_tensor.dtype)
            comparable_ref = ref_tensor
            comparable_tr = tr_tensor * scale + zero_point
            integer_error = (tr_tensor - tr_tensor.round()).abs().max().item()
            q_ref = _quantize_with_activation(ref_module, ref_input_tensor)
            q_tr = tr_tensor

        if comparable_ref.shape != comparable_tr.shape:
            logger.warning(
                "First no-matching round-clamp act %s: shape mismatch reference=%s transformed=%s.",
                ref_name,
                tuple(comparable_ref.shape),
                tuple(comparable_tr.shape),
            )
            return

        if integer_error > atol:
            logger.warning(
                "First no-matching round-clamp act %s: transformed output is not integer, max_integer_error=%s.",
                ref_name,
                integer_error,
            )
            return

        if not torch.allclose(comparable_ref, comparable_tr, atol=atol, rtol=rtol):
            diff = (comparable_ref - comparable_tr).abs()
            input_diff = None
            input_diff_msg = ""
            if ref_input_tensor is not None and tr_input_tensor is not None:
                input_diff = (ref_input_tensor - tr_input_tensor).abs()
                input_diff_msg = (
                    f" input_max_abs_diff={input_diff.max().item()} "
                    f"input_mean_abs_diff={input_diff.mean().item()}"
                )

            q_diff_msg = ""
            q_abs_diff = None
            if q_ref is not None and q_tr is not None:
                q_abs_diff = (q_ref - q_tr).abs()
                q_flip_count = int((q_abs_diff > 0).sum().item())
                q_diff_msg = (
                    f" q_flip_count={q_flip_count}"
                    f" q_max_abs_diff={q_abs_diff.max().item()}"
                    f" act_scale={scale.item()}"
                )

            is_material, material_summary = _is_material_activation_mismatch(diff, q_abs_diff, input_diff)
            if not is_material:
                logger.info(
                    "Ignoring non-material round-clamp act mismatch %s: max_abs_diff=%s mean_abs_diff=%s.%s%s",
                    ref_name,
                    diff.max().item(),
                    diff.mean().item(),
                    input_diff_msg,
                    q_diff_msg + material_summary,
                )
                continue

            logger.warning(
                "First no-matching round-clamp act %s: max_abs_diff=%s mean_abs_diff=%s ref_range=%s transformed_dequantized_range=%s max_integer_error=%s.%s%s",
                ref_name,
                diff.max().item(),
                diff.mean().item(),
                tuple(comparable_ref.aminmax()),
                tuple(comparable_tr.aminmax()),
                integer_error,
                input_diff_msg,
                q_diff_msg,
            )
            return

    logger.info("No materially divergent round-clamp act layer found on the debug batch.")


def debug_first_divergence(
    reference_model: torch.nn.Module,
    transformed_model: torch.nn.Module,
    datamodule,
    atol: float = 1e-5,
    rtol: float = 1e-4,
):
    inputs = _prepare_debug_inputs(reference_model, datamodule)

    ref_records = []
    tr_records = []
    ref_handles = _register_debug_hooks(reference_model, ref_records)
    tr_handles = _register_debug_hooks(transformed_model, tr_records)

    try:
        with torch.no_grad():
            ref_output = reference_model(inputs)
            tr_output = transformed_model(inputs)
    finally:
        _remove_hook_handles(ref_handles + tr_handles)

    for idx, ((ref_name, ref_module, ref_value), (tr_name, tr_module, tr_value)) in enumerate(
        zip(ref_records, tr_records)
    ):
        if ref_name != tr_name:
            logger.warning(
                "Debug divergence trace mismatch at step %d: reference=%s transformed=%s.",
                idx,
                ref_name,
                tr_name,
            )
            return

        ref_tensor = _first_tensor(ref_value)
        tr_tensor = _first_tensor(tr_value)
        if ref_tensor is None or tr_tensor is None:
            continue

        ref_tensor = _canonicalize_debug_tensor(ref_module, ref_tensor.detach()).float()
        tr_tensor = _canonicalize_debug_tensor(tr_module, tr_tensor.detach()).float()
        if ref_tensor.shape != tr_tensor.shape:
            logger.warning(
                "First divergent layer %s: shape mismatch reference=%s transformed=%s.",
                ref_name,
                tuple(ref_tensor.shape),
                tuple(tr_tensor.shape),
            )
            return

        if isinstance(ref_module, NoisyAct) or isinstance(tr_module, _RoundClampAct):
            continue

        if not torch.allclose(ref_tensor, tr_tensor, atol=atol, rtol=rtol):
            diff = (ref_tensor - tr_tensor).abs()
            logger.warning(
                "First divergent layer %s: max_abs_diff=%s mean_abs_diff=%s ref_range=%s transformed_range=%s.",
                ref_name,
                diff.max().item(),
                diff.mean().item(),
                tuple(ref_tensor.aminmax()),
                tuple(tr_tensor.aminmax()),
            )
            return

    ref_out = _first_tensor(ref_output)
    tr_out = _first_tensor(tr_output)
    if ref_out is not None and tr_out is not None:
        diff = (ref_out.detach().float() - tr_out.detach().float()).abs()
        logger.info(
            "No hooked layer divergence found. Final output max_abs_diff=%s mean_abs_diff=%s.",
            diff.max().item(),
            diff.mean().item(),
        )
    else:
        logger.info("No hooked layer divergence found.")


def build_debug_reference_model(model_composer, quantizer, state_dict, use_cuda: bool):
    """
    Rebuild the pre-normalization fused model from scratch.
    This avoids deepcopy issues with non-leaf tensors inside the live model.
    """
    ref_model = model_composer.compose()
    ref_qmodel = quantizer.quantize(ref_model, in_place=True)
    ref_qmodel.load_state_dict(state_dict, strict=False)
    ref_qmodel.eval()
    if use_cuda:
        ref_qmodel = ref_qmodel.to("cuda")
    fuse_meannorm_into_conv_bias_and_remove(ref_qmodel)
    return ref_qmodel


def restore_plain_validation_step(model: torch.nn.Module):
    """
    Remove GDNSQ validation decoration so transformed inference-only models
    can be validated without quantizer-width bookkeeping.
    """
    model.validation_step = type(model).validation_step.__get__(model, type(model))


def fuse_meannorm_into_conv_bias_and_remove(model: torch.nn.Module) -> int:
    """
    Walk model, find MeanNorm layers whose previous sibling is (or contains) NoisyConv2d.
    Fuse MeanNorm into that conv's bias and remove the MeanNorm from the graph
    (rebuild Sequential without it, or replace with passthrough for non-Sequential parents).
    Returns number of fused pairs.
    """
    root = _get_model_root(model)
    mean_norm_types = (MeanNorm1d, MeanNorm2d, MeanNorm3d)
    fused_count = 0

    # Collect (parent, child_name, mean_norm_module, conv) for each MeanNorm we can fuse
    to_fuse = []
    for name, mod in root.named_modules():
        if not isinstance(mod, mean_norm_types):
            continue
        parts = name.split(".")
        child_name = parts[-1]
        parent_name = ".".join(parts[:-1]) if len(parts) > 1 else None
        parent = root.get_submodule(parent_name) if parent_name else root

        keys = list(parent._modules.keys())
        try:
            idx = keys.index(child_name)
        except ValueError:
            continue
        if idx == 0:
            continue
        prev_key = keys[idx - 1]
        prev_module = parent._modules[prev_key]
        conv = _get_noisy_conv(prev_module)
        if conv is None:
            continue

        to_fuse.append((parent, child_name, mod, conv))

    for parent, child_name, mean_norm, conv in to_fuse:
        with torch.no_grad():
            delta = mean_norm.bias.detach() - mean_norm.running_mean.detach()
            if conv.bias is None:
                conv.bias = torch.nn.Parameter(delta.clone().to(conv.weight.device))
            else:
                conv.bias.add_(delta.to(conv.bias.device))

        if isinstance(parent, torch.nn.Sequential):
            parent._modules.pop(child_name)
        else:
            setattr(parent, child_name, _PassThrough())
        fused_count += 1
        logger.info("Fused MeanNorm into conv bias and removed: %s", child_name)

    return fused_count


def print_weight_bias_stats(model: torch.nn.Module):
    """Print per-layer weight and bias value-sets for integer affine layers."""
    root = _get_model_root(model)
    for name, mod in root.named_modules():
        if isinstance(mod, (_ExactIntegerConv2d, _ExactIntegerLinear)):
            wvals = sorted(torch.unique(mod.weight.detach()).cpu().tolist())
            bvals = None
            if mod.bias is not None:
                bvals = sorted(torch.unique(mod.bias.detach()).cpu().tolist())
            logger.info(
                "layer=%s weight_values=%s bias_values=%s input_scale=%s input_zero_point=%s weight_scale=%s weight_zero_point=%s",
                name,
                wvals,
                bvals,
                float(mod.input_scale.detach().cpu().item()),
                float(mod.input_zero_point.detach().cpu().item()),
                sorted(torch.unique(mod.weight_scale.detach()).cpu().tolist()),
                sorted(torch.unique(mod.weight_zero_point.detach()).cpu().tolist()),
            )


def main():
    args = parse_args()

    config = load_and_validate_config(args.config)
    dataset_composer = DatasetComposer(config=config)
    model_composer = ModelComposer(config=config)
    quantizer = Quantizer(config=config)()
    validator = Validator(config=config)

    data = dataset_composer.compose()
    model = model_composer.compose()
    qmodel = quantizer.quantize(model, in_place=True)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    qmodel.load_state_dict(state, strict=False)
    qmodel.eval()

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        qmodel = qmodel.to("cuda")

    n_fused = fuse_meannorm_into_conv_bias_and_remove(qmodel)
    logger.info("Fused %d MeanNorm(s) into previous NoisyConv2d bias and removed from graph.", n_fused)

    n_normalized = normalize_noisyact_scales(qmodel)
    logger.info("Normalized %d NoisyAct scale(s) to 1.", n_normalized)
    restore_plain_validation_step(qmodel)

    if args.debug_divergence:
        debug_reference_model = build_debug_reference_model(
            model_composer=model_composer,
            quantizer=quantizer,
            state_dict=state,
            use_cuda=use_cuda,
        )
        logger.info("Checking inline round-clamp activations against original NoisyAct")
        debug_first_integer_noisyact_mismatch(debug_reference_model, qmodel, data)
        logger.info("Running side-by-side divergence debug on one validation batch")
        debug_first_divergence(debug_reference_model, qmodel, data)

    root = _get_model_root(qmodel)
    logger.info("Model (after fusion):\n%s", root)

    logger.info("Running validation on fused model")
    validator.validate(qmodel, datamodule=data)

    logger.info("Weight and bias value-set stats (unique counts):")
    print_weight_bias_stats(qmodel)

    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    torch.save({"state_dict": qmodel.state_dict()}, args.output)
    logger.info("Saved fused checkpoint to %s", args.output)


if __name__ == "__main__":
    main()
