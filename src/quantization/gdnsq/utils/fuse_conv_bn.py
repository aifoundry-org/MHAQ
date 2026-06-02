import torch

from src.quantization.gdnsq.layers.gdnsq_conv2d import NoisyConv2d
from src.quantization.gdnsq.layers.gdnsq_linear import NoisyLinear
from src.quantization.gdnsq.layers.exact_integer_conv2d import ExactIntegerConv2d
from src.loggers.default_logger import logger

def print_weight_bias_stats(model: torch.nn.Module):
    """Print per-layer weight value-sets for integer affine layers."""
    root = _get_model_root(model)
    for name, mod in root.named_modules():
        if isinstance(mod, ExactIntegerConv2d):
            wvals = sorted(torch.unique(mod.weight.detach()).cpu().tolist())
            logger.info(
                "layer=%s weight_values=%s",
                name,
                wvals,
            )

def _iter_quantized_affines(root: torch.nn.Module):
    """Yield fused GDNSQ affine layers in module order."""
    for name, module in root.named_modules():
        if isinstance(module, (NoisyConv2d, NoisyLinear)):
            parts = name.split(".")
            child_name = parts[-1]
            parent_name = ".".join(parts[:-1]) if len(parts) > 1 else ""
            parent = root.get_submodule(parent_name) if parent_name else root
            yield name, parent, child_name, module, module

def _make_exact_integer_affine_from_quantized(
    module: NoisyConv2d | NoisyLinear,
    input_scale: torch.Tensor,
    post_scale: torch.Tensor | None = None,
    post_shift: torch.Tensor | None = None,
) -> torch.nn.Module:
    if isinstance(module, NoisyConv2d):
        return ExactIntegerConv2d(module, input_scale, post_scale, post_shift)
    raise ValueError(f"Unsupported module type: {type(module)}")

def _get_model_root(model: torch.nn.Module) -> torch.nn.Module:
    return model.model if hasattr(model, "model") else model

def materialize_exact_integer_convs_no_batchnorm_fuse(model: torch.nn.Module) -> int:
    """
    Replace each ``NoisyConv2d`` with ``ExactIntegerConv2d`` **without** BN folding.

    Used only by ``gdnsq_q_binarize_edges.py``: it reloads the ``state_dict`` from
    **this script's** ``--output``. That dict expects ``ExactIntegerConv2d`` at the
    same submodule paths, while ``quantize()`` only constructs ``NoisyConv2d``.
    """
    root = _get_model_root(model)
    n = 0
    affine_sites = list(_iter_quantized_affines(root))
    for name, parent, child_name, act, qmodule in affine_sites:
        if not isinstance(qmodule, NoisyConv2d):
            continue
        ref = qmodule.weight
        current_scale = torch.exp2(act.log_act_s).to(device=ref.device, dtype=ref.dtype)
        with torch.no_grad():
            exact_affine = _make_exact_integer_affine_from_quantized(
                qmodule,
                current_scale,
                None,
                None,
            )
        parent._modules[child_name] = exact_affine
        n += 1
        logger.info("Materialized ExactIntegerConv2d at %s (checkpoint already fused; no BN fold).", name)
    return n

def fuse_batchnorm_and_normalize_activation_scales(model: torch.nn.Module) -> int:
    """
    Walk the model once to:
      1) Fuse BatchNorm layers into the preceding NoisyConv2d and remove the BatchNorm from the graph.
      2) Replace each fused activation-aware affine layer with an inline integer-output activation
         and an exact integer affine that absorbs both the activation dequantization and any fused
         BatchNorm post-affine parameters.
    Returns (n_fused_batchnorm, n_normalized_activations).
    """
    root = _get_model_root(model)
    batch_norm_types = (
        torch.nn.BatchNorm1d,
        torch.nn.BatchNorm2d,
        torch.nn.BatchNorm3d,
    )
    fused_count = 0

    # Materialize the list of quantized affine sites first to avoid mutating
    # the module hierarchy (removing BatchNorm, replacing children) while
    # PyTorch is still traversing it via named_modules.
    affine_sites = list(_iter_quantized_affines(root))

    # Single logical pass over quantized affine layers:
    # for each one, optionally fuse a following BatchNorm and immediately
    # rewrite the node into an exact integer affine module with inlined activation.
    for name, parent, child_name, act, qmodule in affine_sites:
        ref = qmodule.weight  # device/dtype reference
        current_scale = torch.exp2(act.log_act_s).to(device=ref.device, dtype=ref.dtype)
        post_scale = None
        post_shift = None

        # If this affine is a NoisyConv2d and is immediately followed by a BatchNorm,
        # fold that BatchNorm into per-channel post_scale/post_shift and remove it.
        if isinstance(qmodule, NoisyConv2d):
            keys = list(parent._modules.keys())
            try:
                idx = keys.index(child_name)
            except ValueError:
                idx = -1

            if 0 <= idx < len(keys) - 1:
                next_key = keys[idx + 1]
                next_module = parent._modules[next_key]
                if isinstance(next_module, batch_norm_types):
                    batch_norm = next_module
                    with torch.no_grad():
                        mean = batch_norm.running_mean
                        var = batch_norm.running_var
                        gamma = batch_norm.weight
                        beta = batch_norm.bias
                        std = torch.sqrt(var + batch_norm.eps)
                        bn_scale = gamma / std
                        bn_post_shift = beta - mean * bn_scale
                        post_scale = bn_scale.to(device=ref.device, dtype=ref.dtype)
                        post_shift = bn_post_shift.to(device=ref.device, dtype=ref.dtype)

                    # Remove the BatchNorm node from the execution graph.
                    # For Sequential containers we can drop it entirely; for
                    # other parents (e.g., ConvBlock with a .bn attribute) we
                    # must leave a placeholder Module so attribute access in
                    # forward() still works.
                    if isinstance(parent, torch.nn.Sequential):
                        parent._modules.pop(next_key, None)
                    else:
                        setattr(parent, next_key, torch.nn.Identity())
                    fused_count += 1
                    logger.info(
                        "Removed BatchNorm and folded into %s: post_scale/post_shift.",
                        name,
                    )

        with torch.no_grad():
            exact_affine = _make_exact_integer_affine_from_quantized(
                qmodule,
                current_scale,
                post_scale,
                post_shift,
            )
            parent._modules[child_name] = exact_affine

        # Count of normalized activations is no longer returned, but we keep
        # this transformation for side effects and detailed logging.
        logger.info(
            "Replaced fused activation in %s with inline round-clamp act and exact integer affine: act_scale=%s.",
            name,
            current_scale.cpu().item() if current_scale.numel() == 1 else current_scale.flatten()[0].cpu().item(),
        )

    return fused_count