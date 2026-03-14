"""
Fuse MeanNorm into previous NoisyConv2d bias, remove MeanNorm from graph,
run validation, print weight/bias value-set stats, and save fused checkpoint.
"""
import os
import sys
import resource
from collections import OrderedDict

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))
import torch
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from src.config.config_loader import load_and_validate_config
from src.data.compose.composer import DatasetComposer
from src.models.compose.composer import ModelComposer
from src.quantization.quantizer import Quantizer
from src.quantization.gdnsq.layers.gdnsq_conv2d import NoisyConv2d
from src.quantization.gdnsq.layers.gdnsq_linear import NoisyLinear
from src.quantization.gdnsq.layers.gdnsq_mean_norm import MeanNorm1d, MeanNorm2d, MeanNorm3d
from src.training.trainer import Validator
from src.loggers.default_logger import logger

torch.set_float32_matmul_precision("high")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fuse MeanNorm into previous NoisyConv2d bias, validate, print stats, save fused checkpoint."
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
    return parser.parse_args()


class _PassThrough(torch.nn.Module):
    """Passthrough so a slot in the graph can be removed semantically (conv output goes to next layer)."""

    def forward(self, x):
        return x


def _get_noisy_conv(module):
    """Return NoisyConv2d from module or from Sequential(..., '0'=NoisyConv2d)."""
    if isinstance(module, NoisyConv2d):
        return module
    if isinstance(module, torch.nn.Sequential) and "0" in module._modules:
        sub = module._modules["0"]
        if isinstance(sub, NoisyConv2d):
            return sub
    return None


def fuse_meannorm_into_conv_bias_and_remove(model: torch.nn.Module) -> int:
    """
    Walk model, find MeanNorm layers whose previous sibling is (or contains) NoisyConv2d.
    Fuse MeanNorm into that conv's bias and remove the MeanNorm from the graph
    (rebuild Sequential without it, or replace with passthrough for non-Sequential parents).
    Returns number of fused pairs.
    """
    root = model.model if hasattr(model, "model") else model
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
            new_od = OrderedDict((k, m) for k, m in parent._modules.items() if k != child_name)
            parent._modules.clear()
            parent._modules.update(new_od)
        else:
            setattr(parent, child_name, _PassThrough())
        fused_count += 1
        logger.info("Fused MeanNorm into conv bias and removed: %s", child_name)

    return fused_count


def print_weight_bias_stats(model: torch.nn.Module):
    """Print per-layer weight and bias value-sets for NoisyConv2d/NoisyLinear."""
    root = model.model if hasattr(model, "model") else model
    for name, mod in root.named_modules():
        if isinstance(mod, NoisyConv2d):
            qw = mod.Q.quantize(mod.weight.detach())
            wvals = sorted(torch.unique(qw).cpu().tolist())
            bvals = None
            if mod.bias is not None:
                if hasattr(mod, "Q_b") and mod.Q_b is not None:
                    qb = mod.Q_b.quantize(mod.bias.detach())
                    bvals = sorted(torch.unique(qb).cpu().tolist())
                else:
                    bvals = sorted(torch.unique(mod.bias.detach()).cpu().tolist())
            logger.info("layer=%s weight_values=%s bias_values=%s", name, wvals, bvals)
        elif isinstance(mod, NoisyLinear):
            qw = mod.Q.quantize(mod.weight.detach())
            wvals = sorted(torch.unique(qw).cpu().tolist())
            bvals = None
            if mod.bias is not None:
                bvals = sorted(torch.unique(mod.bias.detach()).cpu().tolist())
            logger.info("layer=%s weight_values=%s bias_values=%s", name, wvals, bvals)


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

    if torch.cuda.is_available():
        qmodel = qmodel.to("cuda")

    n_fused = fuse_meannorm_into_conv_bias_and_remove(qmodel)
    logger.info("Fused %d MeanNorm(s) into previous NoisyConv2d bias and removed from graph.", n_fused)

    logger.info("Running validation on fused model")
    validator.validate(qmodel, datamodule=data)

    logger.info("Weight and bias value-set stats (unique counts):")
    print_weight_bias_stats(qmodel.cuda())

    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    torch.save({"state_dict": qmodel.state_dict()}, args.output)
    logger.info("Saved fused checkpoint to %s", args.output)


if __name__ == "__main__":
    main()
