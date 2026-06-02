import argparse
import os
import resource
import sys

import torch

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from src.config.config_loader import load_and_validate_config
from src.data.compose.composer import DatasetComposer
from src.models.compose.composer import ModelComposer
from src.quantization.quantizer import Quantizer
from src.training.trainer import Validator

torch.set_float32_matmul_precision("high")


def parse_args(default_mode: str | None = None):
    parser = argparse.ArgumentParser(description="Run GDNSQ evaluation.")
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
        help="Path to the checkpoint (.ckpt) to use for evaluation.",
    )
    if default_mode is None:
        parser.add_argument(
            "--mode",
            type=str,
            default="validate",
            choices=("validate", "predict"),
            required=False,
            help="Evaluation mode to run.",
        )
    return parser.parse_args()


def build_quantized_model(config):
    data = DatasetComposer(config=config).compose()
    model = ModelComposer(config=config).compose()
    qmodel = Quantizer(config=config)().quantize(model, in_place=True)
    return data, qmodel


def main(default_mode: str | None = None):
    args = parse_args(default_mode)
    mode = default_mode or args.mode

    config = load_and_validate_config(args.config)
    data, qmodel = build_quantized_model(config)
    validator = Validator(config=config, logger=False)

    getattr(validator, mode)(qmodel, datamodule=data, ckpt_path=args.checkpoint)


if __name__ == "__main__":
    main()
