import os
import sys
import resource

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))
import torch
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from src.config.config_loader import load_and_validate_config
from src.data.compose.composer import DatasetComposer
from src.models.compose.composer import ModelComposer
from src.quantization.quantizer import Quantizer
from src.training.trainer import Validator
from src.loggers.default_logger import logger

torch.set_float32_matmul_precision("high")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Script to fuse Batchnor into Convolution layer"
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
        required=False,
        default="fused.ckpt",
        help="Output checkpoint path.",
    )
    return parser.parse_args()


def build_quantized_model(config):
    model = ModelComposer(config=config).compose()
    quantizer = Quantizer(config=config)()
    qmodel = quantizer.quantize(model, in_place=True)
    return qmodel, quantizer


def main():
    args = parse_args()

    config = load_and_validate_config(args.config)
    data = DatasetComposer(config=config).compose()
    qmodel, quantizer = build_quantized_model(config)
    validator = Validator(config=config, logger=False, callbacks=False)

    logger.info("Validating checkpoint before fusion")
    before_metrics = validator.validate(
        qmodel,
        datamodule=data,
        ckpt_path=args.checkpoint,
    )
    logger.info("Validation before fusion: %s", before_metrics)

    logger.info("Performing batchnorm fuse")
    n_fused_batchnorm = quantizer.fuse_conv_bn(qmodel)
    logger.info("Fused %d BatchNorm layer(s).", n_fused_batchnorm)

    logger.info("Validating checkpoint after fusion")
    after_metrics = validator.validate(qmodel, datamodule=data)
    logger.info("Validation after fusion: %s", after_metrics)

    validator.save_checkpoint(filepath=args.output)
    logger.info("Saved fused checkpoint to %s", args.output)


if __name__ == "__main__":
    main()
