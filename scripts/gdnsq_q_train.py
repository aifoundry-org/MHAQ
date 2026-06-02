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
from src.training.trainer import Trainer, Validator
from src.loggers.default_logger import logger

torch.set_float32_matmul_precision('high')

def parse_args():
    parser = argparse.ArgumentParser(description="Run GDNSQ quantization.")
    parser.add_argument(
        "--config", 
        type=str, 
        required=False, 
        help="Path to the configuration file (YAML).",
        # default="config/gdnsq_config_yolo11.yaml"
        default="config/gdnsq_config_resnet20_cifar100_aewgs_w1a1.yaml"
        # default="config/gdnsq_config_rfdn.yaml"
    )
    return parser.parse_args()


def get_best_checkpoint_path(trainer):
    checkpoint_callback = trainer.checkpoint_callback
    if checkpoint_callback is None or not checkpoint_callback.best_model_path:
        return "best"
    return checkpoint_callback.best_model_path


def main():
    args = parse_args()

    config = load_and_validate_config(args.config)
    dataset_composer = DatasetComposer(config=config)
    model_composer = ModelComposer(config=config)
    quantizer = Quantizer(config=config)()
    validator = Validator(config=config)
    trainer = Trainer(config=config)

    data = dataset_composer.compose()
    model = model_composer.compose()

    logger.info(f"Validate Model before quantization:\n{model}")
    validator.validate(model, datamodule=data)

    qmodel = quantizer.quantize(model, in_place=True)

    logger.info("Validate model after layers replacement")
    validator.validate(qmodel, datamodule=data)
  
    logger.info("Calibrating model initial weights and scales")
    validator.calibrate(qmodel, datamodule=data)

    logger.info(f"Model after calibration:\n{qmodel}")
    # Finetune model
    trainer.fit(qmodel, datamodule=data)

    validator.test(qmodel, datamodule=data, ckpt_path=get_best_checkpoint_path(trainer))

    if config.quantization.fuse_batchnorm:
        logger.info("Performing batchnorm fuse")
        n_fused_batchnorm = quantizer.fuse_conv_bn(qmodel)
        logger.info(
        "Fused %d BatchNorm layer(s) into previous NoisyConv2d and removed from graph.",
        n_fused_batchnorm,
        )
        ckpt_path = validator.save_checkpoint()
        validator.test(qmodel, datamodule=data, ckpt_path=ckpt_path)

    validator.predict(qmodel, datamodule=data)

if __name__ == "__main__":
    main()
