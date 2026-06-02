import argparse
import os
import resource
import sys
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from src.config.config_loader import load_and_validate_config
from src.data.compose.composer import DatasetComposer
from src.models.compose.composer import ModelComposer
from src.quantization.quantizer import Quantizer
from src.training.trainer import Trainer, Validator
from src.loggers.default_logger import logger

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

torch.set_float32_matmul_precision('highest')
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False


def parse_args():
    parser = argparse.ArgumentParser(description="Run GDNSQ quantization.")
    parser.add_argument(
        "--config",
        type=str,
        required=False,
        help="Path to the configuration file (YAML).",
        # default="config/gdnsq_config_resnet20_cifar100_ste_w1a1.yaml"
        # default="config/gdnsq_config_resnet20_old.yaml"
        default="config/gdnsq_config_resnet20_new.yaml"
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        required=False,
        help="Path to resume/load checkpoint",
        default=None
    )
    return parser.parse_args()


def main():
    args = parse_args()

    onnx_file_path = "resnet20_cifar100_w2a2.onnx"

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

    qmodel.train()
    trainer.fit(qmodel, datamodule=data, ckpt_path=args.ckpt)

    logger.info("Final Validation of PyTorch Model:")
    validator.validate(qmodel, datamodule=data)

    onnx_acc = trainer.export(
        model=qmodel,
        datamodule=data,
        onnx_file_path=onnx_file_path,
        use_simplifier=False
    )

    logger.info(f"ONNX Accuracy: {onnx_acc:.2f}%")


if __name__ == "__main__":
    main()
