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
from src.training.trainer import Trainer

torch.set_float32_matmul_precision('high')

def parse_args():
    parser = argparse.ArgumentParser(description="Run RNIQ quantization.")
    parser.add_argument(
        "--config", 
        type=str, 
        required=False, 
        help="Path to the configuration file (YAML).",
        default="config/rniq_config_yolo11.yaml"
        # default="config/rniq_config_resnet20_new_4bit.yaml"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    config = load_and_validate_config(args.config)
    dataset_composer = DatasetComposer(config=config)
    model_composer = ModelComposer(config=config)
    quantizer = Quantizer(config=config)()
    trainer = Trainer(config=config)

    data = dataset_composer.compose()
    model = model_composer.compose()

    # Validate  model before quantization
    trainer.validate(model, datamodule=data)

    qmodel = quantizer.quantize(model, in_place=True)

    # Validate model after layers replacement
    trainer.validate(qmodel, datamodule=data)
  
    # Calibrating model initial weights and scales if defined in config
    # trainer.calibrate(qmodel, datamodule=data)

    # Finetune model
    trainer.fit(qmodel, datamodule=data)

    # Test model after quantization
    trainer.test(qmodel, datamodule=data, ckpt_path="best")

if __name__ == "__main__":
    main()
