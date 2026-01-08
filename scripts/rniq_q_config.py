import os
import sys
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))
import torch
import torch.distributed
import argparse
from lightning.pytorch.callbacks import ModelCheckpoint

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
        # default="config/rniq_config_yolo11.yaml"
        default="config/rniq_config_resnet20_old.yaml"
    )
    return parser.parse_args()


def find_latest_checkpoint(trainer):
    """Return the most recently modified checkpoint file from the first
    ModelCheckpoint callback's directory, or None if none found."""
    ckpt_dir = None
    for cb in trainer.callbacks:
        if isinstance(cb, ModelCheckpoint):
            ckpt_dir = getattr(cb, "dirpath", None) or getattr(cb, "save_dir", None)
            if ckpt_dir:
                break

    if not ckpt_dir or not os.path.isdir(ckpt_dir):
        return None

    candidates = [
        os.path.join(ckpt_dir, f)
        for f in os.listdir(ckpt_dir)
        if f.endswith((".ckpt", ".pt", ".pth", ".th"))
    ]

    if not candidates:
        return None

    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


def main():
    args = parse_args()

    config = load_and_validate_config(args.config)
    dataset_composer = DatasetComposer(config=config)
    model_composer = ModelComposer(config=config)
    quantizer = Quantizer(config=config)()
    trainer = Trainer(config=config)
    val_trainer = Trainer(config=config, devices=1)

    data = dataset_composer.compose()
    model = model_composer.compose()

    print("Validate  model before quantization")
    val_trainer.validate(model, datamodule=data)

    qmodel = quantizer.quantize(model, in_place=True)

    print("Validate model after layers replacement")
    val_trainer.validate(qmodel, datamodule=data)
  
    print("Calibrating model initial weights and scales")
    val_trainer.calibrate(qmodel, datamodule=data)

    # Finetune model
    trainer.fit(qmodel, datamodule=data)

    # Only rank 0 performs testing and saving
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group() # shutdown DDP to allow single-GPU testing
        if trainer.global_rank != 0:
            return

    best_ckpt_path = find_latest_checkpoint(trainer)
    print(f"Using latest checkpoint from dir: {best_ckpt_path}")
    val_trainer.test(qmodel, datamodule=data, ckpt_path=best_ckpt_path)

if __name__ == "__main__":
    main()
