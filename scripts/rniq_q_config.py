import os
import sys
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))
import torch
import torch.distributed
import argparse
import tempfile
from pathlib import Path
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


def get_best_ckpt_path(trainer):
    """Return best_model_path from any ModelCheckpoint callback, or None."""
    for cb in list(getattr(trainer, "callbacks", [])) + list(getattr(trainer, "checkpoint_callbacks", [])):
        if isinstance(cb, ModelCheckpoint):
            bp = getattr(cb, "best_model_path", None)
            if bp:
                return bp
    return None


# inline simple checkpoint save/wait in main for brevity
def sync_pre_checkpoint(path):
    """Ensure all ranks see the pre-fit checkpoint: barrier when DDP, otherwise poll."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()
        return
    import time
    p = Path(path)
    while not p.exists():
        time.sleep(1)


def load_pre_checkpoint(path, qmodel):
    """Load checkpoint into qmodel if path exists."""
    p = Path(path)
    if p.exists():
        ckpt = torch.load(str(p), map_location="cpu")
        qmodel.load_state_dict(ckpt.get("state_dict", {}), strict=False)


def main():
    args = parse_args()

    config = load_and_validate_config(args.config)
    dataset_composer = DatasetComposer(config=config)
    model_composer = ModelComposer(config=config)
    quantizer = Quantizer(config=config)()
    val_trainer = Trainer(config=config, devices=1)
    trainer = Trainer(config=config)

    data = dataset_composer.compose()
    model = model_composer.compose()
    
    # path where pre-fit checkpoint will be saved/loaded (use a temporary shared file and remove after run)
    pre_ckpt_path = Path(tempfile.gettempdir()) / "prequant_latest.ckpt"

    global_zero = trainer.is_global_zero
    print("global_zero:", global_zero)

    # Run heavy pre-fit ops only on rank 0
    if global_zero:
        print("Validate model before quantization")
        val_trainer.validate(model, datamodule=data)
    
    qmodel = quantizer.quantize(model, in_place=True)
    
    if global_zero:
        print("Validate model after layers replacement")
        val_trainer.validate(qmodel, datamodule=data)
        print("Calibrating model initial weights and scales")
        val_trainer.calibrate(qmodel, datamodule=data)

        # Save to a temporary shared file (system temp dir). We'll remove it after run.
        torch.save({"state_dict": qmodel.state_dict()}, str(pre_ckpt_path))
        # let other ranks proceed
        sync_pre_checkpoint(pre_ckpt_path)

    # On non-zero ranks, wait and load checkpoint now 
    sync_pre_checkpoint(pre_ckpt_path)
    load_pre_checkpoint(pre_ckpt_path, qmodel)

    print("Starting quantization-aware training")
    trainer.fit(qmodel, datamodule=data)

    # Shutdown DDP for single-rank testing and only run test on rank 0
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

    if global_zero:
        best_ckpt_path = get_best_ckpt_path(trainer)
        print(f"Using latest checkpoint from dir: {best_ckpt_path}")
        val_trainer.test(qmodel, datamodule=data, ckpt_path=best_ckpt_path)
        # Remove temporary prequant checkpoint file
        pre_ckpt_path.unlink(missing_ok=True)

if __name__ == "__main__":
    main()
