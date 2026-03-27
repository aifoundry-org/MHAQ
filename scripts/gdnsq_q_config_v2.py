import os
import sys
import re
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))
import torch
import argparse
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from scripts.gdnsq_q_config import run
from src.config.config_loader import load_and_validate_config
from src.data.compose.composer import DatasetComposer
from src.models.compose.composer import ModelComposer
from src.quantization.quantizer import Quantizer
from src.training.trainer import Trainer, Validator
from src.loggers.default_logger import logger

torch.set_float32_matmul_precision('high')

LOG_CALLBACK_NAMES = ("PrintClassificationMetrics", "PrintSrMetrics")
LOG_FOLDER_BY_GRAD_NOISE = {
    "BER3": "ber_3",
    "BER1": "ber_1",
    "NORM3": "norm_3",
    "NORM1": "norm_1",
    "UNIFORM": "uniform",
    "ROUNDING": "rounding",
}
LR_FILENAME_PREFIX = "lr_"
LR_FILENAME_SUFFIX = ".log"
INVALID_FILENAME_CHARS_PATTERN = re.compile(r"\.")


def _normalize_lr_value(lr_value: float | str) -> str:
    lr_text = str(lr_value).strip()
    return INVALID_FILENAME_CHARS_PATTERN.sub("_", lr_text)


def _build_metrics_log_path(logs_folder: str, grad_noise: str, lr_value: float | str) -> str:
    noise_folder = LOG_FOLDER_BY_GRAD_NOISE[grad_noise]
    lr_suffix = _normalize_lr_value(lr_value)
    target_dir = Path(logs_folder) / noise_folder
    target_dir.mkdir(parents=True, exist_ok=True)

    base_name = f"{LR_FILENAME_PREFIX}{lr_suffix}"
    suffix_index = 1
    candidate_path = target_dir / f"{base_name}_{suffix_index}{LR_FILENAME_SUFFIX}"

    while candidate_path.exists():
        suffix_index += 1
        candidate_path = target_dir / f"{base_name}_{suffix_index}{LR_FILENAME_SUFFIX}"

    return str(candidate_path)


def _override_metrics_log_filenames(config, grad_noise: str, logs_folder: str, lr_value: float | str) -> None:
    callbacks = getattr(config.training, "callbacks", {}) or {}
    if not isinstance(callbacks, dict):
        return

    for callback_name in LOG_CALLBACK_NAMES:
        callback = callbacks.get(callback_name)
        if callback is None or callback.params is None:
            continue
        callback.params["filename"] = _build_metrics_log_path(logs_folder, grad_noise, lr_value)



def _apply_cli_overrides(config, grad_noise: str | None, logs_folder: str | None, lr_value: float | None) -> None:
    params = getattr(config.quantization, "params", None)
    if params is None:
        raise ValueError("Quantization params are required for CLI overrides.")

    if grad_noise is not None:
        params.grad_noise = grad_noise
        logger.info(f"Override grad_noise from CLI: {grad_noise}")

    if lr_value is not None:
        config.training.learning_rate = lr_value
        logger.info(f"Override learning_rate from CLI: {lr_value}")

    if logs_folder is not None:
        _override_metrics_log_filenames(config, params.grad_noise, logs_folder, config.training.learning_rate)
        logger.info(f"Override metrics logs folder from CLI: {logs_folder}; noise={params.grad_noise}; lr={config.training.learning_rate}")        


def parse_args():
    parser = argparse.ArgumentParser(description="Run GDNSQ quantization.")
    parser.add_argument(
        "--config", 
        type=str, 
        required=False, 
        help="Path to the configuration file (YAML).",
        default="config/gdnsq_config_rfdn.yaml"
    )

    parser.add_argument(
        "--grad-noise",
        type=str,
        required=False,
        help="Override STE gradient noise type: BER3, BER1, NORM3, NORM1, UNIFORM, ROUNDING.",
        default="BER3"
    )

    parser.add_argument(
        "--lr",
        dest="learning_rate",
        type=float,
        required=False,
        help="training learning rate",
    )

    parser.add_argument(
        "--mitrics-folder",
        type=str,
        required=False,
        help="Base folder for metrics logs. Final path: mitrics-folder/noise/lr_XXX[_X].log",
        default="metrics_logs"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_and_validate_config(args.config)
    _apply_cli_overrides(config, args.grad_noise, args.mitrics_folder, args.learning_rate)
    run(config)
    

if __name__ == "__main__":
    main()
