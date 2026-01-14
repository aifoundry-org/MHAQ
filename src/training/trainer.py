from datetime import timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Union
from lightning import Callback
import lightning.pytorch as pl
import logging
import torch
from lightning.pytorch.accelerators import Accelerator
from lightning.pytorch.loggers import Logger
from lightning.pytorch.plugins import _PLUGIN_INPUT, Precision
from lightning.pytorch.profilers import Profiler
from lightning.pytorch.strategies import Strategy, DDPStrategy, SingleDeviceStrategy
from lightning.pytorch.trainer.connectors.accelerator_connector import _LITERAL_WARN
from lightning.pytorch.utilities.types import EVAL_DATALOADERS
from src.loggers import WandbLogger, TensorBoardLogger
from src.training.loops import SrEvalLoop
from src.quantization.gdnsq.calib.minmaxobserver import (
    MinMaxObserver,
    apply_mean_stats_activations,
    apply_quantile_weights_s,
)
from src.quantization.gdnsq.calib.hooks import register_lightning_activation_forward_hook

from src import callbacks as compose_callbacks
from src import loggers as compose_loggers

log = logging.getLogger("lightning.pytorch")


class Trainer(pl.Trainer):
    def __init__(
        self,
        config: Dict | None = None,
        *,
        accelerator: str | Accelerator = "auto",
        strategy: str | Strategy = "auto",
        devices: List[int] | str | int = "auto",
        num_nodes: int = 1,
        precision: (
            None
            | Literal[64]
            | Literal[32]
            | Literal[16]
            | Literal["transformer-engine"]
            | Literal["transformer-engine-float16"]
            | Literal["16-true"]
            | Literal["16-mixed"]
            | Literal["bf16-true"]
            | Literal["bf16-mixed"]
            | Literal["32-true"]
            | Literal["64-true"]
            | Literal["64"]
            | Literal["32"]
            | Literal["16"]
            | Literal["bf16"]
        ) = None,
        logger: Logger | Iterable[Logger] | bool | None = None,
        callbacks: List[Callback] | pl.Callback | None = None,
        fast_dev_run: int | bool = False,
        max_epochs: int | None = None,
        min_epochs: int | None = None,
        max_steps: int = -1,
        min_steps: int | None = None,
        max_time: str | timedelta | Dict[str, int] | None = None,
        limit_train_batches: int | float | None = None,
        limit_val_batches: int | float | None = None,
        limit_test_batches: int | float | None = None,
        limit_predict_batches: int | float | None = None,
        overfit_batches: int | float = 0,
        val_check_interval: int | float | None = None,
        check_val_every_n_epoch: int | None = 1,
        num_sanity_val_steps: int | None = None,
        log_every_n_steps: int | None = None,
        enable_checkpointing: bool | None = None,
        enable_progress_bar: bool | None = None,
        enable_model_summary: bool | None = None,
        accumulate_grad_batches: int = 1,
        gradient_clip_val: int | float | None = None,
        gradient_clip_algorithm: str | None = None,
        deterministic: bool | None | Literal["warn"] = None,
        benchmark: bool | None = None,
        inference_mode: bool = True,
        use_distributed_sampler: bool = True,
        profiler: Profiler | str | None = None,
        detect_anomaly: bool = False,
        barebones: bool = False,
        plugins: Optional[Union[_PLUGIN_INPUT, List[_PLUGIN_INPUT]]] = None,
        sync_batchnorm: bool = True,
        reload_dataloaders_every_n_epochs: int = 0,
        default_root_dir: str | Path | None = None
    ) -> None:
        if torch.cuda.device_count() > 1:
            strategy = DDPStrategy(
                find_unused_parameters=True,
            )
        else:
            strategy = "auto"

        if config:
            self.config = config
            tconfig = config.training

            max_epochs = tconfig.max_epochs
            log_every_n_steps = tconfig.log_every_n_steps
            callbacks = [
                getattr(compose_callbacks, _callback)(
                    **tconfig.callbacks[_callback].params
                )
                for _callback in tconfig.callbacks
            ]

            logger = [
                getattr(compose_loggers, _logger)(**tconfig.loggers[_logger].params)
                for _logger in tconfig.loggers
            ]

            if TensorBoardLogger not in logger:
                logger.append(TensorBoardLogger(save_dir="logs"))

            for _logger in logger:
                if isinstance(_logger, WandbLogger):
                    _logger.log_hyperparams(config.dict())

            check_val_every_n_epoch = tconfig.val_every_n_epochs
            val_check_interval = tconfig.val_check_interval
            # precision = "bf16-mixed"
            precision = "32"

        super().__init__(
            accelerator=accelerator,
            strategy=strategy,
            devices=devices,
            num_nodes=num_nodes,
            precision=precision,
            logger=logger,
            callbacks=callbacks,
            fast_dev_run=fast_dev_run,
            max_epochs=max_epochs,
            min_epochs=min_epochs,
            max_steps=max_steps,
            min_steps=min_steps,
            max_time=max_time,
            limit_train_batches=limit_train_batches,
            limit_val_batches=limit_val_batches,
            limit_test_batches=limit_test_batches,
            limit_predict_batches=limit_predict_batches,
            overfit_batches=overfit_batches,
            val_check_interval=val_check_interval,
            check_val_every_n_epoch=check_val_every_n_epoch,
            num_sanity_val_steps=num_sanity_val_steps,
            log_every_n_steps=log_every_n_steps,
            enable_checkpointing=enable_checkpointing,
            enable_progress_bar=enable_progress_bar,
            enable_model_summary=enable_model_summary,
            accumulate_grad_batches=accumulate_grad_batches,
            gradient_clip_val=gradient_clip_val,
            gradient_clip_algorithm=gradient_clip_algorithm,
            deterministic=deterministic,
            benchmark=benchmark,
            inference_mode=inference_mode,
            use_distributed_sampler=use_distributed_sampler,
            profiler=profiler,
            detect_anomaly=detect_anomaly,
            barebones=barebones,
            plugins=plugins,
            sync_batchnorm=sync_batchnorm,
            reload_dataloaders_every_n_epochs=reload_dataloaders_every_n_epochs,
            default_root_dir=default_root_dir,
        )

        if config:
            if config.model.type == "VISION_SR":
                sr_loop = SrEvalLoop(trainer=self.validate_loop.trainer,
                                     trainer_fn=self.validate_loop._trainer_fn,
                                     stage=self.validate_loop._stage,
                                     verbose=self.validate_loop.verbose,
                                     inference_mode=self.validate_loop.inference_mode)
                sr_loop_test = SrEvalLoop(trainer=self.test_loop.trainer,
                                     trainer_fn=self.test_loop._trainer_fn,
                                     stage=self.test_loop._stage,
                                     verbose=self.test_loop.verbose,
                                     inference_mode=self.test_loop.inference_mode)

                self.validate_loop = sr_loop
                self.test_loop = sr_loop_test

    def calibrate(
        self,
        model=None,
        dataloaders=None,
        ckpt_path=None,
        verbose=True,
        datamodule=None,
    ):
        if (
            "calibration" in self.config.quantization.__dict__
            and self.config.quantization.calibration
        ):

            log.info("\nPerforming calibration...")
            c_config = self.config.quantization.calibration

            if c_config.weight_bit != 0:
                apply_quantile_weights_s(model.model, wbits=c_config.weight_bit)
            else:
                log.info("Skipping weights calibration...")

            if c_config.act_bit != 0:
                observer_hook = MinMaxObserver()
                handlers = register_lightning_activation_forward_hook(
                    model.model, observer_hook
                )
                self.validate(model, dataloaders, ckpt_path, False, datamodule)

                for handler in handlers:
                    handler.remove()

                apply_mean_stats_activations(model.model, abits=c_config.act_bit)
            else:
                log.info("Skipping activations scales calibration...")

            log.info("\nChecking quality of calibration...")
            self.validate(model, dataloaders, ckpt_path, verbose, datamodule)

    def test(
        self,
        model: pl.LightningModule | None = None,
        dataloaders: pl.LightningDataModule | None = None,
        ckpt_path: _LITERAL_WARN | Path | None = None,
        verbose: bool = True,
        datamodule: pl.LightningDataModule | None = None,
        weights_only: bool | None = None,
    ):
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()  # shutdown DDP to allow single-GPU testing
        return super().test(
            model, dataloaders, ckpt_path, verbose, datamodule, weights_only
        )
