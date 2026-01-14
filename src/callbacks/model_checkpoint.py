from datetime import timedelta
from typing import Literal
from lightning.pytorch.callbacks import ModelCheckpoint
import logging
from pathlib import Path
from typing_extensions import override
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from lightning.pytorch.utilities.rank_zero import rank_prefixed_message, rank_zero_warn
from src.quantization.gdnsq.utils import model_stats

log = logging.getLogger(__name__)

class CustomModelCheckpoint(ModelCheckpoint):
    def __init__(
        self,
        dirpath: str | None = None,
        filename: str | None = None,
        monitor: str | None = None,
        verbose: bool = True,
        save_last: bool | None | Literal["link"] = True,
        save_top_k: int = 4,
        save_on_exception: bool = False,
        save_weights_only: bool = False,
        mode: str = "min",
        auto_insert_metric_name: bool = False,
        every_n_train_steps: int | None = None,
        train_time_interval: timedelta | None = None,
        every_n_epochs: int | None = None,
        save_on_train_epoch_end: bool | None = None,
        enable_version_counter: bool = True,
    ):
        super().__init__(
            dirpath,
            filename,
            monitor,
            verbose,
            save_last,
            save_top_k,
            save_on_exception,
            save_weights_only,
            mode,
            auto_insert_metric_name,
            every_n_train_steps,
            train_time_interval,
            every_n_epochs,
            save_on_train_epoch_end,
            enable_version_counter,
        )


class NoiseModelCheckpoint(ModelCheckpoint):
    def __init__(
        self,
        dirpath: str | Path | None = None,
        filename: str | None = None,
        monitor: str | None = None,
        verbose: bool = False,
        save_last: bool | None | Literal["link"] = None,
        save_top_k: int = 3,
        save_on_exception: bool = False,
        save_weights_only: bool = False,
        mode: str = "min",
        auto_insert_metric_name: bool = False,
        every_n_train_steps: int | None = None,
        train_time_interval: timedelta | None = None,
        every_n_epochs: int | None = None,
        save_on_train_epoch_end: bool | None = False,
        enable_version_counter: bool = True,
    ):
        save_on_train_epoch_end = False
        super().__init__(
            dirpath,
            filename,
            monitor,
            verbose,
            save_last,
            save_top_k,
            save_on_exception,
            save_weights_only,
            mode,
            auto_insert_metric_name,
            every_n_train_steps,
            train_time_interval,
            every_n_epochs,
            save_on_train_epoch_end,
            enable_version_counter,
        )
        self.tracking_metric = False
        self.log_rank_zero_only = False

    @staticmethod
    def _log_info(
        trainer: "pl.Trainer", message: str, log_rank_zero_only: bool
    ) -> None:
        rank = trainer.global_rank if trainer.world_size > 1 else None
        message = rank_prefixed_message(message, rank)
        if rank is None or not log_rank_zero_only or rank == 0:
            log.info(message)

    @override
    def on_validation_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        if self.tracking_metric:
            return super().on_validation_end(trainer, pl_module)

        try:
            if model_stats.is_converged(pl_module):
                self._log_info(
                    trainer, "Started saving checkpoints", self.log_rank_zero_only
                )
                self.tracking_metric = True
        except AttributeError: # Assuming that model is not quantized, that's why is_converged is falling
            pass
