from datetime import timedelta
from typing import Literal
from lightning.pytorch.callbacks import ModelCheckpoint


class CustomModelCheckpoint(ModelCheckpoint):
    def __init__(
        self,
        dirpath: str | None = None,
        filename: str | None = None,
        monitor: str | None = None,
        verbose: bool = True,
        save_last: bool | None | Literal["link"] = None,
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
