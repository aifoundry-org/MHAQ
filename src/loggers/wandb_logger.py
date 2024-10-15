from typing import Any, Literal, Optional, Union
from lightning.fabric.utilities.types import _PATH
from lightning.pytorch import loggers as pl_loggers
import uuid
from datetime import datetime


class WandbLogger(pl_loggers.WandbLogger):
    def __init__(
        self,
        name: str | None = None,
        save_dir: _PATH = ".",
        version: str | None = None,
        offline: bool = False,
        dir: _PATH | None = None,
        id: str | None = None,
        anonymous: bool | None = None,
        project: str | None = None,
        log_model: bool | Literal["all"] = False,
        experiment: Any | None = None,
        prefix: str = "",
        checkpoint_name: str | None = None,
        note: str | None = None,
        **kwargs: Any
    ) -> None:
        version = (
            str(uuid.uuid4())[:6] + "_" +
            datetime.now().strftime("%Y-%m-%d %H_%M")
        )
        name = name + "_" + str(uuid.uuid4())[:6]
        project = "MHAQ"
        dir = "logs"
        save_dir = "logs"
        super().__init__(
            name,
            save_dir,
            version,
            offline,
            dir,
            id,
            anonymous,
            project,
            log_model,
            experiment,
            prefix,
            checkpoint_name,
            **kwargs
        )
        if note:
            self.experiment.notes = note
