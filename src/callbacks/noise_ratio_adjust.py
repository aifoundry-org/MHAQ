import lightning.pytorch as pl
import torch
import logging
import numpy as np

from lightning.pytorch.callbacks.callback import Callback
from lightning.pytorch import Trainer, LightningModule

logger = logging.getLogger("lightning.pytorch")

class RandNoiseScale(Callback):
    def __init__(self, reduce_scale=2) -> None:
        self.q_loss = 0
        self.noise_ratio = None
        super().__init__()


    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.noise_ratio = pl_module._noise_ratio
        return super().on_fit_start(trainer, pl_module)


    def on_train_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs, batch: torch.Any, batch_idx: int) -> None:
        self.q_loss += pl_module.wrapped_criterion.wloss.mean().item() + pl_module.wrapped_criterion.aloss.mean().item()
        return super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)


    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        return super().on_train_epoch_start(trainer, pl_module)


    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        # scale = 1.0 if self.q_loss > 1e-3 else 0.985
        # scale = 1.0 if self.q_loss > 1e-3 else 0.95
        if self.noise_ratio > 0:
            summand = 0 if self.q_loss > 1e-3  else 0.01

            # self.noise_ratio.data.mul_(scale)
            self.noise_ratio.data.sub_(summand)
            # pl_module._noise_ratio.data.mul_(scale)
            
            # pl_module.noise_ratio(pl_module._noise_ratio)
            pl_module.noise_ratio(self.noise_ratio)
        pl_module.log("RNoise ratio", pl_module._noise_ratio, prog_bar=True)

        self.q_loss = 0
        return super().on_train_epoch_end(trainer, pl_module)
