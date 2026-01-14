import lightning.pytorch as pl
import logging
import numpy as np

from lightning.pytorch.callbacks.callback import Callback
from lightning.pytorch import Trainer, LightningModule

logger = logging.getLogger("lightning.pytorch")

class RandNoiseScale(Callback):
    def __init__(self, reduce_scale=2, update_every=-1) -> None:
        self.q_loss = 0
        self.update_every = update_every
        super().__init__()


    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        return super().on_fit_start(trainer, pl_module)

    def on_train_start(self, trainer, pl_module):
        if self.update_every == -1:
            self.update_every = trainer.num_training_batches
        return super().on_train_start(trainer, pl_module)

    def on_train_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs, batch, batch_idx: int) -> None:
        self.q_loss += pl_module.wrapped_criterion.wloss.mean().item() + pl_module.wrapped_criterion.aloss.mean().item()
        if (batch_idx + 1) % self.update_every == 0:
            noise_ratio = pl_module._noise_ratio
            if noise_ratio >= 0 and self.q_loss <= 1e-3:

                noise_ratio.data.sub_(0.01)
                
                if noise_ratio < 0:
                    noise_ratio.data.zero_()

                pl_module.noise_ratio(noise_ratio)
                
            pl_module.log("RNoise ratio", pl_module._noise_ratio, prog_bar=True, sync_dist=True)

            self.q_loss = 0
            
        return super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)


    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        return super().on_train_epoch_start(trainer, pl_module)


    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        
        return super().on_train_epoch_end(trainer, pl_module)
