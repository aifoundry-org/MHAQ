import lightning.pytorch as pl
import logging
import torch

from lightning.pytorch.callbacks.callback import Callback
from lightning.pytorch import Trainer, LightningModule

logger = logging.getLogger("lightning.pytorch")

class TemperatureScale(Callback):
    def __init__(self, scale=1.e-3, scale_lr=0.999, tol=1e-2, warmup = 50) -> None:
        self.scale = scale
        self.scale_lr = scale_lr
        self.tol = tol
        self.warmup = warmup
        super().__init__()


    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        return super().on_fit_start(trainer, pl_module)

    def on_train_start(self, trainer, pl_module):
        self.lr = pl_module.lr
        self.total_batch = 0
        self.t = 0
        self.lr_t = 1.0
        self.change_lr(pl_module, trainer, 0)

        return super().on_train_start(trainer, pl_module)

    def on_train_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs, batch, batch_idx: int) -> None:

        self.total_batch += 1
                
        pl_module.log("temperature", self.t, prog_bar=True)

        self.t = (self.t + self.scale) if self.total_batch > self.warmup else self.t

        loss = pl_module.wrapped_criterion

        scale_lr = self.scale_lr if loss.aloss > 0 or loss.wloss > 0 else 0.995        
        self.lr_t = (self.lr_t * scale_lr) if self.total_batch > self.warmup else self.lr_t

        loss.t = torch.tensor(self.t ** 2)
        
        new_lr = self.lr * self.lr_t if self.total_batch > self.warmup else self.lr * self.total_batch / self.warmup
        self.change_lr(pl_module, trainer, new_lr)
            
        return super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)


    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        return super().on_train_epoch_start(trainer, pl_module)


    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        
        return super().on_train_epoch_end(trainer, pl_module)


    def change_lr(self, pl_module: LightningModule, trainer: Trainer, new_lr: float) -> None:
            optimizer = trainer.optimizers[0]
            for param_group in trainer.optimizers[0].param_groups:
                param_group['lr'] = new_lr

            pl_module.lr = new_lr
            trainer.optimizers[0] = optimizer
