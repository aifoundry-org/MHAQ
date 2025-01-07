import lightning.pytorch as pl
import torch
import logging
import numpy as np

from copy import deepcopy
from lightning.pytorch.callbacks.callback import Callback
from lightning.pytorch import Trainer, LightningModule

logger = logging.getLogger("lightning.pytorch")


class ReduceLrOnOutlier(Callback):
    def __init__(self, reduce_scale=2, lr_lim=0.005) -> None:
        self.LR_scale = reduce_scale
        self.epoch_mean_loss = []
        self.batch_loss = 0
        self.q_loss = 0
        self.lr_lim = lr_lim
        self.model_state = None
        self.optimizer_state = None
        self.noise_ratio = 1.0
        super().__init__()

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.num_epoch = trainer.max_epochs
        return super().on_fit_start(trainer, pl_module)

    def on_train_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs, batch, batch_idx: int) -> None:
        self.batch_loss += pl_module.wrapped_criterion.base_loss.item()
        self.q_loss += pl_module.wrapped_criterion.wloss.mean().item() + \
            pl_module.wrapped_criterion.aloss.mean().item()
        return super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        return super().on_train_epoch_start(trainer, pl_module)

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        w = 10
        revert = False
        save = False
        if trainer.current_epoch >= 2 * w:
            lp = np.convolve(self.epoch_mean_loss, np.ones(w) / w, 'valid')
            std = np.std(self.epoch_mean_loss[-lp.size:] - lp)
            if torch.isnan(torch.as_tensor(self.batch_loss)) or self.batch_loss - lp[-1] > 3 * std:
                logger.warning(
                    f"Loss is too high {self.batch_loss - lp[-1]} > {3 * std}! Reverting epoch, reducing LR by {self.LR_scale}")
                revert = True
            elif self.batch_loss - lp[-1] <= std:
                save = True
        elif trainer.current_epoch == 0:
            save = True

        if save:
            self.model_state = deepcopy(trainer.model.state_dict())
            self.optimizer_state = deepcopy(trainer.optimizers[0].state_dict())
            self.noise_ratio = pl_module._noise_ratio.detach().clone()

        if revert:
            trainer.model.load_state_dict(self.model_state)
            trainer.optimizers[0].load_state_dict(self.optimizer_state)
            self.change_lr(pl_module, trainer, pl_module.lr / self.LR_scale)
            pl_module.noise_ratio(self.noise_ratio)
        else:
            # exponent growth
            # self.epoch_mean_loss.append(self.batch_loss)
            # scale = 1.05 if self.q_loss > 1e-3 else 0.995
            # self.change_lr(pl_module, trainer, pl_module.lr * scale)
            
            # Moving parabolic growth
            self.epoch_mean_loss.append(self.batch_loss)
            eta = self.q_loss * 1e-4
            hscale = 1 + (eta * (self.lr_lim - pl_module.lr) / pl_module.lr)
            scale = hscale if self.q_loss > 1e-3 or pl_module._noise_ratio > 1e-3 else 0.98
            self.change_lr(pl_module, trainer, pl_module.lr * scale)

        self.batch_loss = 0
        self.q_loss = 0
        return super().on_train_epoch_end(trainer, pl_module)

    def change_lr(self, pl_module: LightningModule, trainer: Trainer, new_lr: float) -> None:
        optimizer = trainer.optimizers[0]
        for param_group in trainer.optimizers[0].param_groups:
            param_group['lr'] = new_lr

        pl_module.lr = new_lr
        trainer.optimizers[0] = optimizer
