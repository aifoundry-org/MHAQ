from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from typing_extensions import override


class NoiseEarlyStopping(EarlyStopping):
    def __init__(
        self,
        monitor,
        min_delta=0,
        patience=3,
        verbose=False,
        mode="min",
        strict=True,
        check_finite=True,
        stopping_threshold=None,
        divergence_threshold=None,
        check_on_train_epoch_end=None,
        log_rank_zero_only=False,
    ):
        super().__init__(
            monitor,
            min_delta,
            patience,
            verbose,
            mode,
            strict,
            check_finite,
            stopping_threshold,
            divergence_threshold,
            check_on_train_epoch_end,
            log_rank_zero_only,
        )
        self.tracking_metric = False

    @override
    def on_train_epoch_end(self, trainer, pl_module):
        if self.tracking_metric:
            return super().on_train_epoch_end(trainer, pl_module)

        if pl_module._noise_ratio <= 0 and not self.tracking_metric:
            self._log_info(
                trainer,
                f"Noise ratio is {pl_module._noise_ratio}, started tracking metric.",
                self.log_rank_zero_only
            )
            self.tracking_metric = True

    @override
    def on_validation_end(self, trainer, pl_module):
        if self.tracking_metric:
            return super().on_validation_end(trainer, pl_module)
