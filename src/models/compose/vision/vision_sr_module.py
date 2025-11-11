import lightning.pytorch as pl
import torchmetrics
from typing import Any, Dict, Tuple


class LVisionSR(pl.LightningModule):
    def __init__(self, setup: Dict, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.model = setup["model"]
        self.criterion = setup["criterion"]
        self.optimizer = setup["optimizer"]
        self.lr = setup["lr"]
        self.metrics = []

        config = setup.get("config")
        if config and getattr(config, "model", None):
            model_params = getattr(config.model, "params", {}) or {}
        else:
            model_params = {}
        self._data_range = model_params.get("data_range", 1.0)
        self.psnr = torchmetrics.PeakSignalNoiseRatio(
            data_range=self._data_range
        )
        self.ssim = torchmetrics.StructuralSimilarityIndexMeasure(
            data_range=self._data_range
        )

        self._init_metrics()

    def _init_metrics(self) -> None:
        self.metrics.append(("PSNR", self.psnr))
        self.metrics.append(("SSIM", self.ssim))

    def configure_optimizers(self):
        return self.optimizer(self.parameters(), self.lr)

    def forward(self, inputs):
        return self.model(inputs)

    def _shared_step(self, batch: Tuple):
        inputs, target = batch
        outputs = self.forward(inputs)
        loss = self.criterion(outputs, target)
        return outputs, target, loss

    def training_step(self, batch, batch_idx):
        _, _, loss = self._shared_step(batch)
        self.log(
            "loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=False,
            batch_size=batch[0].size(0),
        )
        return loss

    def validation_step(self, val_batch, val_index):
        outputs, target, loss = self._shared_step(val_batch)
        for name, metric in self.metrics:
            metric_value = metric(outputs, target)
            self.log(
                name,
                metric_value,
                prog_bar=False,
                on_step=False,
                on_epoch=True,
                batch_size=val_batch[0].size(0),
            )
        self.log(
            "val_loss",
            loss,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            batch_size=val_batch[0].size(0),
        )
        return loss

    def test_step(self, test_batch, test_index):
        outputs, target, loss = self._shared_step(test_batch)
        for name, metric in self.metrics:
            metric_value = metric(outputs, target)
            self.log(
                f"test_{name}",
                metric_value,
                prog_bar=False,
                on_step=False,
                on_epoch=True,
                batch_size=test_batch[0].size(0),
            )
        self.log(
            "test_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=test_batch[0].size(0),
        )
        return loss

    def predict_step(self, pred_batch, batch_idx, dataloader_idx=0):
        inputs = pred_batch[0] if isinstance(pred_batch, (tuple, list)) else pred_batch
        return self.forward(inputs)
