import torch
import torchmetrics
import lightning.pytorch as pl

from typing import Any, Dict


class LVisionCls(pl.LightningModule):
    def __init__(self, setup: Dict, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.model = setup["model"]
        self.criterion = setup["criterion"]
        self.optimizer = setup["optimizer"]
        self.metrics = []
        self.acc_metric = torchmetrics.Accuracy(
            task="multiclass",
            num_classes=setup["config"].model.params["num_classes"],
            top_k=1,
        )
        self.lr = setup["lr"]

        self._init_metrics()

    def _init_metrics(self):
        self.metrics.append(["Accuracy_top1", self.acc_metric])

    def configure_optimizers(self):
        return self.optimizer(self.parameters(), self.lr)

    def forward(self, inputs):
        return self.model(inputs)

    def training_step(self, batch, batch_idx):
        inputs, target = batch
        output = self.model(inputs)
        loss = self.criterion(output, target)
        self.log("loss", loss, prog_bar=True)
        return loss

    def validation_step(self, val_batch, val_index):
        inputs, target = val_batch
        outputs = self.forward(inputs)
        val_loss = self.criterion(outputs, target)
        for name, metric in self.metrics:
            metric_value = metric(outputs, target)
            self.log(f"{name}", metric_value, prog_bar=False)

        self.log("val_loss", val_loss, prog_bar=False)
    
    def test_step(self, test_batch, test_index):
        inputs, target = test_batch
        outputs = self.forward(inputs)
        val_loss = self.criterion(outputs, target)
        for name, metric in self.metrics:
            metric_value = metric(outputs, target)
            self.log(f"{name}", metric_value, prog_bar=False)

        self.log("test_loss", val_loss, prog_bar=True)


    def predict_step(self, pred_batch):
        inputs, target = pred_batch
        return self.forward(inputs)
