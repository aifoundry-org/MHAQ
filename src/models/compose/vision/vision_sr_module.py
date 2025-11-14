from collections import defaultdict
from typing import Any, Dict, Tuple
from src.data.compose.vision.sr.transforms.transforms import to_luminance

import lightning.pytorch as pl
import torchmetrics

class LVisionSR(pl.LightningModule):
    def __init__(self, setup: Dict, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.model = setup["model"]
        self.criterion = setup["criterion"]
        self.optimizer = setup["optimizer"]
        self.lr = setup["lr"]

        config = setup.get("config")
        self.denormalize = config.data.params.get("denormalize", False)
        self.to_luminance = config.data.params.get("to_luminance", False)

        if config and getattr(config, "model", None):
            model_params = getattr(config.model, "params", {}) or {}
        else:
            model_params = {}
        self._data_range = model_params.get("data_range", 1.0)

        self._metrics = {
            "PSNR": torchmetrics.PeakSignalNoiseRatio(
                data_range=self._data_range
            ),
            "SSIM": torchmetrics.StructuralSimilarityIndexMeasure(
                data_range=self._data_range
            ),
        }
        self._stage_metrics: Dict[str, Dict[str, Dict[str, torchmetrics.Metric]]] = defaultdict(dict)

    def configure_optimizers(self):
        return self.optimizer(self.parameters(), self.lr)

    def forward(self, inputs):
        return self.model(inputs)

    def _shared_step(self, batch: Tuple):
        dataset_name = None
        if isinstance(batch, (list, tuple)):
            if len(batch) == 3:
                inputs, target, meta = batch
                dataset_name = self._extract_dataset_name(meta)
            else:
                inputs, target = batch
        else:
            inputs = batch
            target = None
        if self.denormalize:
            outputs = self.forward(inputs * 255).div(255)
        else:
            outputs = self.forward(inputs)
        loss = self.criterion(outputs, target)
        return outputs, target, loss, dataset_name

    @staticmethod
    def _extract_dataset_name(meta):
        if meta is None:
            return None
        if isinstance(meta, str):
            return meta
        if isinstance(meta, (list, tuple)):
            if not meta:
                return None
            first = meta[0]
            if isinstance(first, str):
                return first
        return str(meta)

    def _get_metrics(self, stage: str, dataset_name: str):
        stage_metrics = self._stage_metrics[stage]
        if dataset_name not in stage_metrics:
            stage_metrics[dataset_name] = {
                name: metric for name, metric in self._metrics.items()
            }
        return stage_metrics[dataset_name]

    def _log_dataset_metrics(
        self,
        stage: str,
        batch,
        outputs,
        target,
        loss,
        dataset_name: str | None,
        loader_idx: int,
    ):
        batch_size = batch[0].size(0) if isinstance(batch, (tuple, list)) else batch.size(0)
        # dataset_key = dataset_name or f"loader_{loader_idx}"
        metrics = self._get_metrics(stage, "")
        for metric_name, metric in metrics.items():
            metric.to(batch[0].device)
            metric_value = metric(outputs, target)
            self.log(
                f"{metric_name}/{dataset_name}",
                metric_value,
                prog_bar=False,
                on_step=False,
                on_epoch=True,
                batch_size=batch_size,
                add_dataloader_idx=False
            )
        self.log(
            f"Val_loss/{dataset_name}",
            loss,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
            add_dataloader_idx=False
        )

    def training_step(self, batch, batch_idx):
        _, _, loss, _ = self._shared_step(batch)
        self.log(
            "Loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=False,
            batch_size=batch[0].size(0),
        )
        return loss

    def validation_step(self, val_batch, val_index, dataloader_idx=0):
        outputs, target, loss, dataset_name = self._shared_step(val_batch)
        if self.to_luminance:
            outputs = to_luminance(outputs)
            target = to_luminance(target)
        outputs = outputs.clamp(0,1)  # rounding pixel values ..
        self._log_dataset_metrics(
            stage="val",
            batch=val_batch,
            outputs=outputs,
            target=target,
            loss=loss,
            dataset_name=dataset_name,
            loader_idx=dataloader_idx,
        )
        return loss

    def test_step(self, test_batch, test_index):
        outputs, target, loss, dataset_name = self._shared_step(test_batch)
        self._log_dataset_metrics(
            stage="test",
            batch=test_batch,
            outputs=outputs,
            target=target,
            loss=loss,
            dataset_name=dataset_name,
            loader_idx=test_index,
        )
        return loss

    def predict_step(self, pred_batch, batch_idx, dataloader_idx=0):
        inputs = pred_batch[0] if isinstance(pred_batch, (tuple, list)) else pred_batch
        return self.forward(inputs)
