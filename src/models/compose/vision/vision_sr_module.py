from collections import defaultdict
from typing import Any, Dict, Tuple
from src.data.compose.vision.sr.transforms.transforms import to_luminance

import lightning.pytorch as pl
import torchmetrics
import piq

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

        self.metrics = {
            "PSNR": piq.psnr,
            "SSIM": piq.ssim,
        }
        self._stage_metrics: Dict[str, Dict[str, Dict[str, torchmetrics.Metric]]] = defaultdict(dict)
        self._stage_psnr_sums: Dict[str, Dict[str, float]] = {
            "val": defaultdict(float),
            "test": defaultdict(float),
        }
        self._stage_psnr_counts: Dict[str, Dict[str, float]] = {
            "val": defaultdict(float),
            "test": defaultdict(float),
        }

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
                name: metric for name, metric in self.metrics.items()
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
            # metric.to(batch[0].device)
            metric_value = metric(outputs, target)
            self.log(
                f"{metric_name}/{dataset_name}",
                metric_value,
                prog_bar=False,
                on_step=False,
                on_epoch=True,
                batch_size=batch_size,
                add_dataloader_idx=False,
                sync_dist=True
                )
            self.trainer.logged_metrics[f"{metric_name}/{dataset_name}"] = metric_value
            if metric_name == "PSNR" and dataset_name and stage in self._stage_psnr_sums:
                psnr_scalar = metric_value.detach()
                if psnr_scalar.numel() > 1:
                    psnr_scalar = psnr_scalar.mean()
                psnr_scalar = psnr_scalar.item()
                self._stage_psnr_sums[stage][dataset_name] += psnr_scalar * batch_size
                self._stage_psnr_counts[stage][dataset_name] += batch_size
        self.log(
            f"Val_loss/{dataset_name}",
            loss,
            prog_bar=False,
            on_step=True,
            on_epoch=True,
            batch_size=batch_size,
            add_dataloader_idx=False,
            sync_dist=True
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
            sync_dist=True
        )
        return loss

    def validation_step(self, val_batch, val_index, dataloader_idx=0):
        outputs, target, loss, dataset_name = self._shared_step(val_batch)
        outputs = outputs.clamp(0,1)  # clamping pixel values ..
        # outputs = outputs.mul(255).round().div(255)
        if self.to_luminance:
            outputs = to_luminance(outputs)
            target = to_luminance(target)
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

    def test_step(self, test_batch, test_index, dataloader_idx=0):
        outputs, target, loss, dataset_name = self._shared_step(test_batch)
        outputs = outputs.clamp(0,1) 
        if self.to_luminance:
            outputs = to_luminance(outputs)
            target = to_luminance(target)
        self._log_dataset_metrics(
            stage="test",
            batch=test_batch,
            outputs=outputs,
            target=target,
            loss=loss,
            dataset_name=dataset_name,
            loader_idx=dataloader_idx,
        )
        return loss

    def predict_step(self, pred_batch, batch_idx, dataloader_idx=0):
        inputs = pred_batch[0] if isinstance(pred_batch, (tuple, list)) else pred_batch
        if self.denormalize:
            # return self.forward(inputs * 255).div(255).clamp(0,1)
            return self.forward(inputs * 255).div(255)
        else:
            return self.forward(inputs).clamp(0,1)

    def on_validation_epoch_start(self) -> None:
        self._reset_stage_psnr_tracking("val")

    def on_validation_epoch_end(self) -> None:
        self._log_weighted_psnr(stage="val", log_name="PSNR/Weighted_mean", prog_bar=False)

    def on_test_epoch_start(self) -> None:
        self._reset_stage_psnr_tracking("test")

    def on_test_epoch_end(self) -> None:
        self._log_weighted_psnr(stage="test", log_name="PSNR/Weighted_mean_test", prog_bar=False)

    def _reset_stage_psnr_tracking(self, stage: str) -> None:
        if stage in self._stage_psnr_sums:
            self._stage_psnr_sums[stage].clear()
            self._stage_psnr_counts[stage].clear()

    def _log_weighted_psnr(self, stage: str, log_name: str, prog_bar: bool) -> None:
        if stage not in self._stage_psnr_sums:
            return
        sums = self._stage_psnr_sums[stage]
        counts = self._stage_psnr_counts[stage]
        weighted_sum = 0.0
        weight_total = 0.0
        for dataset_name, total_value in sums.items():
            count = counts.get(dataset_name, 0.0)
            if not dataset_name or count == 0:
                continue
            dataset_avg = total_value / count
            dataset_weight = 1.0 / count
            weighted_sum += dataset_avg * dataset_weight
            weight_total += dataset_weight
        if weight_total > 0:
            weighted_psnr = weighted_sum / weight_total
            self.log(
                log_name,
                weighted_psnr,
                prog_bar=prog_bar,
                on_step=False,
                on_epoch=True,
                sync_dist=True
            )
            self.trainer.logged_metrics[log_name] = weighted_psnr
