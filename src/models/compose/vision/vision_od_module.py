import torch
import logging
import torchmetrics
import lightning.pytorch as pl
from typing import Any, Dict
from src.models.od import YOLO_FAMILY
from src.models.od.loss.yolo_loss import ComputeYoloLoss
from src.models.od.utils.yolo_nms import non_max_suppression, wh2xy
from src.models.od.utils.yolo_decode import decode_yolo_nms, compute_metric, compute_ap
from src.models.od.metrics.map_metrics import MeanAveragePrecisionYolo
from src.data.compose.vision.od.coco import (
    transform_coco_outputs,
    transform_coco_targets,
)

import torchmetrics.detection

import numpy as np


logger = logging.getLogger("lightning.pytorch")


class LVisionOD(pl.LightningModule):
    def __init__(self, setup: Dict, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = setup["model"]
        if self.model._get_name() in YOLO_FAMILY:
            params = {"box": 7.5, "cls": 0.5, "dfl": 1.5}  # Probably move to the config
            self.criterion = ComputeYoloLoss(
                self.model, params
            )  # TODO carefull when passing quantizaed model!!
            self.mAP = MeanAveragePrecisionYolo(box_format="xyxy")
        else:
            self.criterion = setup["criterion"]
            self.mAP = torchmetrics.detection.MeanAveragePrecision(box_format="xyxy")

        self.optimizer = setup["optimizer"]
        self.metrics = []

        self.lr = setup["lr"]
        self._map = []
        self._metrics = []

        self._init_metrics()
    
    def on_train_start(self):
        self._batch_size = self.trainer.config.data.batch_size
        return super().on_train_start()

    def on_validation_start(self):
        self._batch_size = self.trainer.config.data.batch_size
        return super().on_validation_start()

    def on_validation_epoch_end(self):
        map = self.mAP.compute()
        self.log(
            f"mAP",
            map["map"],
            prog_bar=False,
            on_epoch=True,
            batch_size=self._batch_size,
        )
        self.log(
            f"mAP_50",
            map["map_50"],
            prog_bar=False,
            on_epoch=True,
            batch_size=self._batch_size,
        )
        self.mAP.reset()
        return super().on_validation_epoch_end()
    
    def on_test_epoch_end(self):
        map = self.mAP.compute()
        self.log(
            f"mAP",
            map["map"],
            prog_bar=False,
            on_epoch=True,
            batch_size=self._batch_size,
        )
        self.log(
            f"mAP_50",
            map["map_50"],
            prog_bar=False,
            on_epoch=True,
            batch_size=self._batch_size,
        )
        self.mAP.reset()
        return super().on_test_epoch_end()

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        state_dict = checkpoint["state_dict"]
        model_state_dict = self.state_dict()
        is_changed = False
        for k in state_dict:
            if k in model_state_dict:
                if state_dict[k].shape != model_state_dict[k].shape:
                    logger.info(
                        f"Skip loading parameter: {k}, "
                        f"required shape: {model_state_dict[k].shape}, "
                        f"loaded shape: {state_dict[k].shape}"
                    )
                    state_dict[k] = model_state_dict[k]
                    is_changed = True
            else:
                logger.info(f"Dropping parameter {k}")
                is_changed = True

        if is_changed:
            checkpoint.pop("optimizer_states", None)

    def _init_metrics(self):
        self.metrics.append(["mAP", self.mAP])

    def configure_optimizers(self):
        return self.optimizer(self.parameters(), self.lr)

    def forward(self, inputs):
        return self.model(inputs)

    def training_step(self, batch, batch_idx):
        inputs, target = batch
        output = self.forward(inputs)
        if self.model._get_name() in YOLO_FAMILY:
            target = {k: torch.cat([d[k] for d in target], dim=0) for k in target[0]}
            loss_box, loss_cls, loss_dfl = self.criterion(output, target)
            self.log("loss_box", loss_box, prog_bar=False, batch_size=self._batch_size)
            self.log("loss_cls", loss_cls, prog_bar=False, batch_size=self._batch_size)
            self.log("loss_dfl", loss_dfl, prog_bar=False, batch_size=self._batch_size)
            loss = (
                loss_box + loss_cls + loss_dfl
            )  # losses are scaled inside criterion using params
        else:
            loss = self.criterion(output, target)
        self.log("loss", loss, prog_bar=True, batch_size=self._batch_size)
        return loss

    def validation_step(self, val_batch, val_index):
        inputs, target = val_batch
        output = self.forward(inputs)
        if self.model._get_name() in YOLO_FAMILY:
            output = self.forward(inputs)
        else:
            output = self.forward(inputs)
            val_loss = self.criterion(output, target)

        self.mAP.update(output, target)

    def test_step(self, test_batch, test_index):
        inputs, target = test_batch
        output = self.forward(inputs)
        if self.model._get_name() in YOLO_FAMILY:
            output = self.forward(inputs)
        else:
            output = self.forward(inputs)
            test_loss = self.criterion(output, target)

        self.mAP.update(output, target)

    def predict_step(self, *args, **kwargs):
        raise NotImplementedError("Predict is not yet implemented for OD networks!")
