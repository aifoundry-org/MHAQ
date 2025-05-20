import torch
import logging
import torchmetrics
import lightning.pytorch as pl
from typing import Any, Dict
from src.models.od import YOLO_FAMILY
from src.models.od.loss.yolo_loss import ComputeYoloLoss

import torchmetrics.detection


logger = logging.getLogger("lightning.pytorch")

class LVisionOD(pl.LightningModule):
    def __init__(self, setup: Dict, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = setup["model"]
        if self.model.name in YOLO_FAMILY:
            params = {"box": 7.5, "cls": 0.5, "dfl": 1.5} # Probably move to the config
            self.criterion = ComputeYoloLoss(self.model, params) # TODO carefull when passing quantizaed model!!
            self.mAP = torchmetrics.detection.MeanAveragePrecision(box_format="xywh")
        else:
            self.criterion = setup["criterion"]
            self.mAP = torchmetrics.detection.MeanAveragePrecision(box_format="xyxy")

        self.optimizer = setup["optimizer"]
        self.metrics = []

        self.lr = setup["lr"]

        self._init_metrics()
    
    def on_load_checkpoint(self, checkpoint: dict) -> None:
        state_dict = checkpoint["state_dict"]
        model_state_dict = self.state_dict()
        is_changed = False
        for k in state_dict:
            if k in model_state_dict:
                if state_dict[k].shape != model_state_dict[k].shape:
                    logger.info(f"Skip loading parameter: {k}, "
                                f"required shape: {model_state_dict[k].shape}, "
                                f"loaded shape: {state_dict[k].shape}")
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
        if self.model in YOLO_FAMILY:
            loss_box, loss_cls, loss_dfl = self.criterion(output, target)
            self.log("loss_box", loss_box, prog_bar=False)
            self.log("loss_cls", loss_cls, prog_bar=False)
            self.log("loss_dfl", loss_dfl, prog_bar=False)
            loss = loss_box + loss_cls + loss_dfl # losses are scaled inside criterion using params
        else:
            loss = self.criterion(output, target)
        self.log("loss", loss, prog_bar=True)
        return loss

    def validation_step(self, val_batch, val_index):
        inputs, target = val_batch
        output = self.forward(inputs)
        if self.model in YOLO_FAMILY:
            loss_box, loss_cls, loss_dfl = self.criterion(output, target)
            self.log("val_loss_box", loss_box, prog_bar=False)
            self.log("val_loss_cls", loss_cls, prog_bar=False)
            self.log("val_loss_dfl", loss_dfl, prog_bar=False)
            val_loss = loss_box + loss_cls + loss_dfl
        else:
            val_loss = self.criterion(output, target)
        for name, metric in self.metrics:
            metric_value = metric(output, target)
            self.log(f"{name}", metric_value, prog_bar=False)

        self.log("val_loss", val_loss, prog_bar=False)
    
    def test_step(self, test_batch, test_index):
        inputs, target = test_batch
        outputs = self.forward(inputs)
        if self.model in YOLO_FAMILY:
            loss_box, loss_cls, loss_dfl = self.criterion(outputs, target)
            test_loss = loss_box + loss_cls + loss_dfl
        else:
            test_loss = self.criterion(outputs, target)
        for name, metric in self.metrics:
            metric_value = metric(outputs, target)
            self.log(f"{name}", metric_value, prog_bar=False)

        self.log("test_loss", test_loss, prog_bar=True)
    
    def predict_step(self, *args, **kwargs):
        raise NotImplementedError("Predict is not yet implemented for OD networks!")