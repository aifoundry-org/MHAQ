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
    
    def on_validation_end(self):
        # print(np.mean(self._map))
        return super().on_validation_end()

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
            self.log("loss_box", loss_box, prog_bar=False)
            self.log("loss_cls", loss_cls, prog_bar=False)
            self.log("loss_dfl", loss_dfl, prog_bar=False)
            loss = (
                loss_box + loss_cls + loss_dfl
            )  # losses are scaled inside criterion using params
        else:
            loss = self.criterion(output, target)
        self.log("loss", loss, prog_bar=True)
        return loss

    def validation_step(self, val_batch, val_index):
        inputs, target = val_batch
        output = self.forward(inputs)
        if self.model._get_name() in YOLO_FAMILY:
            output = self.forward(inputs)
        else:
            output = self.forward(inputs)
            val_loss = self.criterion(output, target)
        
        # iou_v = torch.linspace(start=0.5, end=0.95, steps=10)
        # n_iou = iou_v.numel()

        # outputs = transform_coco_outputs(output)
        # targets = transform_coco_targets(target)
        # metrics = []
        # for i,o in enumerate(outputs):
        #     m = torch.zeros(o.size(0), n_iou, dtype=torch.bool, device='cuda')
        #     mask = targets['idx']==i
        #     c = targets['cls'][mask]
        #     b = targets['box'][mask]
        #     if o.size(0)==0:
        #         if c.numel(): metrics.append((m, *[torch.zeros((2,0))], c.squeeze(-1)))
        #         continue
        #     if c.numel():
        #         m = compute_metric(o[:,:6], torch.cat((c, wh2xy(b)*1),1), iou_v)
        #     metrics.append((m.cpu(), o[:,4].cpu(), o[:,5].cpu(), c.squeeze(-1).cpu()))



        # metrics = [torch.cat(x, dim=0).cpu().numpy() for x in zip(*metrics)]
        map_value = self.mAP(output, target)
        # tp, fp, m_pre, m_rec, map50, mean_ap = compute_ap(*metrics)
        # if len(metrics) and metrics[0].any():
        #     self._map.append(mean_ap)
        self.log(f"mAP", map_value["map"], prog_bar=True, on_epoch=True, batch_size=5)
        self.log(f"mAP_50", map_value["map_50"], prog_bar=False, on_epoch=True, batch_size=5)


    def test_step(self, test_batch, test_index):
        raise NotImplementedError("Test is not yet implemented for OD networks!")
        inputs, target = test_batch
        outputs = self.forward(inputs)
        if self.model._get_name() in YOLO_FAMILY:
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
