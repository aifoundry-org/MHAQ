import os
import lightning.pytorch as pl
import torch
import src.models as compose_models

from collections import OrderedDict
from torch import nn, optim
from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer
from pathlib import Path

from src.aux.types import MType
from src.aux.find_root import find_project_root
from src.models.compose.vision.vision_cls_module import LVisionCls
from src.models.compose.vision.vision_od_module import LVisionOD
from src.models.compose.vision.vision_sr_module import LVisionSR
from src.models.compose.criterion import get_criterion

current_file_path = Path(__file__).resolve()


class ModelComposer():
    def __init__(self, config=None) -> None:
        self.config = config
        self.model_type: MType
        self.model: nn.Module
        self.criterion: _Loss
        self.optimizer: Optimizer
        self.lr: float = 1e-4

    def compose(self) -> pl.LightningModule:
        if self.config:
            model_config = self.config.model
            training_config = self.config.training

            self.model_type = MType[model_config.type]
            self.model = getattr(compose_models, model_config.name)(
                **model_config.params)
            self.criterion = get_criterion(criterion_name=training_config.criterion, model=self.model)
            self.optimizer = getattr(optim, training_config.optimizer)
            self.lr = training_config.learning_rate

            if model_config.cpt_url:
                if "file://" in model_config.cpt_url:
                    state_dict = torch.load(os.path.join(find_project_root(
                        current_file_path), model_config.cpt_url.split("file://")[1]), weights_only=False)
                else:
                    state_dict = torch.hub.load_state_dict_from_url(
                        model_config.cpt_url)
                state_dict = state_dict.get('model', state_dict)
                try:
                    self.model.load_state_dict(state_dict)
                except:
                    wrapper = nn.Sequential(
                        OrderedDict([('module', self.model)]))
                    wrapper.load_state_dict(state_dict)

        else:
            assert (self.model)
            assert (self.model_type)
            assert (self.criterion)
            assert (self.optimizer)

        if self.model_type == MType.VISION_CLS:
            module = LVisionCls(self.__dict__)
        elif self.model_type == MType.VISION_OD:
            return LVisionOD(self.__dict__)
        elif self.model_type == MType.VISION_DNS:
            raise NotImplementedError()
        elif self.model_type == MType.VISION_SR:
            module = LVisionSR(self.__dict__)
        elif self.model_type == MType.LM:
            raise NotImplementedError()
        else:
            raise NotImplementedError()

        return module
