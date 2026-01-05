import torch.nn as nn
import torch.optim as optim
import src.models as compose_models
import src.callbacks as compose_callbacks
import src.quantization as compose_quantization
from src.aux.types import QScheme, QMethod
from src.quantization.rniq.config.config_schema import RNIQQuantizerParams

from pydantic import BaseModel, field_validator, model_validator
from typing import Any, Literal, Dict, Optional, List


class ModelConfig(BaseModel):
    type: Literal["VISION_CLS", "VISION_DNS", "VISION_SR", "VISION_OD", "LM"]
    name: str
    cpt_url: Optional[str] = None
    params: Dict

class Callback(BaseModel):
    params: Optional[Dict] = None

class Logger(BaseModel):
    params: Optional[Dict]


class TrainingConfig(BaseModel):
    criterion: str | List[str]
    optimizer: str
    learning_rate: float
    max_epochs: int
    val_every_n_epochs: Optional[int] = 1
    val_check_interval: Optional[float] = None
    log_every_n_steps: Optional[int] = None
    callbacks: Optional[Dict[str, Callback]] = []
    loggers: Optional[Dict[str, Logger]] = []


class CalibrationConfig(BaseModel):
    act_bit: int
    weight_bit: int

class QuantizationConfig(BaseModel):
    name: str
    act_bit: int
    weight_bit: int
    qmethod: QMethod = QMethod.RNIQ
    qscheme: Optional[QScheme] = QScheme.PER_TENSOR
    excluded_layers: Optional[List[str]] = None
    calibration: Optional[CalibrationConfig] = None
    freeze_batchnorm: Optional[bool] = False
    fuse_batchnorm: Optional[bool] = True
    quantize_bias: Optional[bool] = True
    activation_zero_point: Optional[float] = 0.0
    params: Optional[RNIQQuantizerParams | Dict[str, Any]] = None

    @model_validator(mode="after")
    def validate_params(self):
        if self.name == "RNIQQuant":
            if self.params is None:
                self.params = RNIQQuantizerParams()
            elif isinstance(self.params, dict):
                self.params = RNIQQuantizerParams.model_validate(self.params)
            elif not isinstance(self.params, RNIQQuantizerParams):
                raise TypeError(
                    "Params for RNIQQuant must be a mapping or RNIQQuantizerParams."
                )
        return self


class DataConfig(BaseModel):
    dataset_name: str
    batch_size: int
    num_workers: int
    augmentations: Optional[List[str]] = None
    data_dir: Optional[str] = "./data"


class ConfigSchema(BaseModel):
    model: ModelConfig
    data: DataConfig
    training: TrainingConfig
    quantization: QuantizationConfig

    @field_validator("training")
    def validate_training(cls, v):
        # if not hasattr(nn, v.criterion):
            # raise ValueError(f"Invalid criterion: {v.criterion}")
        if not hasattr(optim, v.optimizer):
            raise ValueError(f"Invalid optimizer: {v.optimizer}")
        for callback in v.callbacks:
            if not hasattr(compose_callbacks, callback):
                raise ValueError(f"Invalid callback: {callback}")
        return v

    @field_validator("model")
    def validate_model(cls, v):
        if not hasattr(compose_models, v.name):
            raise ValueError(
                f"Invalid model name: {v.name}.\nValid options are: {compose_models.__all__}."
            )
        return v
    
    @field_validator("quantization")
    def validate_quantizer(cls, v):
        if not hasattr(compose_quantization, v.name):
            raise ValueError(
                f"Invalid quantizer name: {v.name}.\nValid options are: {compose_quantization.__all__}."
            )
        return v
