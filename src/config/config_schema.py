from pydantic import BaseModel, field_validator
from typing import Literal, Dict, Optional, List
import torch.nn as nn
import torch.optim as optim


class ModelConfig(BaseModel):
    type: Literal['VISION_CLS', 'VISION_DNS', 'VISION_SR', 'LM']
    name: str
    params: Dict


class TrainingConfig(BaseModel):
    criterion: str
    optimizer: str
    learning_rate: float
    epochs: int


class QuantizationConfig(BaseModel):
    name: str
    act_width: int
    weight_width: int
    params: Optional[Dict] = None
    excluded_layers: Optional[List[str]] = None


class DataConfig(BaseModel):
    dataset_name: str
    batch_size: int
    num_workers: int
    augmentations: Optional[List[str]] = None


class ConfigSchema(BaseModel):
    model: ModelConfig
    data: DataConfig
    training: TrainingConfig
    quantization: QuantizationConfig

    # TODO further validation
    @field_validator('training')
    def validate_training(cls, v):
        # Check if criterion is a valid loss function
        if not hasattr(nn, v['criterion']):
            raise ValueError(f"Invalid criterion: {v['criterion']}")
        # Check if optimizer is a valid optimizer
        if not hasattr(optim, v['optimizer']):
            raise ValueError(f"Invalid optimizer: {v['optimizer']}")
        return v
