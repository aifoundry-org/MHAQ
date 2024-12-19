import os
import sys
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from src.config.config_loader import load_and_validate_config
from src.data.compose.composer import DatasetComposer
from src.models.compose.composer import ModelComposer
from src.quantization.quantizer import Quantizer
from src.training.trainer import Trainer

torch.set_float32_matmul_precision('high')

# config = load_and_validate_config("config/rniq_config_resnet20_cifar100.yaml")
# config = load_and_validate_config("config/rniq_config_resnet20_old.yaml")
config = load_and_validate_config("config/rniq_config_resnet20_new.yaml")
dataset_composer = DatasetComposer(config=config)
model_composer = ModelComposer(config=config)
quantizer = Quantizer(config=config)()
trainer = Trainer(config=config)

data = dataset_composer.compose()
model = model_composer.compose()
qmodel = quantizer.quantize(model, in_place=True)

# Test model before quantization
trainer.test(model, datamodule=data)

# Finetune model
trainer.fit(qmodel, datamodule=data)

# Test model after quantization
trainer.test(qmodel, datamodule=data)
