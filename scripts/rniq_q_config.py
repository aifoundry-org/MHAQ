import os
import sys
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from src.config.config_loader import load_and_validate_config
from src.data import CIFAR10DALIDataModule
from src.data import CIFAR10DataModule
from src.data import ImageNetDALIDataModule
from src.data import ImageNetDataModule
from src.models.compose.composer import ModelComposer
from src.quantization.quantizer import Quantizer
from src.training.trainer import Trainer

torch.set_float32_matmul_precision('high')

config = load_and_validate_config("config/rniq_config_resnet18.yaml")
composer = ModelComposer(config=config)
quantizer = Quantizer(config=config)()
trainer = Trainer(config=config)

# data = CIFAR10DALIDataModule()
data = ImageNetDataModule(data_dir="./data/imagenet", batch_size=1024)
# data = CIFAR10DataModule()
# data.batch_size = config.data.batch_size
# data.num_workers = config.data.num_workers

model = composer.compose()
qmodel = quantizer.quantize(model, in_place=True)

# Test model before quantization
trainer.test(model, datamodule=data)

# Finetune model
trainer.fit(qmodel, datamodule=data)

# Test model after quantization
trainer.test(qmodel, datamodule=data)
