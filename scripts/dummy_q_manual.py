import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from torch import nn, optim
from torchvision.models import resnet18
from lightning.pytorch.callbacks import ModelCheckpoint

from src.aux.types import MType
from src.training.trainer import Trainer
from src.quantization.dummy.dummy_quant import DummyQuant
from src.models.compose.composer import ModelComposer
from src.data.compose.vision import CIFAR10


# Model composer section
composer = ModelComposer()
composer.model = resnet18(num_classes=10)
composer.criterion = nn.CrossEntropyLoss()
composer.optimizer = optim.Adam
composer.model_type = MType.VISION_CLS
composer.lr = 0.001

# Model quantizer section
quantizer = DummyQuant()
quantizer.weight_bit = 0
quantizer.act_bit = 0

# Model trainer section
callbacks = [ModelCheckpoint(filename="dummy_checkpoint_rsnt18")]
trainer = Trainer(max_epochs=20, 
                  val_check_interval=10, 
                  callbacks=callbacks)

# Dataset section
data = CIFAR10()
data.batch_size_train = 2000
data.num_workers = 10

# Composing section
model = composer.compose()
qmodel = quantizer.quantize(model)

# Training section
trainer.fit(qmodel, data)
