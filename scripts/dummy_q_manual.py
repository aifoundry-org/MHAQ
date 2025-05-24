import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from torch import nn, optim
from torchvision.models import resnet18
from lightning.pytorch.callbacks import ModelCheckpoint

from src.models.od import yolo_v11_n
from src.models.od.loss.yolo_loss import ComputeYoloLoss
from src.aux.types import MType
from src.training.trainer import Trainer
from src.quantization.dummy.dummy_quant import DummyQuant
from src.models.compose.composer import ModelComposer
from src.data.compose.vision import CIFAR10
from src.data.compose.vision.od import VOC2012, COCO


# Model composer section
composer = ModelComposer()
params = {"box": 7.5, "cls": 0.5, "dfl": 1.5}
# composer.model = resnet18(num_classes=10)
# composer.model = yolo_v11_n(20).cuda()
composer.model = yolo_v11_n(80).cuda()
# composer.model = yolo_v11_n(80)
composer.criterion = ComputeYoloLoss(composer.model, params)
composer.optimizer = optim.Adam
composer.model_type = MType.VISION_OD
composer.lr = 0.001

# Model quantizer section
# quantizer = DummyQuant()
# quantizer.weight_bit = 0
# quantizer.act_bit = 0

# Model trainer section
callbacks = [ModelCheckpoint(filename="dummy_checkpoint_rsnt18")]
trainer = Trainer(max_epochs=20,
                  check_val_every_n_epoch=10, 
                #   val_check_interval=10, 
                  callbacks=callbacks)

# Dataset section
# data = CIFAR10()

# data = VOC2012()
data = COCO()

data.batch_size = 50
data.num_workers = 5

data.prepare_data()

data.setup("train")

# pass
# Composing section
model = composer.compose()
batch = next(iter(data.train_dataloader()))
# qmodel = quantizer.quantize(model)

# Training section
# trainer.fit(qmodel, data)
trainer.fit(model, data)
