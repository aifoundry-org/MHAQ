from .cls.mnist import MNISTDataModule as MNIST
from .cls.cifar10 import CIFAR10DataModule as CIFAR10
from .cls.cifar100 import CIFAR100DataModule as CIFAR100
from .cls.imagenet import ImageNetDataModule as IMAGENET
from .cls.cifar10_dali import CIFAR10DALIDataModule as CIFAR10_DALI
from .cls.cifar100_dali import CIFAR10DALIDataModule as CIFAR100_DALI
from .cls.cifar100_gen import CIFAR100GenDataModule as CIFAR100_GEN
from .cls.imagenet_dali import ImageNetDALIDataModule as IMAGENET_DALI

from .od.voc_yolo import YOLOVOCDataModule2012 as VOC2012_YOLO
from .od.coco import COCODataModule as COCO

__all__ = [
    "MNIST",
    "CIFAR10",
    "CIFAR10_DALI",
    "CIFAR100",
    "CIFAR100_DALI",
    "CIFAR100_GEN",
    "IMAGENET",
    "IMAGENET_DALI",
    "VOC2012_YOLO",
    "COCO"
]
