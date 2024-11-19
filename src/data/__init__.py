from .vision_cls.mnist import MNISTDataModule
from .vision_cls.cifar_dali import CIFAR10DALIDataModule
from .vision_cls.cifar import CIFAR10DataModule
from .vision_cls.imagenet_dali import ImageNetDALIDataModule
from .vision_cls.imagenet import ImageNetDataModule
__all__ = ["MNISTDataModule", 
           "CIFAR10DataModule", 
           "CIFAR10DALIDataModule", 
           "ImageNetDALIDataModule", 
           "ImageNetDataModule"]