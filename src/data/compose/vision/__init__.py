from .mnist import MNISTDataModule as MNIST
from .cifar10 import CIFAR10DataModule as CIFAR10
from .cifar10_dali import CIFAR10DALIDataModule as CIFAR10_DALI
from .cifar100 import CIFAR100DataModule as CIFAR100

__all__ = ["MNIST", "CIFAR10", "CIFAR10_DALI", "CIFAR100"]