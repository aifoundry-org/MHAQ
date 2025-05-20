from torchvision.models import resnet18
from .cls.resnet.resnet_cifar import resnet20_cifar10
from .cls.resnet.resnet_cifar import resnet20_cifar10_new
from .cls.resnet.resnet_cifar import resnet20_cifar100

__all__ = ["resnet18", "resnet20_cifar10", "resnet20_cifar100", "resnet20_cifar10_new"]