import torch
from lightning import pytorch as pl

from torchvision.datasets import CIFAR10
from torchvision import transforms
from torchvision.transforms import AutoAugmentPolicy

from torch.utils.data import random_split, DataLoader


class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self,
                 data_dir: str = "./data",
                 batch_size=1000,
                 num_workers=5) -> None:

        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.transform_train = transforms.Compose(
            [
                # transforms.AutoAugment(policy=AutoAugmentPolicy.CIFAR10),
                # transforms.RandomHorizontalFlip(),
                # transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                self._normalize(),
            ]
        )

        self.transform_test = transforms.Compose(
            [transforms.ToTensor(), self._normalize()]
        )

    def prepare_data(self):
        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage: str):
        cifar_data = CIFAR10(
            self.data_dir, train=True, transform=self.transform_train
        )
        self.cifar_train, self.cifar_val = random_split(
            cifar_data, [45000, 5000], generator=torch.Generator().manual_seed(42)
        )

        if stage == "test":
            self.cifar_test = CIFAR10(
                self.data_dir, train=False, transform=self.transform_test
            )

        if stage == "predict":
            self.cifar_test = CIFAR10(
                self.data_dir, train=False, transform=self.transform_test
            )

    def train_dataloader(self):
        return DataLoader(
            self.cifar_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=True,
            prefetch_factor=5,
            shuffle=False
        )

    def val_dataloader(self):
        return DataLoader(
            self.cifar_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.cifar_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=True
        )

    def predict_dataloader(self):
        return DataLoader(
            self.cifar_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    @staticmethod
    def _normalize():
        return transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261]
        )
