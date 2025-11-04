import os
from typing import Optional
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from lightning import pytorch as pl

class ImageNetDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 64,
        num_workers: int = 4,
        pin_memory: bool = True,
        train_transforms: Optional[transforms.Compose] = None,
        val_transforms: Optional[transforms.Compose] = None,
    ):
        """
        Args:
            data_dir (str): Path to the root directory containing the `train` and `val` folders.
            batch_size (int): Batch size for data loaders.
            num_workers (int): Number of workers for data loaders.
            pin_memory (bool): Whether to use pinned memory for data loaders.
            train_transforms (transforms.Compose, optional): Data augmentation/transformation for training.
            val_transforms (transforms.Compose, optional): Data augmentation/transformation for validation.
        """
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.train_transforms = train_transforms or self.default_train_transforms()
        self.val_transforms = val_transforms or self.default_val_transforms()

    def default_train_transforms(self):
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def default_val_transforms(self):
        return transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    
    def prepare_data(self):
        print("Preparing!")
        pass

    def setup(self, stage: Optional[str] = None):
        """
        Setup datasets for training, validation, and testing.
        """
        print(f"Stage: {stage}")
        # if stage == "fit" or stage is None:
        self.train_dataset = datasets.ImageFolder(
            root=os.path.join(self.data_dir, "train"),
            transform=self.train_transforms,
        )
        self.val_dataset = datasets.ImageFolder(
            root=os.path.join(self.data_dir, "val"),
            transform=self.val_transforms,
        )
        if stage == "test" or stage is None:
            self.test_dataset = datasets.ImageFolder(
                root=os.path.join(self.data_dir, "val"),
                transform=self.val_transforms,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
