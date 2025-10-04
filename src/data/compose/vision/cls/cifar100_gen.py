import os
import pickle
import numpy as np
from PIL import Image
from torchvision.datasets.vision import VisionDataset
from torchvision import transforms
from lightning import pytorch as pl
from torchvision.datasets import CIFAR100

from torch.utils.data import random_split, DataLoader

class CIFAR100GenDataset(VisionDataset):
    def __init__(
        self,
        root="./data/cifar-100-gen",
        train=True,
        transform=None,
        target_transform=None
    ):
        super().__init__(root, transform=transform, target_transform=target_transform)
        # Discover pickle batch files in directory
        self.root = os.path.join(self.root, "train") if train else os.path.join(self.root, "test")  

        batch_files = sorted([
            fname for fname in os.listdir(self.root)
            if fname.endswith('.pkl')
        ])
        if not batch_files:
            raise RuntimeError(f"No .pkl batch files found in {self.root}")

        data_list = []
        targets = []
        for batch_file in batch_files:
            path = os.path.join(self.root, batch_file)
            with open(path, 'rb') as f:
                batch = pickle.load(f, encoding='latin1')
            arr = np.asarray(batch['data'])  # shape (N,3,32,32)
            # Convert to HWC (PIL) format
            arr = arr.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
            data_list.append(arr)
            targets.extend(batch['labels'])
        self.data = np.vstack(data_list)  # shape (total, H, W, C)
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        img = self.data[index]
        target = self.targets[index]
        # img = Image.fromarray(img)
        # img = Image.fromarray(((img / img.max()).clip(0, 1) * 255).round())
        img = Image.fromarray(((img / img.max()).clip(0, 1) * 255).round().astype(np.uint8))
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            target = self.target_transform(target)
        return img, target

class CIFAR100GenDataModule(pl.LightningDataModule):
    def __init__(self,
                 data_dir: str = "./data/cifar-100-gen",
                 batch_size=1000,
                 num_workers=5) -> None:

        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.transform_train = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
                transforms.ToTensor(),
                self._normalize(),
            ]
        )

        self.transform_test = transforms.Compose(
            [transforms.ToTensor(), self._normalize()]
        )

    def prepare_data(self):
        pass
        # CIFAR100GenDataset(self.data_dir, train=True, download=True)
        # CIFAR100GenDataset(self.data_dir, train=False, download=True)

    def setup(self, stage: str):
        cifar_data = CIFAR100GenDataset(
            self.data_dir, train=True, transform=self.transform_train
        )

        self.cifar_train = cifar_data

        # self.cifar_train, self.cifar_val = random_split(
            # cifar_data, [45000, 5000], generator=torch.Generator().manual_seed(42)
        # )

        self.cifar_test = CIFAR100GenDataset(
            self.data_dir, train=False, transform=self.transform_test
        )

        self.cifar_test = CIFAR100(
            self.data_dir, train=False, transform=self.transform_test, download=True
        )


        if stage == "predict":
            self.cifar_test = CIFAR100GenDataset(
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
            mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]
        )

if __name__ == "__main__":
    ds = CIFAR100GenDataset()

    next(iter(ds))
    pass