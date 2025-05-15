from lightning import pytorch as pl

from torchvision.datasets import VOCDetection
from torchvision import transforms

from torch.utils.data import DataLoader


class VOCDataModule(pl.LightningDataModule):
    def __init__(
        self, data_dir: str = "./data", batch_size=1000, num_workers=5
    ) -> None:
        super().__init__()
        self.year = "2012"
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = False

        self.transform_train = transforms.Compose(
            [
                transforms.RandomCrop((640, 640), pad_if_needed=True),
                transforms.ToTensor(),
            ]
        )

        self.transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        
        self.target_transform = []

    def prepare_data(self):
        VOCDetection(
            root=self.data_dir, year=self.year, image_set="train", download=True
        )
        VOCDetection(
            root=self.data_dir, year=self.year, image_set="val", download=True
        )

    def setup(self, stage: str):
        self.voc_train = VOCDetection(
            root=self.data_dir,
            year=self.year,
            image_set="train",
            download=False,
            transform=self.transform_train,
        )

        self.voc_test = VOCDetection(
            root=self.data_dir,
            year=self.year,
            image_set="val",
            download=False,
            transform=self.transform_train,
        )
    
    def train_dataloader(self):
        return DataLoader(
            self.voc_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=False,
            pin_memory=False,
            prefetch_factor=None,
            shuffle=self.shuffle
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.voc_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=True,
            prefetch_factor=5,
            shuffle=self.shuffle
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.voc_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=True,
            prefetch_factor=5,
            shuffle=self.shuffle
        )


